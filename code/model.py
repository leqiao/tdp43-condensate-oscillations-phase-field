#!/usr/bin/env python3
r"""
model.py -- 1D phase-field model of a TDP-43 condensate with a kinase and a phosphatase.

Fields (periodic; phi_s = 1 - phi0 - phi1 - phiK - phiP):
    phi0  unphosphorylated protein (A)    phi1  phosphorylated protein (B)
    phiK  kinase                          phiP  phosphatase
Protein, kinase and phosphatase are each conserved.

    python3 model.py --T 200 --plot
    python3 model.py --check          # zero driving: F must decrease monotonically
    python3 model.py --where          # enzyme partitioning at fixed p, chemistry off
    overrides: any DEFAULT key, e.g. --c1K --Ktot --kkin --dmu
"""
import os
import sys

import numpy as np

PHI = 1e-9
E_CLIP = 60.0

DEFAULT = dict(
    N=151, L=1.16, dt=3.5e-6, nsave=4000,    # grid, box, converged time step
    # Flory-Huggins and gradient coefficients (0 = A, 1 = B, K, P, s)
    c0s=3.5, c1s=0.0,
    kappa0=2.5e-3, kappa1=2.5e-3, kapK=2.5e-3, kapP=2.5e-3,
    w0=0.0, w1=4.0, wK=0.0, wP=0.0, ws=0.0,  # internal free energies
    c0K=0.0, c1K=-8.0,                       # kinase-protein
    c0P=5.0, c1P=6.0,                        # phosphatase-protein
    # mobilities (Lambda_A = 1 sets the time unit)
    Lam0=1.0, Lam1=1.0, LamK=8e-3, LamP=8e-3,
    chi_dag_k=0.0, chi_dag_p=0.0,            # transition-state interactions (off)
    retard=0.0, retard_amp=1.0,              # binding-retarded enzyme mobility (off)
    KMk=1e6, KMp=1e6,                        # Michaelis constants; >= 1e6 is mass action
    # reactions: bare rate constants and driving (code sign: dmu = -dmu_k, dmup = -dmu_p)
    kkin=0.6, kpho=18.0, dmu=5.0, dmup=0.0,
    # composition
    phi=0.30, Ktot=8.793e-3, Ptot=2.0e-2,    # mean protein, kinase, phosphatase fractions
    phi_dil=0.05, phi_dense=0.90, thr=0.35,  # initial slab; condensate threshold phi_cut
)

ALPHA_NK = 0.30 * 292 / (200 * 154)   # volume fraction per kinase
TSTAR_US = 85.93 ** 2 / 1.23e4        # time unit T* = L*^2 / D_lcd = 0.6003 us
NSER = 24
FIELDS = ('phi0', 'phi1', 'phiK', 'phiP')


def grid(N, L):
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    return k, k ** 2


def ddx(f, k):
    return np.real(np.fft.ifft(1j * k * np.fft.fft(f)))


def lap(f, ksq):
    return np.real(np.fft.ifft(-ksq * np.fft.fft(f)))


def unpack(x, N):
    return x[:N], x[N:2 * N], x[2 * N:3 * N], x[3 * N:]


def E(x):
    return np.exp(np.clip(x, -E_CLIP, E_CLIP))


def pots(p0, p1, pK, pP, P, grids):
    k, ksq = grids
    ps = 1 - p0 - p1 - pK - pP
    a0, a1 = np.maximum(p0, PHI), np.maximum(p1, PHI)
    aK, aP = np.maximum(pK, PHI), np.maximum(pP, PHI)
    asv = np.maximum(ps, PHI)
    mu0 = (np.log(a0) + P['w0'] + P['c0s'] * ps + P['c0K'] * pK + P['c0P'] * pP
           - P['kappa0'] * lap(p0, ksq))
    mu1 = (np.log(a1) + P['w1'] + P['c1s'] * ps + P['c1K'] * pK + P['c1P'] * pP
           - P['kappa1'] * lap(p1, ksq))
    muK = (np.log(aK) + P['wK'] + P['c0K'] * p0 + P['c1K'] * p1
           - P['kapK'] * lap(pK, ksq))
    muP = (np.log(aP) + P['wP'] + P['c0P'] * p0 + P['c1P'] * p1
           - P['kapP'] * lap(pP, ksq))
    mus = np.log(asv) + P['ws'] + P['c0s'] * p0 + P['c1s'] * p1
    return mu0, mu1, muK, muP, mus, ps


def free_energy(x, P, grids):
    """Free energy F (Eq. F of the Methods); decreases monotonically at zero driving."""
    k, ksq = grids
    N, L = P['N'], P['L']
    p0, p1, pK, pP = unpack(x, N)
    ps = 1 - p0 - p1 - pK - pP

    def s(a, w):
        aa = np.maximum(a, PHI)
        return aa * (np.log(aa) - 1.0 + w)

    f = (s(p0, P['w0']) + s(p1, P['w1']) + s(pK, P['wK']) + s(pP, P['wP']) + s(ps, P['ws'])
         + P['c0s'] * p0 * ps + P['c1s'] * p1 * ps
         + P['c0K'] * p0 * pK + P['c1K'] * p1 * pK
         + P['c0P'] * p0 * pP + P['c1P'] * p1 * pP
         - 0.5 * P['kappa0'] * p0 * lap(p0, ksq) - 0.5 * P['kappa1'] * p1 * lap(p1, ksq)
         - 0.5 * P['kapK'] * pK * lap(pK, ksq) - 0.5 * P['kapP'] * pP * lap(pP, ksq))
    return float(np.sum(f) * L / N)


def rhs(x, P, grids):
    k, ksq = grids
    N = P['N']
    p0, p1, pK, pP = unpack(x, N)
    mu0, mu1, muK, muP, mus, ps = pots(p0, p1, pK, pP, P, grids)

    def flux(Lam, p, mu):
        return ddx(Lam * np.maximum(p, 0) * ddx(mu - mus, k), k)

    # optional binding-retarded mobility, Lam exp(min(0, c0 phi0 + c1 phi1)); off by default
    def lam_ret(Lam, c0, c1):
        if not P.get('retard', 0.0):
            return Lam
        return Lam * np.exp(P.get('retard_amp', 1.0)
                            * np.minimum(0.0, c0 * np.maximum(p0, 0) + c1 * np.maximum(p1, 0)))

    gdk = E(-P.get('chi_dag_k', 0.0) * ps)
    gdp = E(-P.get('chi_dag_p', 0.0) * ps)
    a0e, a1e = E(mu0), E(mu1)
    # reaction rates, Eq. (rates); sat = 1 for mass action
    satk = 1.0 + gdk * a0e / P['KMk']
    satp = 1.0 + gdp * a1e / P['KMp']
    rk = P['kkin'] * np.maximum(pK, 0) * gdk * (E(mu0 + P['dmu']) - E(mu1)) / satk
    rp = P['kpho'] * np.maximum(pP, 0) * gdp * (E(mu1 - P['dmup']) - E(mu0)) / satp

    return np.concatenate([flux(P['Lam0'], p0, mu0) - rk + rp,
                           flux(P['Lam1'], p1, mu1) + rk - rp,
                           flux(lam_ret(P['LamK'], P['c0K'], P['c1K']), pK, muK),
                           flux(lam_ret(P['LamP'], P['c0P'], P['c1P']), pP, muP)])


def step(x, P, grids):
    """One semi-implicit step: the gradient term implicitly, the rest explicitly."""
    k, ksq = grids
    N, dt = P['N'], P['dt']
    k4 = ksq ** 2
    A = (P['Lam0'] * P['kappa0'], P['Lam1'] * P['kappa1'],
         P['LamK'] * P['kapK'], P['LamP'] * P['kapP'])
    f = rhs(x, P, grids)
    out = []
    for j in range(4):
        xi, fi = x[j * N:(j + 1) * N], f[j * N:(j + 1) * N]
        if A[j] <= 0.0:
            out.append(xi + dt * fi)
            continue
        F0 = np.fft.fft(xi)
        out.append(np.real(np.fft.ifft(
            (F0 + dt * (np.fft.fft(fi) + A[j] * k4 * F0)) / (1 + dt * A[j] * k4))))
    return np.clip(np.concatenate(out), PHI, 1 - 5 * PHI)


def tanh_width(P):
    """Interface half-width xi of the binary A-solvent system (Eq. xi)."""
    from scipy.optimize import brentq
    chi = P['c0s']
    f = lambda q: q * np.log(q) + (1 - q) * np.log(1 - q) + chi * q * (1 - q)
    fp = lambda q: np.log(q / (1 - q)) + chi * (1 - 2 * q)
    pd = brentq(fp, 1e-14, 0.5 - 1e-12); pD = 1 - pd
    mu = (f(pD) - f(pd)) / (pD - pd)
    return 0.5 * (pD - pd) * np.sqrt(P['kappa0'] / (2 * (f(0.5) - (f(pd) + mu * (0.5 - pd)))))


def seed_enzymes(P, p0, p1, iters=200):
    """Enzyme profiles at equilibrium partitioning for a given protein profile.

    phi_E = lambda_E phi_s exp[(c0s - c0E) phi0 + (c1s - c1E) phi1], with phi_s solved
    self-consistently and lambda_E fixed by the enzyme total.
    """
    gK = np.exp(np.clip((P['c0s'] - P['c0K']) * p0 + (P['c1s'] - P['c1K']) * p1, -E_CLIP, E_CLIP))
    gP = np.exp(np.clip((P['c0s'] - P['c0P']) * p0 + (P['c1s'] - P['c1P']) * p1, -E_CLIP, E_CLIP))
    if P.get('seed', 'equilibrium') == 'legacy':
        return P['Ktot'] * gK / gK.mean(), P['Ptot'] * gP / gP.mean()
    ps0 = np.maximum(1.0 - p0 - p1, PHI)
    lK = P['Ktot'] / np.mean(ps0 * gK)
    lP = P['Ptot'] / np.mean(ps0 * gP)
    for _ in range(iters):
        ps = ps0 / (1.0 + lK * gK + lP * gP)
        nK, nP = P['Ktot'] / np.mean(ps * gK), P['Ptot'] / np.mean(ps * gP)
        if abs(nK / lK - 1) < 1e-12 and abs(nP / lP - 1) < 1e-12:
            break
        lK, lP = nK, nP
    ps = ps0 / (1.0 + lK * gK + lP * gP)
    return lK * ps * gK, lP * ps * gP


def initial(P):
    """Condensate slab (tanh profile), unphosphorylated, enzymes at equilibrium."""
    N, L = P['N'], P['L']
    dx = L / N
    w = tanh_width(P)
    pd, pc, pbar = P['phi_dil'], P['phi_dense'], P['phi']
    width = min(max(L * (pbar - pd) / (pc - pd), 4 * dx), L - 4 * w)
    xg = (np.arange(N) + 0.5) * dx - 0.5 * L
    prof = 0.5 * (np.tanh((width / 2 - np.abs(xg)) / w) + 1.0)
    p0 = pd + (pc - pd) * prof
    p0 = p0 + (pbar - p0.mean())
    p1 = np.full(N, 1e-6)
    pKf, pPf = seed_enzymes(P, p0, p1)
    return np.concatenate([p0, p1, pKf, pPf])


def enzyme_split(p0, p1, pK, pP, thr=0.35):
    """Mean enzyme volume fraction inside the dense phase vs outside."""
    tot = p0 + p1
    dense = tot > thr
    if dense.sum() == 0 or (~dense).sum() == 0:
        return np.nan, np.nan
    return (float(pK[dense].mean() / max(pK[~dense].mean(), 1e-30)),
            float(pP[dense].mean() / max(pP[~dense].mean(), 1e-30)))


def run(P, T, quiet=False, track_F=True):
    P = dict(P)
    grids = grid(P['N'], P['L'])
    N, thr = P['N'], P['thr']
    x = initial(P)
    nsteps = int(round(T / P['dt']))
    stride = max(1, nsteps // int(P['nsave']))
    rec = {kk: [] for kk in ('t', 'mass', 'pfrac', 'maxphi0', 'F',
                             'protein', 'kinase', 'phosphatase', 'Kin', 'Pin')}
    frames = {f: [] for f in FIELDS}
    for n in range(nsteps + 1):
        if n % stride == 0:
            p0, p1, pK, pP = unpack(x, N)
            tot = p0 + p1
            Kr, Pr = enzyme_split(p0, p1, pK, pP, thr)
            rec['t'].append(n * P['dt'])
            # condensed mass M = int_{phi > thr} phi dx, phi = phi0 + phi1
            rec['mass'].append(float(np.sum(np.where(tot > thr, tot, 0.0)) * P['L'] / N))
            rec['pfrac'].append(float(p1.sum() / max(tot.sum(), PHI)))
            rec['maxphi0'].append(float(p0.max()))
            rec['protein'].append(float(tot.sum() * P['L'] / N))
            rec['kinase'].append(float(pK.sum() * P['L'] / N))
            rec['phosphatase'].append(float(pP.sum() * P['L'] / N))
            rec['Kin'].append(Kr)     # enzyme enrichment, dense / dilute
            rec['Pin'].append(Pr)
            rec['F'].append(free_energy(x, P, grids) if track_F else np.nan)
            for f, fld in zip(FIELDS, (p0, p1, pK, pP)):
                frames[f].append(fld.copy())
        if n < nsteps:
            x = step(x, P, grids)
        if not np.isfinite(x).all():
            if not quiet:
                print("!! diverged at step %d (t=%.4f)" % (n, n * P['dt']))
            break
    out = {kk: np.array(v) for kk, v in rec.items()}
    for f in FIELDS:
        out[f] = np.array(frames[f])
    for kk in ('N', 'L', 'dt', 'kkin', 'kpho', 'dmu', 'dmup', 'Ktot', 'Ptot',
               'chi_dag_k', 'chi_dag_p',
               'c0K', 'c1K', 'c0P', 'c1P', 'phi'):
        out[kk] = np.float64(P[kk])
    return out


def classify(out, amp_tol=2e-3, mass_tol=1e-4, tail=0.4):
    """Classify a run from M(t) over the last `tail` fraction (see Phase Classification).

    Returns (code, info): 0 dissolved, 1 steady, 2 oscillating, 3 diverged.
    """
    t, m = out['t'], out['mass']
    i0 = int(len(t) * (1 - tail))
    tt, mm = t[i0:], m[i0:]
    d = dict(amp=np.nan, period=np.nan, pbar=np.nan, sustain=np.nan, Kin=np.nan, Pin=np.nan)
    if len(mm) < 8 or not np.isfinite(mm).all():
        return 3, d
    h = slice(i0, None)
    d['pbar'] = float(np.nanmean(out['pfrac'][h]))
    d['Kin'] = float(np.nanmedian(out['Kin'][h]))
    d['Pin'] = float(np.nanmedian(out['Pin'][h]))
    if mm.max() < mass_tol:
        d['amp'] = 0.0
        return 0, d
    d['amp'] = float(mm.max() - mm.min())
    # period: first autocorrelation maximum of M
    z = mm - mm.mean()
    cr = np.where((z[:-1] < 0) & (z[1:] >= 0))[0]          # kept for the sustain test
    if z.size > 8 and np.any(z):
        ac = np.correlate(z, z, mode='full')[z.size - 1:]
        ac = ac / ac[0]
        neg = np.nonzero(ac < 0)[0]
        if neg.size:
            k = int(neg[0] + np.argmax(ac[neg[0]:]))
            if k < z.size - 1:
                d['period'] = float(tt[k] - tt[0])
    n4 = max(1, len(m) // 4)
    a2 = float(m[n4:2 * n4].max() - m[n4:2 * n4].min())
    a4 = float(m[3 * n4:].max() - m[3 * n4:].min())
    d['sustain'] = a4 / a2 if a2 > 0 else np.nan
    ok = (a4 > amp_tol) and (a2 <= 0 or 0.5 < a4 / a2 < 2.0)
    if d['amp'] > amp_tol and len(cr) >= 2 and ok:
        return 2, d
    return 1, d


CLASS = ['dissolved', 'steady', 'oscillating', 'diverged']


def summarise(out, label=""):
    c, d = classify(out)
    per = d['period']
    print("%-20s %-11s amp=%.3e sus=%s per=%s%s pbar=%.2f pSer  K_in/out=%.2f P_in/out=%.3f"
          % (label, CLASS[c], d['amp'],
             "%.3f" % d['sustain'] if np.isfinite(d['sustain']) else " n/a ",
             "%.3f" % per if np.isfinite(per) else "n/a",
             " (%.2f us)" % (per * TSTAR_US) if np.isfinite(per) and c == 2 else "",
             NSER * d['pbar'], d['Kin'], d['Pin']))
    return c, d


def check_zero_affinity(P, T=20.0):
    """Zero driving: F must not rise, and every species must be conserved."""
    Q = dict(P); Q.update(dmu=0.0, dmup=0.0)
    o = run(Q, T, quiet=True)
    F = o['F'][np.isfinite(o['F'])]
    dF = np.diff(F)
    print("zero affinity: F %.8f -> %.8f   rises %d/%d (max %.2e)"
          % (F[0], F[-1], int((dF > 0).sum()), len(dF),
             dF.max() if (dF > 0).any() else 0.0))
    print("   conservation: protein %+.1e  kinase %+.1e  phosphatase %+.1e"
          % (o['protein'][-1] - o['protein'][0], o['kinase'][-1] - o['kinase'][0],
             o['phosphatase'][-1] - o['phosphatase'][0]))
    return o


def where_are_the_enzymes(P, T=20.0, plist=(0.0, 0.10, 0.26, 0.40)):
    """Enzyme partitioning at fixed p with the chemistry off."""
    print("enzyme partitioning at fixed p, chemistry off.  ratio = <phi_E>dense / <phi_E>dilute")
    print("%-7s %-9s %-14s %-14s %-10s" % ("p", "pSer", "kinase in/out", "phosphatase", "max phi0"))
    grids = grid(P['N'], P['L'])
    for p in plist:
        Q = dict(P); Q.update(kkin=0.0, kpho=0.0, dmu=0.0, dmup=0.0)
        x = initial(Q)
        N = Q['N']
        p0i = x[:N] + x[N:2 * N]
        x = np.concatenate([(1 - p) * p0i, p * p0i, x[2 * N:3 * N], x[3 * N:]])
        for _ in range(int(T / Q['dt'])):
            x = step(x, Q, grids)
        a0, a1, aK, aP = unpack(x, N)
        Kr, Pr = enzyme_split(a0, a1, aK, aP, Q['thr'])
        print("%-7.2f %-9.2f %-14.3f %-14.4f %-10.4f"
              % (p, NSER * p, Kr, Pr, a0.max()))


def parse(argv):
    P = dict(DEFAULT)
    T, mode, vals, plot = 100.0, 'run', None, False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--T':
            T = float(argv[i + 1]); i += 2; continue
        if a == '--plot':
            plot = True; i += 1; continue
        if a == '--check':
            mode = 'check'; i += 1; continue
        if a == '--where':
            mode = 'where'; i += 1; continue
        if a == '--scan-kpho':
            mode = 'scan'; vals = [float(v) for v in argv[i + 1].split(',')]; i += 2; continue
        key = a.lstrip('-')
        if key in P:
            P[key] = int(argv[i + 1]) if key in ('N', 'nsave') else float(argv[i + 1])
            i += 2
        else:
            print("unknown flag: %s" % a); i += 1
    return P, T, mode, vals, plot


def plot_run(out, P, tag='run'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = out['t']
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(t, out['mass']); ax[0, 0].set_ylabel('condensate mass')
    ax[0, 1].plot(t, out['pfrac'] * NSER, color='#8B008B'); ax[0, 1].set_ylabel('pSer per chain')
    ax[1, 0].plot(t, out['Kin'], label='kinase', color='#D55E00')
    ax[1, 0].plot(t, out['Pin'], label='phosphatase', color='#0072B2')
    ax[1, 0].axhline(1, color='0.6', lw=0.8)
    ax[1, 0].set_ylabel('enrichment dense/dilute'); ax[1, 0].set_yscale('log')
    ax[1, 0].legend(fontsize=8, frameon=False)
    im = ax[1, 1].imshow(out['phi0'].T, aspect='auto', origin='lower', cmap='viridis',
                         extent=[t[0], t[-1], 0, float(out['L'])])
    ax[1, 1].set_ylabel('x'); fig.colorbar(im, ax=ax[1, 1], label=r'$\phi_0$')
    for a in ax.ravel():
        a.set_xlabel('t  (1 unit = %.4g us)' % TSTAR_US)
    fig.suptitle('c0K=%.2g c1K=%.2g | c0P=%.2g c1P=%.2g | kkin=%.3g kpho=%.3g'
                 % (P['c0K'], P['c1K'], P['c0P'], P['c1P'], P['kkin'], P['kpho']), fontsize=10)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    p = os.path.join('figures', '%s.pdf' % tag)
    fig.savefig(p); plt.close(fig)
    print("  wrote %s" % p)


def main(argv):
    P, T, mode, vals, plot = parse(argv)
    if mode == 'check':
        check_zero_affinity(P, T=min(T, 20.0))
    elif mode == 'where':
        where_are_the_enzymes(P, T=min(T, 20.0))
    elif mode == 'scan':
        for v in vals:
            Q = dict(P); Q['kpho'] = v
            o = run(Q, T, quiet=True)
            summarise(o, "kpho=%-8.3g" % v)
            if plot:
                plot_run(o, Q, tag='kpho%g' % v)
    else:
        o = run(P, T)
        summarise(o, "run")
        if plot:
            plot_run(o, P)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
