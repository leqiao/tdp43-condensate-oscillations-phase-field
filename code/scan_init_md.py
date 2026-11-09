#!/usr/bin/env python3
r"""
scan_init_md.py -- does the limit cycle at each oscillating MD set-up depend on the initial
condensate?  (Supplementary Fig. "The limit cycle does not depend on the initial condensate".)

The state diagram is computed from one start, a condensate slab with the enzymes at
equilibrium partitioning ('C_slab', model.initial()).  Each set-up is also started from:

    C  one droplet of 0.25, 0.5, 0.75, 1.0 x the coexistence (lever-rule) size, p = 0
    D  droplet size {0.5, 1.0} x pre-phosphorylation p0 {0.2, 0.4}
    E  enzymes neutrally partitioned (phi_E ~ 1 - phi), or the kinase in the dilute phase only
    F  two (two seeds) or three droplets of unequal size
    G  restarts from the maximum and minimum of the converged cycle (data/md_points)

All starts conserve total protein at phi_bar.  Each run is compared with C_slab on its tail
orbit (period and peak condensed mass).

    python3 scan_init_md.py --procs 30                 # 6 set-ups x 16 starts, T = 150 (90 us)
    python3 scan_init_md.py --points 2:10,3:12
    python3 scan_init_md.py --summary                  # table from data/initial_conditions
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M                                            # noqa: E402

import config as C                                           # noqa: E402

POINTS = [(2, 10), (3, 12), (2, 12), (3, 15), (4, 20), (3, 18)]
CYCLE_DIR = os.path.join(HERE, '..', 'data', 'md_points')
PMIN, PMAX = 1.0, 20.0           # us; period window, as in scan_phase.py


def starts():
    S = []
    S += [('C_slab', dict(kind='slab'))]
    S += [('C_size%g' % f, dict(kind='droplets', n=1, size=f, p=0.0))
          for f in (0.25, 0.5, 0.75, 1.0)]
    S += [('D_size%g_p%g' % (f, p), dict(kind='droplets', n=1, size=f, p=p))
          for f in (0.5, 1.0) for p in (0.2, 0.4)]
    S += [('E_enz_neutral', dict(kind='droplets', n=1, size=1.0, p=0.0, enz='neutral')),
          ('E_kin_dilute', dict(kind='droplets', n=1, size=1.0, p=0.0, enz='kin_dilute'))]
    S += [('F_drop2_s0', dict(kind='droplets', n=2, size=1.0, p=0.0, seed=0)),
          ('F_drop2_s1', dict(kind='droplets', n=2, size=1.0, p=0.0, seed=1)),
          ('F_drop3_s0', dict(kind='droplets', n=3, size=1.0, p=0.0, seed=0))]
    S += [('G_cyc_max', dict(kind='cycle', at='max')),
          ('G_cyc_min', dict(kind='cycle', at='min'))]
    return S


def coex(chi):
    g = lambda q: np.log(q / (1 - q)) + chi * (1 - 2 * q)
    lo = brentq(g, 1e-14, 0.5 - 1e-12, xtol=1e-15)
    return lo, 1.0 - lo


def _name(v):
    return ('%g' % v).replace('.', 'p')


def make(P, N_k, N_p, s):
    N, L, pbar = P['N'], P['L'], P['phi']
    dx = L / N
    x0 = M.initial(P)
    _, _, pK0, pP0 = M.unpack(x0, N)
    kind = s['kind']
    if kind == 'slab':
        return x0
    if kind == 'cycle':
        f = os.path.join(CYCLE_DIR, 'cell_nk%s_np%s.npz' % (_name(N_k), _name(N_p)))
        z = np.load(f, allow_pickle=True)
        m = np.asarray(z['mass'], float)
        i0 = int(0.6 * len(m))
        i = i0 + int(np.argmax(m[i0:]) if s['at'] == 'max' else np.argmin(m[i0:]))
        return np.concatenate([np.asarray(z[k][i], float) for k in M.FIELDS])
    if kind == 'uniform':
        rng = np.random.default_rng(s['seed'])
        tot = pbar * (1 + s['eps'] * rng.standard_normal(N))
        tot += pbar - tot.mean()
        p = s['p']
        return np.concatenate([(1 - p) * tot, np.maximum(p * tot, 1e-6),
                               np.full(N, pK0.mean()), np.full(N, pP0.mean())])
    if kind == 'droplets':
        p, n = s['p'], s['n']
        pd0, pc = coex((1 - p) * P['c0s'] + p * P['c1s'])
        frac = (pbar - pd0) / (pc - pd0) * s['size']
        pd = (pbar - frac * pc) / (1 - frac)       # raise the dilute arm to conserve mass
        w = M.tanh_width(P)
        xg = (np.arange(N) + 0.5) * dx
        rng = np.random.default_rng(s.get('seed', 0))
        if n == 1:
            widths, centres = np.array([L * frac]), np.array([0.5 * L])
        else:
            share = rng.uniform(0.6, 1.4, n)
            widths = L * frac * share / share.sum()
            centres = (rng.uniform() + np.arange(n) + rng.uniform(-0.15, 0.15, n)) * L / n
        widths = np.clip(widths, 4 * dx, None)
        prof = np.zeros(N)
        for wd, c in zip(widths, centres):
            d = np.abs((xg - c + 0.5 * L) % L - 0.5 * L)          # periodic distance
            prof = np.maximum(prof, 0.5 * (np.tanh((wd / 2 - d) / w) + 1.0))
        tot = pd + (pc - pd) * prof
        tot += pbar - tot.mean()
        a0, a1 = (1 - p) * tot, np.maximum(p * tot, 1e-6)
        enz = s.get('enz', 'equilibrium')
        if enz == 'equilibrium':
            pK, pP = M.seed_enzymes(P, a0, a1)
        elif enz == 'neutral':                     # no affinity: follow the free volume
            free = 1.0 - tot
            pK, pP = (free * pK0.mean() / free.mean(), free * pP0.mean() / free.mean())
        elif enz == 'kin_dilute':                  # kinase only where the protein is dilute
            _, pP = M.seed_enzymes(P, a0, a1)
            pK = (1 - prof) + 1e-6
            pK *= pK0.mean() / pK.mean()
        return np.concatenate([a0, a1, pK, pP])
    raise ValueError(kind)


def one(job):
    N_k, N_p, label, s, T, out_dir = job
    fn = os.path.join(out_dir, 'run_nk%s_np%s_%s.npz' % (_name(N_k), _name(N_p), label))
    if os.path.exists(fn):
        return summarise_file(fn)
    P = C.params(M, N_k=N_k, N_p=N_p)
    grids = M.grid(P['N'], P['L'])
    N, L, thr = P['N'], P['L'], P['thr']
    x = make(P, N_k, N_p, s)
    nsteps = int(round(T / P['dt']))
    stride = max(1, nsteps // 2000)
    fstride = max(1, nsteps // 300)
    rec = {k: [] for k in ('t', 'mass', 'pfrac', 'Kin', 'Pin')}
    frames = {f: [] for f in M.FIELDS}
    x_init = x.copy()
    diverged = False
    for n in range(nsteps + 1):
        if n % stride == 0:
            p0, p1, pK, pP = M.unpack(x, N)
            if not np.isfinite(x).all():
                diverged = True
                break
            tot = p0 + p1
            Kr, Pr = M.enzyme_split(p0, p1, pK, pP, thr)
            rec['t'].append(n * P['dt'])
            rec['mass'].append(float(np.sum(np.where(tot > thr, tot, 0.0)) * L / N))
            rec['pfrac'].append(float(p1.sum() / max(tot.sum(), 1e-12)))
            rec['Kin'].append(Kr); rec['Pin'].append(Pr)
        if n % fstride == 0:
            for f, v in zip(M.FIELDS, M.unpack(x, N)):
                frames[f].append(v.astype(np.float32))
        if n < nsteps:
            x = M.step(x, P, grids)
    out = {k: np.array(v) for k, v in rec.items()}
    c, d = M.classify(out)
    cls = 'diverged' if diverged else M.CLASS[c]
    per = d['period'] * M.TSTAR_US if np.isfinite(d['period']) else np.nan
    if cls == 'oscillating' and np.isfinite(per) and not (PMIN < per < PMAX):
        cls = 'steady'
    np.savez_compressed(fn, **out, **{f: np.array(v) for f, v in frames.items()},
                        x_init=x_init, frame_stride=fstride, dt=P['dt'],
                        N_k=N_k, N_p=N_p, label=label, start=repr(s), T=T,
                        cls=cls, period=per, amp=d['amp'], pbar=d['pbar'],
                        m0=out['mass'][0] if len(out['mass']) else np.nan,
                        Tstar=M.TSTAR_US, c0K=P['c0K'], LamK=P['LamK'])
    return summarise_file(fn)


def summarise_file(fn):
    z = np.load(fn, allow_pickle=True)
    m = np.asarray(z['mass'], float)
    cls = str(z['cls'])
    # flag runs that break conservation or drive the solvent negative
    tot = np.asarray(z['phi0'], float) + np.asarray(z['phi1'], float)
    ps = 1 - tot - np.asarray(z['phiK'], float) - np.asarray(z['phiP'], float)
    if np.abs(tot.mean(1) - tot[0].mean()).max() > 1e-3 or ps.min() < -1e-6:
        cls = 'numerical'
    return dict(N_k=float(z['N_k']), N_p=float(z['N_p']), label=str(z['label']),
                cls=cls, period=float(z['period']), amp=float(z['amp']),
                pbar=float(z['pbar']), m0=float(z['m0']),
                mmax=float(m[int(0.6 * len(m)):].max()) if len(m) else np.nan)


def verdict(r, ref, ptol=0.03, mtol=0.015):
    # same cycle: period and peak mass within tolerance of the reference
    if r['cls'] != 'oscillating':
        return r['cls']
    if ref is None or ref['cls'] != 'oscillating':
        return 'osc (no ref)'
    same = (abs(r['period'] / ref['period'] - 1) < ptol
            and abs(r['mmax'] / ref['mmax'] - 1) < mtol)
    return 'same cycle' if same else 'OTHER cycle'


def report(res, points):
    labels = [l for l, _ in starts()]
    print('\n%-16s' % 'start' + ''.join('%-24s' % ('(%g,%g)' % pt) for pt in points))
    by = {(r['N_k'], r['N_p'], r['label']): r for r in res}
    for l in labels:
        row = '%-16s' % l
        for k, p in points:
            r, ref = by.get((k, p, l)), by.get((k, p, 'C_slab'))
            if r is None:
                row += '%-24s' % '-'
                continue
            v = verdict(r, ref)
            per = '%.2f' % r['period'] if r['cls'] == 'oscillating' else ''
            row += '%-24s' % ('%s %s' % (v, per)).strip()
        print(row)
    print('\nreference (C_slab): ' + '   '.join(
        '(%g,%g) T=%.2f us mmax=%.3f pbar=%.3f' % (k, p, by[(k, p, 'C_slab')]['period'],
                                                  by[(k, p, 'C_slab')]['mmax'],
                                                 by[(k, p, 'C_slab')]['pbar'])
        for k, p in points if (k, p, 'C_slab') in by))
    print('\nper point:')
    for k, p in points:
        rr = [r for r in res if (r['N_k'], r['N_p']) == (k, p)]
        if not rr:
            continue
        ref = by.get((k, p, 'C_slab'))
        vs = [verdict(r, ref) for r in rr]
        n_same = vs.count('same cycle')
        bad = ['%s:%s' % (r['label'], v) for r, v in zip(rr, vs) if v != 'same cycle']
        print('  (%g,%g)  %d/%d starts reach the slab cycle   %s'
              % (k, p, n_same, len(rr), ', '.join(bad) if bad else 'IC-INDEPENDENT'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--T', type=float, default=150.0, help='code units (x0.6 us); 150 = production')
    ap.add_argument('--procs', type=int, default=30)
    ap.add_argument('--points', default=None, help='comma-separated Nk:Np, e.g. 2:10,3:12')
    ap.add_argument('--only', default=None, help='comma-separated start labels to run')
    ap.add_argument('--families', default=None,
                    help="start families to run, e.g. CDEFG (the condensed starts, no uniform)")
    ap.add_argument('--out', default=None)
    ap.add_argument('--summary', action='store_true', help='only re-print from saved runs')
    a = ap.parse_args()
    points = POINTS if a.points is None else [tuple(float(v) for v in c.split(':'))
                                              for c in a.points.split(',')]
    points = [(int(k) if k == int(k) else k, int(p) if p == int(p) else p) for k, p in points]
    out_dir = a.out or os.path.join(HERE, '..', 'data', 'initial_conditions')
    os.makedirs(out_dir, exist_ok=True)
    S = starts()
    if a.only:
        keep = set(a.only.split(','))
        S = [(l, s) for l, s in S if l in keep]
    if a.families:
        S = [(l, s) for l, s in S if l[0] in a.families]
    if a.summary:
        res = [summarise_file(f) for f in sorted(glob.glob(os.path.join(out_dir, 'run_*.npz')))]
        report(res, points)
        return
    print(__doc__)
    print('config  : %s   c0K %.2f   LamK %.5f' % (C.__file__, C.EXPLICIT['c0K'], C.EXPLICIT['LamK']))
    print('out     : %s' % out_dir)
    jobs = [(k, p, l, s, a.T, out_dir) for k, p in points for l, s in S]
    print('runs    : %d points x %d starts = %d   T = %g (%.0f us)\n'
          % (len(points), len(S), len(jobs), a.T, a.T * M.TSTAR_US), flush=True)
    res = []
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(one, jobs):
            res.append(r)
            print('  done (%g,%g) %-16s %-12s T=%5.2f us  amp %.3f  pbar %.3f  m0 %.3f'
                  % (r['N_k'], r['N_p'], r['label'], r['cls'], r['period'], r['amp'],
                     r['pbar'], r['m0']), flush=True)
    report(res, points)


if __name__ == '__main__':
    main()
