#!/usr/bin/env python3
"""
scan_phase.py -- state diagram over the enzyme numbers (N_k, N_p).

    python3 scan_phase.py --nproc 16 --out phase_nk_np.npz

Enzyme numbers are converted to volume fractions with config.alphas; all rate constants are
fixed across the plane.
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

import model as M
import config as C


DX = 0.5 / 64
LSTAR = float(np.sqrt(M.TSTAR_US * 1.23e4 / M.DEFAULT['Lam0']))   # 85.93 nm


def _cell_name(N_k, N_p):
    """Filename-safe cell id: nk3p25 means N_k = 3.25.  Keeps sorting sane and avoids
    dots in names, which some tools truncate at."""
    g = lambda v: ('%g' % v).replace('.', 'p')
    return 'nk%s_np%s' % (g(N_k), g(N_p))


def _save_cell(P, out, c, d, a, N_k, N_p):
    """One .npz per cell, holding the four density fields so any cell can be re-examined
    later without rerunning the scan.  Frames are thinned to --save-frames: the full
    nsave=2000 is ~10 MB per cell and 285 cells would be 2-3 GB, while a few hundred
    frames is enough to see the profiles and the cycle."""
    import os
    os.makedirs(a['save_dir'], exist_ok=True)
    keep = max(1, len(out['t']) // a['save_frames'])
    dat = {k: (v[::keep] if getattr(v, 'ndim', 0) else v)
           for k, v in out.items() if isinstance(v, np.ndarray)}
    fn = os.path.join(a['save_dir'], 'cell_%s.npz' % _cell_name(N_k, N_p))
    np.savez_compressed(fn, **dat,
                        **{('P_' + k): v for k, v in P.items()
                           if isinstance(v, (int, float))},
                        N_k=N_k, N_p=N_p, cls=M.CLASS[c], Tstar=M.TSTAR_US,
                        Lstar=LSTAR, thin=keep,
                        period=d['period'] * M.TSTAR_US if np.isfinite(d['period']) else np.nan)
    return fn


def _run_from(P, T, kind):
    """model.run() always starts from model.initial(); this repeats its recording loop from
    a different state.  Kept minimal -- only the fields classify() and the cell files need."""
    grids = M.grid(P['N'], P['L'])
    x = _start(P, kind)
    N, L, thr = P['N'], P['L'], P['thr']
    nsteps = int(T / P['dt'])
    stride = max(1, nsteps // P.get('nsave', 2000))
    rec = {k: [] for k in ('t', 'mass', 'pfrac', 'maxphi0', 'protein', 'kinase',
                           'phosphatase', 'Kin', 'Pin', 'F')}
    frames = {f: [] for f in M.FIELDS}
    for n in range(nsteps + 1):
        if n % stride == 0:
            p0, p1, pK, pP = M.unpack(x, N)
            tot = p0 + p1
            Kr, Pr = M.enzyme_split(p0, p1, pK, pP, thr)
            rec['t'].append(n * P['dt'])
            rec['mass'].append(float(np.sum(np.where(tot > thr, tot, 0.0)) * L / N))
            rec['pfrac'].append(float(p1.sum() / max(tot.sum(), 1e-12)))
            rec['maxphi0'].append(float(p0.max()))
            rec['protein'].append(float(tot.sum() * L / N))
            rec['kinase'].append(float(pK.sum() * L / N))
            rec['phosphatase'].append(float(pP.sum() * L / N))
            rec['Kin'].append(Kr); rec['Pin'].append(Pr); rec['F'].append(np.nan)
            for f, v in zip(M.FIELDS, (p0, p1, pK, pP)):
                frames[f].append(v.copy())
        x = M.step(x, P, grids)
    out = {k: np.array(v) for k, v in rec.items()}
    out.update({f: np.array(v) for f, v in frames.items()})
    return out


def _start(P, kind):
    """Initial condition.  'slab' is model.initial(): a tanh slab at the coexistence
    compositions with each enzyme pre-seeded at its equilibrium partitioning.  'uniform' is
    the dispersed state with a small perturbation, and 'cycle' a slab already 25% converted.
    Running the grid from two of these and comparing is how the bistable region is found --
    at N_k >~ 4 a dissolved fixed point and a limit cycle coexist, so a single initial
    condition draws a boundary that is really a basin boundary."""
    x = M.initial(P)
    if kind == 'slab':
        return x
    N = P['N']
    p0, p1, pK, pP = M.unpack(x, N)
    if kind == 'uniform':
        tot = (p0 + p1).mean()
        rng = np.random.default_rng(0)
        return np.concatenate([tot * (1 + 1e-3 * rng.standard_normal(N)),
                               np.full(N, 1e-6), np.full(N, pK.mean()),
                               np.full(N, pP.mean())])
    if kind == 'cycle':
        tot = p0 + p1
        pK, pP = M.seed_enzymes(P, 0.75 * tot, 0.25 * tot)
        return np.concatenate([0.75 * tot, 0.25 * tot, pK, pP])
    raise ValueError(kind)


def _one(job):
    N_k, N_p, a = job
    over = {} if a.get('c0K') is None else dict(c0K=a['c0K'])
    if a.get('retard'):                       # else config.KIN_RETARD
        over['LamK'] = C.LAMK_ROUSE / a['retard']
    P = C.params(M, N_k=N_k, N_p=N_p, K_pho=a['kpho'] * a['rate_scale'], nsave=a['nsave'],
                 N=a['N'], L=a['L'], **over)
    P['kkin'] *= a['rate_scale']
    try:
        out = (M.run(P, a['T'], quiet=True, track_F=False) if a.get('init', 'slab') == 'slab'
               else _run_from(P, a['T'], a['init']))
        c, d = M.classify(out)
        h = len(out['mass']) // 2
        per = d['period'] * M.TSTAR_US if np.isfinite(d['period']) else np.nan
        # oscillations with a period outside (pmin, pmax) us count as steady
        cls = M.CLASS[c]
        if cls == 'oscillating' and np.isfinite(per) and not (a['pmin'] < per < a['pmax']):
            cls = 'steady'
        if a['save'] == 'all' or (a['save'] == 'osc' and cls == 'oscillating'):
            _save_cell(P, out, c, d, a, N_k, N_p)
        return dict(N_k=N_k, N_p=N_p, cls=cls,
                    period=per,
                    pbar=float(np.nanmean(out['pfrac'][h:])),
                    amp=d['amp'], mass=float(np.nanmean(out['mass'][h:])))
    except Exception as e:
        return dict(N_k=N_k, N_p=N_p, cls='error', period=np.nan, pbar=np.nan,
                    amp=np.nan, mass=np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # default grid: N_k step 0.25, N_p step 0.5 (the published diagram uses N_p step 1)
    ap.add_argument('--N_k', default=','.join('%g' % (1.5 + 0.25 * i) for i in range(15)))
    ap.add_argument('--N_p', default=','.join('%g' % (8 + 0.5 * i) for i in range(37)))
    ap.add_argument('--kpho', type=float, default=C.K_PHO,
                    help='phosphatase forward coefficient, input convention (default %(default)s)')
    ap.add_argument('--N', type=int, default=C.NGRID,
                    help='number of modes (resolution only; default %(default)s)')
    ap.add_argument('--L', type=float, default=C.LBOX,
                    help='box size in model units; x L* = %.1f nm gives the physical box '
                         '(default %%(default)s -> %.1f nm)' % (LSTAR, 0.6 * LSTAR))
    ap.add_argument('--T', type=float, default=120.0,
                    help='model time; the transient is ~5 us so this is ample')
    ap.add_argument('--nproc', type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument('--out', default='phase_nk_np.npz')
    ap.add_argument('--shard', type=int, default=0,
                    help='this task index, 0..nshards-1 (SLURM_ARRAY_TASK_ID)')
    ap.add_argument('--nshards', type=int, default=1,
                    help='split the grid across this many array tasks; each writes its own '
                         'npz and merge_phase.py concatenates them')
    ap.add_argument('--save', choices=('none', 'osc', 'all'), default='none',
                    help="write one .npz per cell with the density fields: 'osc' saves only "
                         "oscillating cells, 'all' saves every cell (~2-3 GB for 285)")
    ap.add_argument('--save-dir', default=None,
                    help='directory for the per-cell files.  DEFAULT is derived from the '
                         'parameters, e.g. cells_c0K-2_r30_T150, so runs at different c0K '
                         'or retardation cannot silently overwrite each other -- the cell '
                         'filenames carry only N_k and N_p.')
    ap.add_argument('--retard', type=float, default=None,
                    help='kinase retardation; default is config.KIN_RETARD')
    ap.add_argument('--rate-scale', type=float, default=1.0,
                    help='EXTRA factor on both forward coefficients, on top of the '
                         'config.RATE_SCALE already folded into K_KIN/K_PHO')
    ap.add_argument('--init', choices=('slab', 'uniform', 'cycle'), default='slab',
                    help='initial condition; run the grid from two of these and compare to '
                         'map the bistable region')
    ap.add_argument('--c0K', type=float, default=None,
                    help='kinase-protein coupling; default is config.EXPLICIT["c0K"]')
    ap.add_argument('--nsave', type=int, default=2000,
                    help='frames recorded during the run, before thinning')
    ap.add_argument('--overwrite', action='store_true',
                    help='allow writing into a cell directory that already holds data')
    ap.add_argument('--save-frames', type=int, default=400,
                    help='frames kept per cell file; set equal to --nsave for no thinning')
    ap.add_argument('--pmin', type=float, default=1.0,
                    help='us; an "oscillation" faster than this is not the condensate cycle')
    ap.add_argument('--pmax', type=float, default=20.0)
    a = ap.parse_args()

    NK = [float(x) for x in a.N_k.split(',')]
    NP = [float(x) for x in a.N_p.split(',')]
    if a.save_dir is None:
        c0k = a.c0K if a.c0K is not None else C.EXPLICIT['c0K']
        rt = a.retard if a.retard else C.KIN_RETARD
        a.save_dir = ('cells_c0K%g_r%g_T%g' % (c0k, rt, a.T)).replace('.', 'p')
        if a.init != 'slab':
            a.save_dir += '_' + a.init
        if a.rate_scale != 1.0:
            a.save_dir += ('_s%g' % a.rate_scale).replace('.', 'p')
    # never overwrite saved cells unless --overwrite
    if a.save != 'none' and os.path.isdir(a.save_dir):
        existing = [x for x in os.listdir(a.save_dir) if x.endswith('.npz')]
        if existing and not a.overwrite:
            raise SystemExit(
                "refusing to write into %s: it already holds %d cell files.\n"
                "  pass --save-dir <other> to keep both, or --overwrite to replace them."
                % (a.save_dir, len(existing)))

    cfg = dict(kpho=a.kpho, T=a.T, N=a.N, L=a.L, pmin=a.pmin, pmax=a.pmax,
               save=a.save, save_dir=a.save_dir, save_frames=a.save_frames,
               nsave=a.nsave, c0K=a.c0K, retard=a.retard, rate_scale=a.rate_scale,
               init=a.init)
    jobs = [(k, p, cfg) for k in NK for p in NP]
    ncells_total = len(jobs)
    if a.nshards > 1:
        # strided split, so shards have similar run times
        jobs = jobs[a.shard::a.nshards]

    print("four-field explicit-phosphatase model: %d x %d = %d cells%s, T = %.0f (%.1f us)"
          % (len(NK), len(NP), ncells_total,
             ("; shard %d of %d -> %d cells here" % (a.shard, a.nshards, len(jobs))
              if a.nshards > 1 else ""), a.T, a.T * M.TSTAR_US))
    print("  %d modes, box L = %.4f = %.1f nm, dx = %.3f nm  (simulations: ~100 nm)"
          % (a.N, a.L, a.L * LSTAR, a.L / a.N * LSTAR))
    print("  kinase retarded %.0fx from Rouse (Lam_K = %.5f); phosphatase at Rouse %.5f"
          % (C.KIN_RETARD, C.EXPLICIT['LamK'], C.EXPLICIT['LamP']))
    print("  c0P = %.1f, c1P = %.1f (weakened);  K_pho = %.3g, K_kin = %.3g  (input convention)"
          % (C.EXPLICIT['c0P'], C.EXPLICIT['c1P'], a.kpho, C.K_KIN))
    print("  Delta = %.2f kT = %.0f kJ/mol at T = %.0f K (the applied ATP driving)"
          % (C.DELTA, C.DELTA * C.RT_KJMOL, C.T_MD))
    if a.save != 'none':
        print("  per-cell files -> %s/  (--save %s, %d frames each)"
              % (a.save_dir, a.save, a.save_frames))
    print("  simulated points overlaid: %s\n"
          % ", ".join("(N_k=%d,N_p=%d)" % (k, p) for p, k, _ in C.MD_POINTS), flush=True)

    t0, res = time.time(), []
    with Pool(a.nproc) as pool:
        for r in pool.imap_unordered(_one, jobs):
            res.append(r)
            if r['cls'] == 'oscillating':
                print("   OSC  N_k=%-5.3g N_p=%-5.3g  period %6.2f us  pbar %.4f (%.2f pSer)"
                      % (r['N_k'], r['N_p'], r['period'], r['pbar'],
                         M.NSER * r['pbar']), flush=True)
    osc = [r for r in res if r['cls'] == 'oscillating']
    print("\n  %d cells in %.1f min; %d oscillating"
          % (len(res), (time.time() - t0) / 60, len(osc)))

    # how many oscillating MD set-ups the model reproduces
    if a.nshards > 1:
        print("  (sharded run: coverage is only meaningful after merge_phase.py)")
    else:
        exact = sum(any(abs(r['N_k'] - k) < 1e-9 and abs(r['N_p'] - p) < 1e-9 for r in osc)
                    for p, k, _ in C.MD_POINTS)
        near = sum(any(abs(r['N_k'] - k) < 0.51 and abs(r['N_p'] - p) < 2.01 for r in osc)
                   for p, k, _ in C.MD_POINTS)
        print("  simulated points whose OWN cell oscillates: %d of %d"
              % (exact, len(C.MD_POINTS)))
        print("  ... allowing a neighbouring cell (+-0.5, +-2): %d of %d"
              % (near, len(C.MD_POINTS)))

    keys = ['N_k', 'N_p', 'period', 'pbar', 'amp', 'mass']
    np.savez_compressed(a.out, **{k: np.array([r[k] for r in res]) for k in keys},
                        cls=np.array([r['cls'] for r in res]),
                        NK=np.array(NK), NP=np.array(NP), kpho=a.kpho,
                        Ngrid=a.N, Lbox=a.L, box_nm=a.L * LSTAR,
                        T_MD=C.T_MD, Delta=C.DELTA, dmu=C.DMU_K, dmup=C.DMU_P,
                        kkin=C.EXPLICIT['kkin'], kpho_bare=a.kpho * np.exp(C.DMU_P),
                        md_points=np.array(C.MD_POINTS), Tstar=M.TSTAR_US)
    print("  saved -> %s" % a.out)


if __name__ == '__main__':
    main()
