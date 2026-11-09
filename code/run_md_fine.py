#!/usr/bin/env python3
r"""
run_md_fine.py -- the six MD set-ups where the model oscillates, finely sampled
(4000 frames over 90 us), for the model traces and phase portraits (data/md_points).

    python3 run_md_fine.py --check          # parameters only, no run
    python3 run_md_fine.py --nsave 4000 --nproc 6 --save-dir ../data/md_points
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

import config as C
import model as M
import scan_phase as S

PAIRS = [(2, 10), (2, 12), (3, 12), (3, 15), (4, 20), (3, 18)]


def args_dict(a):
    """The same `a` dict scan_phase.main() builds, for the published settings."""
    return dict(kpho=C.K_PHO, rate_scale=1.0, nsave=a.nsave, save_frames=a.nsave,
                N=C.NGRID, L=C.LBOX, T=a.T, init=a.init, c0K=a.c0K, retard=a.retard,
                pmin=1.0, pmax=a.pmax, save='all', save_dir=a.save_dir)


def check(a, ref):
    """Assert the reconstructed P equals the one stored in a published cell file."""
    d = args_dict(a)
    over = dict(c0K=d['c0K'], LamK=C.LAMK_ROUSE / d['retard'])
    P = C.params(M, N_k=3, N_p=15, K_pho=d['kpho'] * d['rate_scale'], nsave=d['nsave'],
                 N=d['N'], L=d['L'], **over)
    P['kkin'] *= d['rate_scale']
    z = np.load(ref, allow_pickle=True)
    bad = []
    for k, v in sorted(P.items()):
        if not isinstance(v, (int, float)) or 'P_' + k not in z.files:
            continue
        if k == 'nsave':
            continue
        if not np.isclose(float(z['P_' + k]), float(v), rtol=1e-12, atol=0.0):
            bad.append('%-10s published %.12g  here %.12g' % (k, float(z['P_' + k]), v))
    print('reference: %s' % ref)
    print('nsave      published %d  here %d' % (int(z['P_nsave']), P['nsave']))
    print('\n'.join(bad) if bad else 'every other parameter identical to the published run')
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nsave', type=int, default=4000, help='frames kept, no thinning')
    ap.add_argument('--T', type=float, default=150.0, help='model time units (90 us)')
    ap.add_argument('--c0K', type=float, default=-1.6)
    ap.add_argument('--retard', type=float, default=50.0)
    ap.add_argument('--pmax', type=float, default=20.0)
    ap.add_argument('--init', choices=('slab', 'uniform', 'cycle'), default='slab')
    ap.add_argument('--nproc', type=int, default=len(PAIRS))
    ap.add_argument('--save-dir', default=None)
    ap.add_argument('--check', action='store_true', help='print parameters and exit')
    ap.add_argument('--ref', default='', help='optional cell of the phase scan to check the parameters against')
    a = ap.parse_args()
    if a.save_dir is None:
        a.save_dir = ('cells_mdfine_c0K%g_r%g_T%g_%s'
                      % (a.c0K, a.retard, a.T, a.init)).replace('.', 'p')

    ok = check(a, a.ref) if a.ref and os.path.exists(a.ref) else None
    if a.check:
        return
    if ok is False:
        raise SystemExit('parameters differ from the published run -- not launching')

    d = args_dict(a)
    print('\n%d cells on %d procs, T = %g (%.1f us), %d frames each (%.2f us/frame)'
          % (len(PAIRS), a.nproc, a.T, a.T * M.TSTAR_US, a.nsave,
             a.T * M.TSTAR_US / a.nsave))
    print('c0K = %g, retard = %gx (Lam_K = %.6g), -> %s/\n'
          % (a.c0K, a.retard, C.LAMK_ROUSE / a.retard, a.save_dir))
    t0 = time.time()
    with mp.Pool(a.nproc) as pool:
        for r in pool.imap_unordered(S._one, [(k, p, d) for k, p in PAIRS]):
            print('  N_k=%-4g N_p=%-4g %-12s period %6.2f us  pbar %.4f   [%.0f min]'
                  % (r['N_k'], r['N_p'], r['cls'], r['period'], r['pbar'],
                     (time.time() - t0) / 60), flush=True)
    print('\n%d cells in %.0f min -> %s/' % (len(PAIRS), (time.time() - t0) / 60, a.save_dir))


if __name__ == '__main__':
    main()
