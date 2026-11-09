#!/usr/bin/env python3
r"""
passive_1d.py -- passive controls, started from the condensate slab.

  noreact    reactions off and no enzymes: pSer stays 0, the condensate is stable
  undriven   enzymes and reactions on, zero driving (dmu_k = dmu_p = 0): pSer relaxes to
             equilibrium (N_k = 3, N_p = 15 by default)

    python3 passive_1d.py --mode noreact
    python3 passive_1d.py --mode undriven --Nk 3 --Np 15    # T = 50 model units = 30 us
"""
import argparse
import os
import time

import numpy as np

import config as C
import model as M

NCHAIN, NSER = 200, 24


def params(mode, N_k, N_p):
    P = C.params(M, N_k=N_k, N_p=N_p)
    if mode == 'noreact':
        P.update(kkin=0.0, kpho=0.0)
    elif mode == 'undriven':
        P.update(dmu=0.0, dmup=0.0)
    else:
        raise ValueError(mode)
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=('noreact', 'undriven'), default='noreact')
    ap.add_argument('--Nk', type=float, default=None, help='default 0 (noreact), 3 (undriven)')
    ap.add_argument('--Np', type=float, default=None, help='default 0 (noreact), 15 (undriven)')
    ap.add_argument('--T', type=float, default=50.0, help='model units; 50 = 30 us')
    ap.add_argument('--nsave', type=int, default=2000, help='points in the time series')
    ap.add_argument('--nframes', type=int, default=200, help='field snapshots kept')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.Nk is None:
        a.Nk = 0.0 if a.mode == 'noreact' else 3.0
    if a.Np is None:
        a.Np = 0.0 if a.mode == 'noreact' else 15.0
    if a.out is None:
        g = lambda v: ('%g' % v).replace('.', 'p')
        a.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'passive',
                             'passive_1d_noreact.npz' if a.mode == 'noreact' and a.Nk == a.Np == 0 else
                             'passive_1d_%s_nk%s_np%s.npz' % (a.mode, g(a.Nk), g(a.Np)))

    P = params(a.mode, a.Nk, a.Np)
    g = M.grid(P['N'], P['L'])
    N, thr = P['N'], P['thr']
    x = M.initial(P)
    nsteps = int(round(a.T / P['dt']))
    stride = max(1, nsteps // a.nsave)
    fstride = max(1, nsteps // a.nframes)
    print('%s: N_k = %g, N_p = %g, k_kin = %.3g, k_pho = %.3g, dmu_k = %g, dmu_p = %g;  '
          'T = %g (%.1f us), %d steps' % (a.mode, a.Nk, a.Np, P['kkin'], P['kpho'], P['dmu'],
                                          P['dmup'], a.T, a.T * M.TSTAR_US, nsteps), flush=True)
    t, chains, pser, F, tf, frames = [], [], [], [], [], []
    t0 = time.time()
    for n in range(nsteps + 1):
        if n % stride == 0 or n % fstride == 0:
            p0, p1, pK, pP = M.unpack(x, N)
            tot = p0 + p1
            if n % stride == 0:
                t.append(n * P['dt'] * M.TSTAR_US)
                chains.append(NCHAIN * float(tot[tot > thr].sum() / tot.sum()))
                pser.append(NSER * float(p1.sum() / tot.sum()))
                F.append(M.free_energy(x, P, g))
            if n % fstride == 0:
                tf.append(n * P['dt'] * M.TSTAR_US)
                frames.append(np.stack([p0, p1, pK, pP]).astype(np.float32))
        if n < nsteps:
            x = M.step(x, P, g)
            if not np.isfinite(x).all():
                raise SystemExit('diverged at step %d' % n)

    t, chains, pser, F = map(np.array, (t, chains, pser, F))
    xnm = ((np.arange(N) + 0.5) / N - 0.5) * P['L'] * 85.93
    np.savez_compressed(a.out, t_us=t, chains=chains, pser=pser, F=F, t_frames_us=np.array(tf),
                        frames=np.array(frames), x_nm=xnm, fields=np.array(M.FIELDS),
                        mode=a.mode, N_k=a.Nk, N_p=a.Np, T=a.T, Tstar_us=M.TSTAR_US,
                        **{('P_' + k): v for k, v in P.items() if isinstance(v, (int, float))})
    late = t > 0.8 * t[-1]
    rises = int((np.diff(F) > 1e-12 * np.abs(F).max()).sum())
    print('done in %.1f min -> %s' % ((time.time() - t0) / 60, a.out))
    print('  chains %.1f -> %.1f (last 20%%: %.1f +- %.1f);  pSer/chain %.2f -> %.2f '
          '(last 20%%: %.2f +- %.2f);  F rises %d'
          % (chains[0], chains[-1], chains[late].mean(), chains[late].std(), pser[0], pser[-1],
             pser[late].mean(), pser[late].std(), rises))


if __name__ == '__main__':
    main()
