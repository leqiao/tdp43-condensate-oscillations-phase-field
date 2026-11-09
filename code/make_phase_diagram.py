#!/usr/bin/env python3
r"""
make_phase_diagram.py -- the state diagram of Fig. 4e from the merged phase scan.

    python3 scan_phase.py --nproc 128 --T 150 --c0K -1.6 --retard 50 --pmax 20 \
            --out phase_slab_shard0.npz ...                 # (array job, one shard per task)
    python3 merge_phase.py phase_slab_shard*.npz --out phase_slab_merged.npz
    python3 make_phase_diagram.py phase_slab_merged.npz      # -> data/phase_diagram/

Keeps the plotted grid (N_k 1.75-5 in steps of 0.25, N_p 8-26), one state per cell from the
condensate-slab start (oscillating / steady / dissolved), its period in us and mean pSer
fraction, and the MD set-ups drawn on top of the diagram.
"""
import argparse
import os

import numpy as np

import config as C
import model as M

NK = np.round(np.arange(1.75, 5.0 + 1e-9, 0.25), 2)
NP = np.arange(8.0, 26.0 + 1e-9, 1.0)
# MD set-ups drawn on the diagram, (N_k, N_p); config.py stores them as (N_p, N_k[, pbar])
MD_OSC = [(k, p) for p, k, _ in C.MD_POINTS]
MD_STEADY = [(k, p) for p, k in C.MD_POINTS_STEADY]
MD_DISSOLVED = [(k, p) for p, k in C.MD_POINTS_DISSOLVED]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('merged', help='output of merge_phase.py for the slab start')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  '..', 'data', 'phase_diagram',
                                                  'phase_diagram.npz'))
    a = ap.parse_args()
    z = np.load(a.merged, allow_pickle=True)
    nk = np.round(np.asarray(z['N_k'], float), 2)
    npv = np.asarray(z['N_p'], float)
    keep = np.isin(nk, NK) & np.isin(npv, NP)
    if keep.sum() != NK.size * NP.size:
        raise SystemExit('merged scan covers %d of the %d plotted cells'
                         % (keep.sum(), NK.size * NP.size))
    out = dict(N_k=nk[keep], N_p=npv[keep], state=np.asarray(z['cls'])[keep],
               period=np.asarray(z['period'], float)[keep],
               pbar=np.asarray(z['pbar'], float)[keep],
               NK=NK, NP=NP, md_osc=np.array(MD_OSC, float),
               md_steady=np.array(MD_STEADY, float), md_dissolved=np.array(MD_DISSOLVED, float),
               c0K=C.EXPLICIT['c0K'], retard=C.KIN_RETARD, T=150.0,
               Lstar=85.93, Tstar=round(M.TSTAR_US, 4))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **out)
    s = out['state']
    print('wrote %s: %d cells, %d oscillating, %d steady, %d dissolved'
          % (a.out, s.size, (s == 'oscillating').sum(), (s == 'steady').sum(),
             (s == 'dissolved').sum()))


if __name__ == '__main__':
    main()
