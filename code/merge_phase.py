#!/usr/bin/env python3
r"""
merge_phase.py -- merge the shards of a phase scan into one file, checking that every cell of
the N_k x N_p grid is present exactly once.

    python3 merge_phase.py phase_shard*.npz --out phase_merged.npz
"""
import argparse
import numpy as np

CELL = ['N_k', 'N_p', 'period', 'pbar', 'amp', 'mass']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shards', nargs='+')
    ap.add_argument('--out', default='phase_merged.npz')
    ap.add_argument('--overwrite', action='store_true',
                    help='replace --out if it exists; refused otherwise')
    a = ap.parse_args()

    zs = [np.load(f, allow_pickle=True) for f in a.shards]
    print("  %d shards" % len(zs))
    out = {k: np.concatenate([np.asarray(z[k]) for z in zs]) for k in CELL}
    out['cls'] = np.concatenate([np.asarray(z['cls']) for z in zs])

    # metadata must agree across shards, or they are not the same scan
    meta = [k for k in zs[0].files if k not in CELL + ['cls']]
    for k in meta:
        vals = [np.asarray(z[k]).tolist() for z in zs if k in z.files]
        if any(v != vals[0] for v in vals):
            raise SystemExit("shards disagree on '%s': %s -- not the same scan" % (k, vals))
        out[k] = zs[0][k]

    # sort rows by (N_k, N_p): shards hold them in completion order
    order = np.lexsort((out['N_p'], out['N_k']))
    for k in CELL + ['cls']:
        out[k] = out[k][order]

    nk, npv = np.unique(out['N_k']), np.unique(out['N_p'])
    want, got = len(nk) * len(npv), len(out['N_k'])
    pairs = set(zip(out['N_k'].tolist(), out['N_p'].tolist()))
    print("  grid %d N_k x %d N_p = %d cells; got %d rows, %d distinct"
          % (len(nk), len(npv), want, got, len(pairs)))
    if len(pairs) != want:
        missing = [(k, p) for k in nk for p in npv if (k, p) not in pairs]
        print("  MISSING %d cells (a shard failed?): %s%s"
              % (len(missing), missing[:10], " ..." if len(missing) > 10 else ""))
    if got != len(pairs):
        print("  WARNING: %d duplicate rows -- shards overlap" % (got - len(pairs)))

    osc = out['cls'] == 'oscillating'
    print("  %d of %d cells oscillating" % (osc.sum(), got))
    import config as C
    exact = sum(any(abs(k2 - k) < 1e-9 and abs(p2 - p) < 1e-9
                    for k2, p2 in zip(out['N_k'][osc], out['N_p'][osc]))
                for p, k, _ in C.MD_POINTS)
    print("  simulated points whose OWN cell oscillates: %d of %d" % (exact, len(C.MD_POINTS)))
    import os
    if os.path.exists(a.out) and not a.overwrite:
        raise SystemExit('refusing to overwrite %s; pass a different --out, or --overwrite' % a.out)
    np.savez_compressed(a.out, **out)
    print("  wrote %s" % a.out)


if __name__ == '__main__':
    main()
