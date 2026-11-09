#!/usr/bin/env python3
r"""
plot_md_cells.py -- the model's trace at each (N_k, N_p) the simulations oscillate at.
    python3 plot_md_cells.py                          # data/md_points, 4000 frames over 90 us
    python3 plot_md_cells.py --t0 20 --t1 70 --thr 0.5 --out fig_md_cells.pdf
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import resample

MM = 1 / 25.4
NSER, NCHAIN = 24, 200
CCH, CPS = '#1f4e79', '#c0392b'
# the five simulated oscillating points
ROWS = [(2, 10, ''), (2, 12, ''), (3, 12, ''),
        (3, 15, ''), (4, 20, ''), (3, 18, '')]


def load(k, p, thr, up, d=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'md_points')):
    g = lambda v: ('%g' % v).replace('.', 'p')
    f = os.path.join(d, 'cell_nk%s_np%s.npz' % (g(k), g(p)))
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    tot = np.asarray(z['phi0'], float) + np.asarray(z['phi1'], float)
    if thr is None:
        thr = float(z['P_thr'])
    fine = resample(tot, tot.shape[1] * up, axis=1) if up > 1 else tot
    chains = NCHAIN * np.where(fine > thr, fine, 0).sum(1) / fine.sum(1)
    return dict(t=np.asarray(z['t'], float) * float(z['Tstar']), chains=chains,
                pser=NSER * np.asarray(z['pfrac'], float),
                period=float(z['period']) if 'period' in z.files else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'fig_md_cells.pdf'))
    ap.add_argument('--dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'md_points'),
                    help='directory of cell_nk*_np*.npz runs (default data/md_points)')
    ap.add_argument('--t0', type=float, default=20.0)
    ap.add_argument('--t1', type=float, default=70.0)
    ap.add_argument('--thr', type=float, default=None)
    ap.add_argument('--upsample', type=int, default=8)
    ap.add_argument('--width', type=float, default=120.0)
    ap.add_argument('--fs', type=float, default=7.0)
    a = ap.parse_args()

    D = [(k, p, note, load(k, p, a.thr, a.upsample, a.dir)) for k, p, note in ROWS]
    D = [d for d in D if d[3] is not None]
    plt.rcParams.update({'font.size': a.fs, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
    W = a.width * MM
    fig, axes = plt.subplots(len(D), 1, figsize=(W, 0.46 * W * len(D) / 2),
                             sharex=True, constrained_layout=True)
    clim = (0, max(d[3]['chains'].max() for d in D) * 1.06)
    plim = (min(d[3]['pser'].min() for d in D) * 0.96,
            max(d[3]['pser'].max() for d in D) * 1.04)

    twins = []
    for ax, (k, p, note, d) in zip(np.atleast_1d(axes), D):
        m = (d['t'] >= a.t0) & (d['t'] <= a.t1)
        ax.plot(d['t'][m], d['chains'][m], color=CCH, lw=1.0)
        ax.set_ylim(*clim); ax.tick_params(axis='y', colors=CCH)
        tw = ax.twinx()
        twins.append(tw)
        tw.plot(d['t'][m], d['pser'][m], color=CPS, lw=1.0)
        tw.set_ylim(*plim); tw.tick_params(axis='y', colors=CPS, width=0.6, length=2.5)
        for sp in tw.spines.values():
            sp.set_linewidth(0.6)
        lab = r'$N_\mathrm{k}=%d$, $N_\mathrm{p}=%d$' % (k, p)
        if np.isfinite(d['period']):
            lab += r'   $T=%.2f\,\mu$s' % d['period']
        if note:
            lab += '   (%s)' % note
        ax.set_title(lab, loc='left', fontsize=a.fs - 0.5, pad=2.5)
    mid = len(D) // 2
    np.atleast_1d(axes)[mid].set_ylabel('condensed chains', color=CCH)
    np.atleast_1d(axes)[-1].set_xlabel(r'time ($\mu$s)')
    twins[mid].set_ylabel('pSer per chain', color=CPS)      # laid out, so it clears the ticks
    fig.savefig(a.out)
    print("  wrote %s  (%d panels, %.0f-%.0f us, from %s)" % (a.out, len(D), a.t0, a.t1, a.dir))
    for k, p, note, d in D:
        print("     (%d,%2d) chains %5.1f-%5.1f   pSer %5.2f-%5.2f   T %s %s"
              % (k, p, d['chains'].min(), d['chains'].max(), d['pser'].min(),
                 d['pser'].max(),
                 ("%.2f us" % d['period']) if np.isfinite(d['period']) else "-", note))


if __name__ == '__main__':
    main()
