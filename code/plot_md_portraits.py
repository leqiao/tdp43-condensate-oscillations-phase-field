#!/usr/bin/env python3
r"""
plot_md_portraits.py -- phase portraits (condensed chains vs pSer per chain) at the MD set-ups.

    python3 plot_md_portraits.py
    python3 plot_md_portraits.py --panels          # small multiples instead of overlay
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
OKABE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
ROWS = [(2, 10, ''), (2, 12, ''), (3, 12, ''),
        (3, 15, ''), (4, 20, ''), (3, 18, '')]


def refine_period(d, T0, span=0.06, n=801):
    """Sharpen the folding period by phase-dispersion minimisation.

    classify() reports the period to about the frame spacing, and folding ten cycles with a
    1% error walks the phase by a tenth of a cycle, which reopens the loop.  Scanning T
    around T0 and keeping the value that makes the folded curve shortest -- the string
    length through the phase-sorted points -- pins it down far more tightly, because the
    true period is the one at which every cycle superimposes.
    """
    best, bestL = T0, np.inf
    for T in np.linspace(T0 * (1 - span), T0 * (1 + span), n):
        o = np.argsort(np.mod(d['t'], T))
        x, y = d['chains'][o], d['pser'][o]
        xs = (x - x.min()) / max(np.ptp(x), 1e-12)
        ys = (y - y.min()) / max(np.ptp(y), 1e-12)
        L = np.hypot(np.diff(xs), np.diff(ys)).sum()
        if L < bestL:
            best, bestL = T, L
    return best


def fold(d):
    """Pool every cycle onto one orbit.
    """
    T = d['period']
    if not np.isfinite(T) or T <= 0:
        return d['chains'], d['pser']
    T = refine_period(d, T)
    d['period_refined'] = T
    ph = np.mod(d['t'], T) / T
    o = np.argsort(ph)
    x, y = d['chains'][o], d['pser'][o]
    return np.append(x, x[0]), np.append(y, y[0])          # close the loop


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
    return dict(t=np.asarray(z['t'], float) * float(z['Tstar']),
                chains=NCHAIN * np.where(fine > thr, fine, 0).sum(1) / fine.sum(1),
                pser=NSER * np.asarray(z['pfrac'], float),
                period=float(z['period']) if 'period' in z.files else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=None)
    ap.add_argument('--t0', type=float, default=30.0, help='us of transient to drop')
    ap.add_argument('--thr', type=float, default=None)
    ap.add_argument('--upsample', type=int, default=8)
    ap.add_argument('--panels', action='store_true', help='small multiples')
    ap.add_argument('--dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'md_points'),
                    help='directory of cell_nk*_np*.npz runs (default data/md_points)')
    ap.add_argument('--fold', action='store_true',
                    help='pool all cycles onto one orbit; use with the 400-frame cells')
    ap.add_argument('--width', type=float, default=90.0)
    ap.add_argument('--fs', type=float, default=7.0)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', ('fig_md_portraits_panels.pdf' if a.panels else 'fig_md_portraits.pdf'))

    D = [(k, p, n, load(k, p, a.thr, a.upsample, a.dir)) for k, p, n in ROWS]
    for k, p, n, d in D:
        if d is not None and np.isfinite(d['period']) and not a.fold:
            fpc = d['period'] / np.median(np.diff(d['t']))
            if fpc < 50:
                print('  WARNING (%g,%g): only %.0f frames per cycle in %s -- the raw loop will '
                      'look jagged and shifted; add --fold' % (k, p, fpc, a.dir))
    D = [d for d in D if d[3] is not None]
    plt.rcParams.update({'font.size': a.fs, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
    W = a.width * MM

    if a.panels:
        nc = 3
        nr = int(np.ceil(len(D) / nc))
        fig, ax = plt.subplots(nr, nc, figsize=(W * 1.9, W * 0.72 * nr),
                               sharex=True, sharey=True, constrained_layout=True)
        ax = np.atleast_1d(ax).ravel()
        for i, (k, p, note, d) in enumerate(D):
            m = d['t'] >= a.t0
            dd = {kk: (vv[m] if getattr(vv, 'ndim', 0) else vv) for kk, vv in d.items()}
            X, Y = (fold(dd) if a.fold else (dd['chains'], dd['pser']))
            ax[i].plot(X, Y, color=OKABE[i % len(OKABE)], lw=0.8)
            t = r'$N_\mathrm{k}=%d$, $N_\mathrm{p}=%d$' % (k, p)
            if note:
                t += '  (%s)' % note
            ax[i].set_title(t, loc='left', fontsize=a.fs - 0.5, pad=2.5)
        for j in range(len(D), len(ax)):
            ax[j].axis('off')
        for j in range(len(D)):
            if j // nc == nr - 1 or j + nc >= len(D):
                ax[j].set_xlabel('condensed chains')
            if j % nc == 0:
                ax[j].set_ylabel('pSer per chain')
    else:
        fig, axs = plt.subplots(figsize=(W, W * 0.82), constrained_layout=True)
        for i, (k, p, note, d) in enumerate(D):
            m = d['t'] >= a.t0
            lab = r'$N_\mathrm{k}=%d$, $N_\mathrm{p}=%d$' % (k, p)
            if note:
                lab += ' (%s)' % note.split(':')[0]
            dd = {kk: (vv[m] if getattr(vv, 'ndim', 0) else vv) for kk, vv in d.items()}
            X, Y = (fold(dd) if a.fold else (dd['chains'], dd['pser']))
            axs.plot(X, Y, color=OKABE[i % len(OKABE)], lw=0.8, label=lab)
        axs.set_xlabel('condensed chains')
        axs.set_ylabel('pSer per chain')
        axs.legend(frameon=False, fontsize=a.fs - 1, handlelength=1.4, labelspacing=0.3)

    fig.savefig(out)
    print("  wrote %s  (transient before %.0f us dropped; %s, %s)"
          % (out, a.t0, a.dir, 'folded' if a.fold else 'raw trajectory'))
    for k, p, note, d in D:
        m = d['t'] >= a.t0
        print("     (%d,%2d) chains %5.1f-%5.1f  pSer %5.2f-%5.2f  T %s"
              % (k, p, d['chains'][m].min(), d['chains'][m].max(),
                 d['pser'][m].min(), d['pser'][m].max(),
                 ("%.2f us" % d['period']) if np.isfinite(d['period']) else "-"))


if __name__ == '__main__':
    main()
