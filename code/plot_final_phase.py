#!/usr/bin/env python3
r"""
plot_final_phase.py -- the manuscript phase diagram.

    python3 plot_final_phase.py
    python3 plot_final_phase.py --out fig_phase.pdf --cmap Blues --width 180
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

MM = 1 / 25.4
FILL = {'steady': '#f6d9b0',        # condensate
        'dissolved': '#cfe3f7'}     # dissolved
HATCH = '////'
HATCH_EC = '#3a3a3a'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'phase_diagram', 'phase_diagram.npz'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'fig_phase_diagram.pdf'))
    ap.add_argument('--cmap', default='Reds',
                    help="sequential ramp for the period; 'Reds' keeps the oscillating "
                         "region in the red of the earlier figures")
    ap.add_argument('--cmap-lo', type=float, default=0.22,
                    help='truncate the pale end of the ramp, so the shortest periods are '
                         'still clearly red rather than washing out against the tan fill')
    ap.add_argument('--cmap-hi', type=float, default=1.0)
    ap.add_argument('--width', type=float, default=120.0, help='figure width in mm')
    ap.add_argument('--fs', type=float, default=7.0)
    ap.add_argument('--pmax', type=float, default=None,
                    help='clip the period scale; outliers otherwise flatten the ramp')
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    NK, NP = np.asarray(z['NK'], float), np.asarray(z['NP'], float)
    state = np.asarray(z['state'])
    per = np.asarray(z['period'], float)
    nk, npv = np.asarray(z['N_k'], float), np.asarray(z['N_p'], float)
    idx = {(round(k, 2), round(p, 0)): i for i, (k, p) in enumerate(zip(nk, npv))}

    osc = np.isin(state, ['oscillating', 'bistable_osc']) & np.isfinite(per)
    vmax = a.pmax if a.pmax else float(np.percentile(per[osc], 95))
    vmin = float(per[osc].min())
    base = plt.get_cmap(a.cmap)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        a.cmap + '_cut', base(np.linspace(a.cmap_lo, a.cmap_hi, 256)))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    plt.rcParams.update({'font.size': a.fs, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
    W = a.width * MM
    fig, ax = plt.subplots(figsize=(W, W * 0.78), constrained_layout=True)

    img = np.ones((len(NP), len(NK), 3))
    hatched = []
    for i, p in enumerate(NP):
        for j, k in enumerate(NK):
            s = state[idx[(round(k, 2), round(p, 0))]]
            v = per[idx[(round(k, 2), round(p, 0))]]
            if s in ('oscillating', 'bistable_osc') and np.isfinite(v):
                img[i, j] = cmap(norm(v))[:3]
            else:
                img[i, j] = matplotlib.colors.to_rgb(
                    FILL.get(s if s in FILL else 'dissolved', '#dddddd'))
            if s.startswith('bistable'):
                hatched.append((i, j))

    ax.imshow(img, origin='lower', aspect='auto', interpolation='nearest')
    for i, j in hatched:                      # texture, not a fourth colour
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch=HATCH,
                               edgecolor=HATCH_EC, linewidth=0.0))

    # MD set-ups: from config.py if available, else from the npz
    md = {k: np.atleast_2d(np.asarray(z[k], float))
          for k in ('md_osc', 'md_steady', 'md_dissolved')}
    try:
        import config as C
        md = {'md_osc': np.array([(k, p) for p, k, _ in C.MD_POINTS], float),
              'md_steady': np.array([(k, p) for p, k in C.MD_POINTS_STEADY], float),
              'md_dissolved': np.array([(k, p) for p, k in C.MD_POINTS_DISSOLVED], float)}
        src = 'config.py'
    except Exception:
        src = 'the npz'

    xi = lambda v: np.interp(v, NK, np.arange(len(NK)))
    yi = lambda v: np.interp(v, NP, np.arange(len(NP)))
    mk = dict(mec='k', mew=0.7, ls='none', ms=4.5)
    for arr, m, lab in ((md['md_osc'], '^', 'oscillating'),
                        (md['md_steady'], 'o', 'condensate'),
                        (md['md_dissolved'], 's', 'dissolved')):
        A = np.atleast_2d(arr)
        ax.plot(xi(A[:, 0]), yi(A[:, 1]), m, mfc='w', label='MD ' + lab, **mk)

    ax.set_xticks(np.arange(len(NK))[::2]); ax.set_xticklabels(['%g' % v for v in NK[::2]])
    ax.set_yticks(np.arange(len(NP))[::2]); ax.set_yticklabels(['%g' % v for v in NP[::2]])
    ax.set_xlabel(r'$N_\mathrm{k}$')
    ax.set_ylabel(r'$N_\mathrm{p}$')

    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      pad=0.02, fraction=0.055)
    cb.set_label(r'oscillation period ($\mu$s)')
    cb.outline.set_linewidth(0.6)

    handles = [Patch(fc=FILL['steady'], ec='none', label='condensate'),
               Patch(fc=FILL['dissolved'], ec='none', label='dissolved'),
               Line2D([], [], marker='^', mfc='w', label='MD oscillating', **mk),
               Line2D([], [], marker='o', mfc='w', label='MD condensate', **mk),
               Line2D([], [], marker='s', mfc='w', label='MD dissolved', **mk)]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 1.22), ncol=3,
              frameon=False, handlelength=1.3, columnspacing=1.0, handletextpad=0.5)

    fig.savefig(a.out)
    print("  wrote %s   (MD points from %s)" % (a.out, src))
    for nm, arr in md.items():
        for k, p in np.atleast_2d(arr):
            i = idx.get((round(float(k), 2), round(float(p), 0)))
            print("     %-13s (%g,%g) -> model: %s"
                  % (nm, k, p, state[i] if i is not None else 'OFF THE PLOTTED GRID'))
    print("  period scale %.2f-%.2f us (95th pct); %d oscillating, %d bistable, "
          "%d condensate, %d dissolved"
          % (vmin, vmax, (state == 'oscillating').sum(),
             np.char.startswith(state.astype(str), 'bistable').sum(),
             (state == 'steady').sum(), (state == 'dissolved').sum()))


if __name__ == '__main__':
    main()
