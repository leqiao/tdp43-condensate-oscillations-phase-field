#!/usr/bin/env python3
r"""
plot_init_md.py -- SI figure: the limit cycle does not depend on the initial condensate.

    a  phase portraits at the six set-ups, five different starts converging on the cycle
    b  deviation of period, mean pSer and pSer amplitude from the slab start, all 15 starts

    python3 plot_init_md.py
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MM, FS = 1.0 / 25.4, 7
# Okabe-Ito colours
CVD = dict(blue='#0072B2', vermillion='#D55E00', green='#009E73', purple='#CC79A7',
           orange='#E69F00', black='#000000')
SANS = ['Arial', 'Helvetica', 'Liberation Sans', 'Arimo', 'Nimbus Sans', 'FreeSans',
        'DejaVu Sans']


def set_style(fs=FS):
    have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    fam = next((n for n in SANS if n in have), 'DejaVu Sans')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': SANS, 'mathtext.fontset': 'custom',
        'mathtext.rm': fam, 'mathtext.it': fam + ':italic', 'mathtext.bf': fam + ':bold',
        'font.size': fs, 'axes.labelsize': fs, 'axes.titlesize': fs,
        'xtick.labelsize': fs - 1, 'ytick.labelsize': fs - 1, 'legend.fontsize': fs - 1,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
        'legend.frameon': False, 'legend.handletextpad': 0.4,
        'legend.columnspacing': 0.8, 'legend.handlelength': 1.1, 'lines.linewidth': 0.7,
        'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def panel_label(ax, letter, dx=-0.13, dy=1.03):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=FS + 1, fontweight='bold', va='bottom', ha='left')


PHIBAR, L, NCHAIN = 0.30, 1.16, 200
POINTS = [(2, 10), (2, 12), (3, 12), (3, 15), (3, 18), (4, 20)]
DROPLET = ('C_', 'D_', 'E_', 'F_', 'G_')
# starts shown in panel a
SHOWN = [('C_slab', 'limit cycle', CVD['black']),
         ('C_size0.25', 'small droplet, 0.25$\\times$', CVD['blue']),
         ('D_size1_p0.4', 'pre-phosphorylated, $p=0.4$', CVD['vermillion']),
         ('E_kin_dilute', 'kinase in dilute phase', CVD['green']),
         ('F_drop3_s0', 'three droplets', CVD['purple']),
         ('G_cyc_min', 'restart at cycle minimum', CVD['orange'])]


def name(v):
    return ('%g' % v).replace('.', 'p')


def find(dirs, k, p, label):
    for d in dirs:
        f = os.path.join(d, 'run_nk%s_np%s_%s.npz' % (name(k), name(p), label))
        if os.path.exists(f):
            return f
    return None


def load(f):
    z = np.load(f, allow_pickle=True)
    m = np.asarray(z['mass'], float)
    t = np.asarray(z['t'], float) * float(z['Tstar'])
    pser = 24 * np.asarray(z['pfrac'], float)
    period = float(z['period'])
    # cycle measures from pSer, over a whole number of periods at the end of the run
    if np.isfinite(period) and period > 0:
        n = max(1, int(0.4 * t[-1] // period))
        w = t >= t[-1] - n * period
    else:
        w = t >= 0.6 * t[-1]
    return dict(t=t, ncond=NCHAIN * m / (PHIBAR * L), npser=pser, period=period,
                cls=str(z['cls']), pmean=float(pser[w].mean()), pamp=float(np.ptp(pser[w])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dirs', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'initial_conditions'))
    ap.add_argument('--t1', type=float, default=12.0, help='us of each transient shown in a')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'fig_SI_init.pdf'))
    a = ap.parse_args()
    dirs = a.dirs.split(',')

    set_style()
    fig = plt.figure(figsize=(183 * MM, 150 * MM), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[0.09, 1, 1, 1.05])   # row 0: shared legend
    P = [fig.add_subplot(gs[r, c]) for r in (1, 2) for c in range(3)]
    for ax in P[1:]:
        ax.sharex(P[0]); ax.sharey(P[0])
    B = fig.add_subplot(gs[3, :])
    for ax in P + [B]:
        ax.spines[['top', 'right']].set_visible(True)      # full box frame

    # a -- phase portraits
    for ax, (k, p) in zip(P, POINTS):
        for label, txt, c in SHOWN:
            f = find(dirs, k, p, label)
            if f is None:
                print('missing', (k, p), label)
                continue
            r = load(f)
            if label == 'C_slab':                   # the limit cycle: last two periods
                m = r['t'] >= r['t'][-1] - 2 * r['period']
                ax.plot(r['npser'][m], r['ncond'][m], color='k', lw=1.3, zorder=1,
                        label='limit cycle')
                continue
            m = r['t'] <= a.t1
            ax.plot(r['npser'][m], r['ncond'][m], color=c, lw=0.6, zorder=2, label=txt)
            ax.plot(r['npser'][0], r['ncond'][0], 'o', ms=2.5, color=c, zorder=3)  # start
        ax.set_title(r'$(N_\mathrm{k},N_\mathrm{p})=(%d,%d)$' % (k, p), loc='left')
    P[0].set_ylim(bottom=0)
    for ax in P[3:]:
        ax.set_xlabel('mean pSer per chain')
    for ax in (P[0], P[3]):
        ax.set_ylabel(r'$N_\mathrm{cond}$')
    for ax in P:
        if ax not in (P[0], P[3]):
            ax.tick_params(labelleft=False)
        if ax not in P[3:]:
            ax.tick_params(labelbottom=False)
    h, l = P[0].get_legend_handles_labels()
    lg = fig.add_subplot(gs[0, :])
    lg.axis('off')
    lg.legend(h, l, loc='center', ncol=6, fontsize=FS, handlelength=1.4,
              columnspacing=1.0, borderaxespad=0.0)

    # b -- deviations from the slab start
    rng = np.random.default_rng(0)
    worst_T = worst_M = 0.0
    for i, (k, p) in enumerate(POINTS):
        fref = find(dirs, k, p, 'C_slab')
        if fref is None:
            print('no reference for', (k, p))
            continue
        ref = load(fref)
        dT, dP, dA, nbad = [], [], [], 0
        for d in dirs:
            for f in sorted(glob.glob(os.path.join(d, 'run_nk%s_np%s_*.npz' % (name(k), name(p))))):
                lab = os.path.basename(f).split('_np%s_' % name(p))[1][:-4]
                if not lab.startswith(DROPLET) or lab == 'C_slab':
                    continue
                r = load(f)
                if r['cls'] != 'oscillating':
                    nbad += 1
                    continue
                dT.append(100 * (r['period'] / ref['period'] - 1))
                dP.append(100 * (r['pmean'] / ref['pmean'] - 1))
                dA.append(100 * (r['pamp'] / ref['pamp'] - 1))
        dT, dP, dA = np.array(dT), np.array(dP), np.array(dA)
        jit = rng.uniform(-0.05, 0.05, len(dT))
        for off, y, mk, c, name_ in ((-0.2, dT, 'o', CVD['blue'], 'period'),
                                     (0.0, dP, '^', CVD['vermillion'], 'mean pSer'),
                                     (0.2, dA, 's', CVD['green'], 'pSer amplitude')):
            B.plot(i + off + jit, y, mk, ms=3, mfc='none', mec=c, mew=0.6,
                   label=name_ if i == 0 else None)
        worst_T = max(worst_T, np.abs(dT).max())
        worst_M = max(worst_M, np.abs(dP).max(), np.abs(dA).max())
        print('(%d,%d): %d droplet starts oscillate, %d do not;  |dT| max %.3f%%  '
              '|d mean pSer| max %.3f%%  |d pSer amp| max %.3f%%'
              % (k, p, len(dT), nbad, np.abs(dT).max(), np.abs(dP).max(), np.abs(dA).max()))
    B.axhline(0, color='0.6', lw=0.5, zorder=0)
    B.set_xticks(range(len(POINTS)))
    B.set_xticklabels(['(%d, %d)' % pt for pt in POINTS])
    B.set_xlabel(r'$(N_\mathrm{k},N_\mathrm{p})$')
    B.set_ylabel('deviation from slab start (%)')
    lim = max(1.25 * max(worst_T, worst_M), 1.0)
    B.set_ylim(-lim, lim)
    B.legend(loc='upper right', ncol=3, fontsize=FS)
    print('largest deviation over all points: period %.3f%%, pSer measures %.3f%%' % (worst_T, worst_M))

    panel_label(P[0], 'a', dx=-0.3, dy=1.04)
    panel_label(B, 'b', dx=-0.08, dy=1.02)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fig.savefig(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
