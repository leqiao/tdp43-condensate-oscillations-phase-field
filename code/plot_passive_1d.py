#!/usr/bin/env python3
r"""
plot_passive_1d.py -- passive controls: condensed chains and pSer per chain against time.

    python3 plot_passive_1d.py
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MM = 1 / 25.4
CCH, CPS = '#1f4e79', '#c0392b'
plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
                     'savefig.bbox': 'tight'})
LABEL = {'noreact': 'reactions off, no enzymes',
         'undriven': 'zero driving ($\\Delta=0$)'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*',
                    default=[os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'passive', 'passive_1d_noreact.npz'),
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'passive', 'passive_1d_undriven_nk3_np15.npz')])
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'fig_passive_1d.pdf'))
    ap.add_argument('--t0', type=float, default=0.0)
    ap.add_argument('--t1', type=float, default=None, help='us; default the whole run')
    ap.add_argument('--pmax', type=float, default=None, help='right-axis max (pSer per chain)')
    a = ap.parse_args()

    fig, axs = plt.subplots(len(a.files), 1, figsize=(89 * MM, 36 * len(a.files) * MM),
                            sharex=True, squeeze=False)
    fig.subplots_adjust(hspace=0.5)
    for ax, fn in zip(axs[:, 0], a.files):
        z = np.load(fn, allow_pickle=True)
        t = z['t_us']
        t1 = a.t1 if a.t1 is not None else t[-1]
        w = (t >= a.t0) & (t <= t1)
        ax.plot(t[w], z['chains'][w], color=CCH)
        ax.set_ylim(0, 200)
        ax.set_ylabel('chains in\ncondensate', color=CCH)
        ax.tick_params(axis='y', colors=CCH)
        bx = ax.twinx()
        bx.plot(t[w], z['pser'][w], color=CPS)
        top = a.pmax if a.pmax else max(1.0, 1.25 * float(z['pser'][w].max()))
        bx.set_ylim(0, top)
        bx.set_ylabel('pSer per chain', color=CPS)
        bx.tick_params(axis='y', colors=CPS)
        mode = str(z['mode'])
        ax.set_title('%s, $N_\\mathrm{k}=%g$, $N_\\mathrm{p}=%g$'
                     % (LABEL.get(mode, mode), float(z['N_k']), float(z['N_p'])),
                     fontsize=7, loc='left')
        ax.set_xlim(a.t0, t1)
        late = t > 0.8 * t[-1]
        print('%s: %s, N_k=%g N_p=%g -- last 20%%: %.1f chains, %.2f pSer/chain'
              % (fn, mode, float(z['N_k']), float(z['N_p']), z['chains'][late].mean(),
                 z['pser'][late].mean()))
    axs[-1, 0].set_xlabel('time ($\\mu$s)')
    fig.savefig(a.out)
    print('-> %s' % a.out)


if __name__ == '__main__':
    main()
