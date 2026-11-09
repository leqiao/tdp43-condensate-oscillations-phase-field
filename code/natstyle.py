"""natstyle.py -- figure style: Nature sizes (89 / 183 mm), 7 pt Arial-compatible font,
live text in the PDF, Okabe-Ito colours.
"""
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

MM = 1.0 / 25.4
FS = 7
WIDTH_1COL = 89.0
WIDTH_2COL = 183.0
MAX_HEIGHT = 247.0

# Arial, or metric-compatible TrueType clones
SANS = ['Arial', 'Helvetica', 'Liberation Sans', 'Arimo', 'Nimbus Sans',
        'FreeSans', 'DejaVu Sans']

# Okabe-Ito colours
CVD = dict(blue='#0072B2', vermillion='#D55E00', green='#009E73', purple='#CC79A7',
           orange='#E69F00', skyblue='#56B4E9', yellow='#F0E442', black='#000000')


def resolved_family():
    """First font in SANS that is actually installed."""
    have = {f.name for f in fm.fontManager.ttflist}
    for n in SANS:
        if n in have:
            return n
    return 'DejaVu Sans'


def nature_style(fs=FS):
    fam = resolved_family()
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': SANS,
        'mathtext.fontset': 'custom',
        'mathtext.rm': fam, 'mathtext.it': fam + ':italic', 'mathtext.bf': fam + ':bold',
        'font.size': fs, 'axes.labelsize': fs, 'axes.titlesize': fs,
        'xtick.labelsize': fs - 1, 'ytick.labelsize': fs - 1, 'legend.fontsize': fs - 1,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'legend.frameon': False, 'legend.handletextpad': 0.4,
        'legend.columnspacing': 0.8, 'legend.handlelength': 1.1,
        'lines.linewidth': 0.7,
        'image.interpolation': 'nearest',   # never blur a categorical map
        'savefig.dpi': 600,
        'savefig.bbox': None,               # keep the page exactly the size we asked for
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    return fam


def panel_label(ax, letter, dx=-0.13, dy=1.03, fs=FS + 1):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=fs, fontweight='bold', va='bottom', ha='left')
