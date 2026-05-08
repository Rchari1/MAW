"""
Unified matplotlib styling for SERPENS Black-Hole paper figures.

Usage
-----
    from src.paperstyle import apply_paper_style, COLUMN_W, FULLPAGE_W
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_W, COLUMN_W * 0.75))

The style targets revtex4 two-column layout: physical column width = 3.42 in,
full-width = 7.08 in. All figures are saved at dpi=300 with figure-size in inches
matching their target on the page, so `\includegraphics{...}` (no scale factor)
fits correctly.

Color scheme is a perceptually uniform, color-blind-safe palette.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


COLUMN_W = 3.42      # inches — single revtex4 column
FULLPAGE_W = 7.08    # inches — revtex4 two-column figure*


# Color-blind-safe palette (Wong 2011 / Bang Wong, "Points of view: Color blindness", Nat. Methods 8, 441)
PALETTE = {
    'black':  '#000000',
    'orange': '#E69F00',
    'sky':    '#56B4E9',
    'green':  '#009E73',
    'yellow': '#F0E442',
    'blue':   '#0072B2',
    'red':    '#D55E00',
    'purple': '#CC79A7',
}


def apply_paper_style(use_tex=True):
    """
    Apply the unified rcParams for paper-quality figures.

    Set use_tex=False on systems without a working LaTeX install; otherwise
    True for proper math rendering.
    """
    rc = {
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.grid': False,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.5,
        'errorbar.capsize': 2.5,
        'image.cmap': 'inferno',
    }
    if use_tex:
        rc['text.usetex'] = True
        rc['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
    else:
        rc['text.usetex'] = False

    mpl.rcParams.update(rc)


def column_figsize(aspect=0.75):
    """Return (width, height) for a single-column figure with given aspect."""
    return (COLUMN_W, COLUMN_W * aspect)


def fullpage_figsize(aspect=0.45):
    """Return (width, height) for a full-page (two-column) figure."""
    return (FULLPAGE_W, FULLPAGE_W * aspect)
