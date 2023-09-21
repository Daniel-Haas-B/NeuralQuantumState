import os

import numpy as np
import pandas as pd

FIGURE_PATH = "./../latex/figures"
TABLE_PATH = "./../latex/tables"


def fig_path(fig_id):
    """
    Input name of figure to load or save with extension as dtype str
    """
    return os.path.join(FIGURE_PATH + "/", fig_id)


def tab_path(tab_id):
    """
    Input name of table to load or save with extension as dtype str
    """
    return os.path.join(TABLE_PATH + "/", tab_id)


def df_to_latex(df, outfile):
    """
    Render dataframe to a LaTeX tabular environment table and write to file
    """
    n = len(df.columns)
    col_format = n * "c"
    # Render dataframe to a LaTeX tabular environment table and write to file
    pd.options.display.float_format = "{:,.3f}".format
    df = df.apply(lambda x: x.astype(int) if np.allclose(x, x.astype(int)) else x)
    pd.options.display.latex.escape = False
    latex = df.to_latex(index=False, column_format=col_format)
    latex = latex.replace("\\toprule", "\\hline \\hline")
    latex = latex.replace("\\midrule", "\\hline \\hline")
    latex = latex.replace("\\bottomrule", "\\hline \\hline")

    with open(tab_path(outfile), "w") as f:
        f.write(latex)


def set_figsize(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
        Document textwidth or columnwidth in pts
    fraction: float, optional
        Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
        Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
