import pathlib as pl

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler


def make_figs_path(filename):
    cur_path = pl.Path(__file__)
    root_path = cur_path

    while root_path.name != "GANQS":
        root_path = root_path.parent

    figs_path = root_path / pl.Path("analysis/figs")

    if not figs_path.exists():
        return None
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    figs_path /= filename

    return str(figs_path)


def save(filename):
    if filename:
        filename = make_figs_path(filename)
        plt.savefig(filename)


colors = [
    sns.color_palette("pastel")[0],
    sns.color_palette("pastel")[1],
    sns.color_palette("pastel")[2],
    sns.color_palette("pastel")[3],
    sns.color_palette("pastel")[4],
    sns.color_palette("pastel")[5],
]

markers = ["s", "o", "X", "*", "^"]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors[:2] + [colors[3]])

cmap_terrain = matplotlib.colors.LinearSegmentedColormap.from_list(
    "",
    [
        "darkslategrey",
        sns.color_palette("dark")[-1],
        sns.color_palette("husl")[-3],
        # sns.color_palette('deep')[-1],
        "darkturquoise",
        sns.color_palette("husl")[-2],
        # 'mediumorchid',
        sns.color_palette("husl")[-1],
    ],
)

sns.set_style("whitegrid")
# Set all fonts to be equal to tex
# https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "monospace"
plt.rcParams["text.usetex"] = True

# Saving parameters
plt.rcParams["savefig.dpi"] = 300

# Figure options, set tight layout
plt.rc("figure", autolayout=True)

# Font sizes
plt.rc("axes", titlesize=18, labelsize=16, prop_cycle=cycler("color", colors))
plt.rc("legend", fontsize=14, shadow=False)

# Tick parameters
_ticks_default_parameters = {"labelsize": 12}
plt.rc("xtick", **_ticks_default_parameters)
plt.rc("ytick", **_ticks_default_parameters)

# Line options
plt.rc("lines", linewidth=2)


# To see more paramteres, print the possible options:
# print(plt.rcParams)
