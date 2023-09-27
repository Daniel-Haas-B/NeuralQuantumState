import sys

import jax
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# from helper import df_to_latex, fig_path, set_figsize, tab_path


# Import code from src
sys.path.insert(0, "../nqs/")
import nqs  # noqa

# Set plot and dataframe style
sns.set(context="paper", style="darkgrid", rc={"axes.facecolor": "0.96"})

fontsize = "large"
params = {
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern"],
    "axes.labelsize": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.handlelength": 2,
}

plt.rcParams.update(params)
plt.rc("text", usetex=True)

pd.set_option("display.max_columns", 50)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
