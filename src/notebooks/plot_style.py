import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
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
