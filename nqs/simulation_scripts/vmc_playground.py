import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nqs.state import nqs
from nqs.state.utils import plot_obd
from nqs.state.utils import plot_tbd  # noqa


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "/Users/haas/Documents/Masters/NQS/data/playground.csv"
nparticles = 2
dim = 2
nsamples = int(2**18)  # 2**18 = 262144
nchains = 1
eta = 0.1  # / np.sqrt(nparticles * dim)  # 0.001  / np.sqrt(nparticles * dim)

training_cycles = 500  # this is cycles for the ansatz
mcmc_alg = "m"
backend = "numpy"
optimizer = "adam"  # reminder: for adam, use bigger learning rate
batch_size = 500
detailed = True
wf_type = "vmc"
seed = 42
save_positions = True
scale = (
    1.0 / np.sqrt(nparticles * dim) if mcmc_alg == "m" else 0.1 / np.sqrt(nparticles)
)
dfs_mean = []
df = []
df_all = []
import time

start = time.time()

system = nqs.NQS(
    nqs_repr="psi",
    backend=backend,
    logger_level="INFO",
    seed=seed,
)

system.set_wf(
    wf_type,
    nparticles,
    dim,
    symmetry="pj",
    correlation="fermion",
)

system.set_sampler(mcmc_alg=mcmc_alg, scale=scale)
system.set_hamiltonian(
    type_="ho", int_type="none", omega=1.0, r0_reg=10, training_cycles=training_cycles
)
system.set_optimizer(
    optimizer=optimizer,
    eta=eta,
    # beta1=0.9,
    # beta2=0.999,
    # epsilon=1e-8,
)

history = system.train(
    max_iter=training_cycles,
    batch_size=batch_size,
    early_stop=False,
    seed=seed,
    history=True,
    tune=False,
)

df = system.sample(
    nsamples, nchains, seed, one_body_density=False, save_positions=save_positions
)
df_all.append(df)

sem_factor = 1 / np.sqrt(len(df))  # sem = standard error of the mean
mean_data = df[["energy", "std_error", "variance", "accept_rate"]].mean().to_dict()
mean_data["sem_energy"] = df["energy"].std() * sem_factor
mean_data["sem_std_error"] = df["std_error"].std() * sem_factor
mean_data["sem_variance"] = df["variance"].std() * sem_factor
mean_data["sem_accept_rate"] = df["accept_rate"].std() * sem_factor
info_data = (
    df[
        [
            "nparticles",
            "dim",
            "eta",
            "scale",
            "mcmc_alg",
            "nqs_type",
            "nsamples",
            "training_cycles",
            "training_batch",
            "Opti",
        ]
    ]
    .iloc[0]
    .to_dict()
)

data = {**mean_data, **info_data}  # ** unpacks the dictionary
df_mean = pd.DataFrame([data])
dfs_mean.append(df_mean)
end = time.time()
print((end - start))
epochs = np.arange(training_cycles)

for key, value in history.items():
    plt.plot(epochs[: len(value)], value, label=key)
    plt.legend()
    plt.show()


df_final = pd.concat(dfs_mean)

# Save results
df_final.to_csv(output_filename, index=False)

# plot energy convergence curve
# energy withour sr
df_all = pd.concat(df_all)
print(df_all)
# energy with sr
if save_positions:
    chain_id = 0  # TODO: make this general to get other chains
    plot_obd(f"energies_and_pos_VMC_ch{chain_id}.h5", nsamples, dim)


if nchains > 1:
    sns.lineplot(data=df_all, x="chain_id", y="energy")
else:
    sns.scatterplot(data=df_all, x="chain_id", y="energy")
# ylim
# plt.ylim(2.9, 3.6)

plt.xlabel("Chain")
plt.ylabel("Energy")
plt.show()
