import sys

sys.path.append("/Users/haas/Documents/Masters/GANQS/src/")
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nqs import nqs

# from nqs.utils import plot_psi2


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "../data/vmc_playground.csv"
nparticles = 2
dim = 2
nsamples = int(2**10)  # 2**18 = 262144
nchains = 1
eta = 0.1

training_cycles = 5  # this is cycles for the ansatz
mcmc_alg = "m"
backend = "jax"
optimizer = "adam"
batch_size = 1
detailed = True
wf_type = "dummy"
seed = 142

dfs_mean = []
df = []
df_all = []
import time

# for max_iter in training_cycles:
start = time.time()
# for i in range(5):

system = nqs.NQS(
    nqs_repr="psi",
    backend=backend,
    log=True,
    logger_level="INFO",
    seed=seed,
)

system.set_wf(
    wf_type,
    nparticles,
    dim,
    sigma2=1.0,
)

system.set_sampler(mcmc_alg=mcmc_alg, scale=1.0)
system.set_hamiltonian(
    type_="ho", int_type="Coulomb", omega=1.0, r0_reg=3, training_cycles=training_cycles
)
system.set_optimizer(
    optimizer=optimizer,
    eta=eta,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
)

history = system.train(
    max_iter=training_cycles,
    batch_size=batch_size,
    early_stop=False,
    seed=seed,
    history=True,
)

df = system.sample(nsamples, nchains=nchains, seed=seed)
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

epochs = np.arange(training_cycles)
plt.plot(epochs, history["energy"], label="energy")
plt.legend()
plt.show()
plt.plot(epochs, history["grads"], label="gradient_norm")
plt.legend()
plt.show()
end = time.time()
print((end - start))


df_final = pd.concat(dfs_mean)

# Save results
df_final.to_csv(output_filename, index=False)

# plot energy convergence curve
# energy withour sr
df_all = pd.concat(df_all)
print(df_all)
# energy with sr
if nchains > 1:
    sns.lineplot(data=df_all, x="chain_id", y="energy")
else:
    sns.scatterplot(data=df_all, x="chain_id", y="energy")
# ylim
# plt.ylim(2.9, 3.6)

plt.xlabel("Chain")
plt.ylabel("Energy")
plt.show()
exit()
positions, one_body_density = system.sample(
    nsamples, nchains=1, seed=seed, one_body_density=True
)

plt.plot(positions, one_body_density)
plt.show()
