import sys

sys.path.append("/Users/haas/Documents/Masters/GANQS/src/")
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nqs import nqs
from nqs.utils import plot_psi2


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "../data/vmc_playground.csv"
nparticles = 2
dim = 1
nsamples = int(2**16)  # 2**18 = 262144
nchains = 4
eta = 0.1

training_cycles = [20_000]  # this is cycles for the ansatz
mcmc_alg = "m"
backend = "jax"
optimizer = "gd"
batch_size = 500
detailed = True
wf_type = "vmc"
seed = 142

dfs_mean = []
df = []
df_all = []
import time

# for max_iter in training_cycles:
start = time.time()
# for i in range(5):


for sr in [False]:
    system = nqs.NQS(
        nqs_repr="psi",
        backend=backend,
        log=True,
        logger_level="INFO",
        use_sr=sr,
        seed=seed,
    )

    system.set_wf(
        wf_type,
        nparticles,
        dim,
        sigma2=1.0,
    )

    system.set_sampler(mcmc_alg=mcmc_alg, scale=1.0)
    system.set_hamiltonian(type_="ho", int_type=None, omega=1.0)
    system.set_optimizer(
        optimizer=optimizer,
        eta=eta,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )

    system.train(
        max_iter=training_cycles[0],
        batch_size=batch_size,
        early_stop=False,
        seed=seed,
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
                # "nvisible",
                # "nhidden",
                "mcmc_alg",
                "nqs_type",
                "nsamples",
                "training_cycles",
                "training_batch",
                "sr",
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


df_final = pd.concat(dfs_mean)

# Save results
df_final.to_csv(output_filename, index=False)

# plot energy convergence curve
# energy withour sr
df_all = pd.concat(df_all)
print(df_all)
# energy with sr
if nchains > 1:
    sns.lineplot(data=df_all, x="chain_id", y="energy", hue="sr")
else:
    sns.scatterplot(data=df_all, x="chain_id", y="energy", hue="sr")
# ylim
# plt.ylim(2.9, 3.6)

plt.xlabel("Chain")
plt.ylabel("Energy")
plt.show()

# system_omega_2 = nqs.NQS(
#     nqs_repr="psi",
#     backend=backend,
#     log=True,
#     logger_level="INFO",
#     use_sr=False,  # Assuming you want to keep Stochastic Reconfiguration the same
#     seed=seed,
# )

# system_omega_2.set_wf(
#     wf_type,
#     nparticles,
#     dim,
#     nhidden=nhidden,
#     sigma2=1.0,
# )

# system_omega_2.set_sampler(mcmc_alg=mcmc_alg, scale=1.0)
# system_omega_2.set_hamiltonian(type_="ho", int_type="Coulomb", omega=2.0)  # Changed omega to 2
# system_omega_2.set_optimizer(
#     optimizer=optimizer,
#     eta=eta,
#     beta1=0.9,
#     beta2=0.999,
#     epsilon=1e-8,
# )

# system_omega_2.train(
#     max_iter=training_cycles[0],
#     batch_size=batch_size,
#     early_stop=False,
#     seed=seed,
# )

# system_omega_2.sample(nsamples, nchains=nchains, seed=seed)

# # Plotting psi2 for both wave functions
# plt.figure(figsize=(10, 6))
plot_psi2(system.wf, r_min=-4, r_max=4, num_points=300)
# plot_psi2(system_omega_2.wf, r_min=-4, r_max=4, num_points=300)
plt.legend()
plt.xlabel("Position")
plt.ylabel("Psi^2")
# plt.title("Comparison of Psi^2 for Different Omega Values")
# plt.show()
