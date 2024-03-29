import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.state import nqs
from src.state.utils import plot_psi2  # noqa

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "/Users/haas/Documents/Masters/NQS/data/playground.csv"
nparticles = 20
dim = 2
nsamples = int(2**18)  # 2**18 = 262144
nchains = 1
eta = 0.01 / np.sqrt(nparticles * dim)  # 0.001  / np.sqrt(nparticles * dim)

training_cycles = 500  # this is cycles for the ansatz
mcmc_alg = "m"
backend = "jax"
optimizer = "adam"
batch_size = 1000
detailed = True
wf_type = "vmc"
seed = 142

dfs_mean = []
df = []
df_all = []
import time

start = time.time()

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
    symmetry=None,
    correlation="j",
)

system.set_sampler(mcmc_alg=mcmc_alg, scale=1.0 / np.sqrt(nparticles * dim))
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
    tune=True,
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
end = time.time()
print((end - start))
epochs = np.arange(training_cycles)

for key, value in history.items():
    plt.plot(epochs, value, label=key)
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
if nchains > 1:
    sns.lineplot(data=df_all, x="chain_id", y="energy")
else:
    sns.scatterplot(data=df_all, x="chain_id", y="energy")
# ylim
# plt.ylim(2.9, 3.6)

plt.xlabel("Chain")
plt.ylabel("Energy")
plt.show()

# positions, one_body_density = system.sample(
#     nsamples, nchains=1, seed=seed, one_body_density=True
# )

# plt.plot(positions, one_body_density)
# plt.show()


# # system_omega_2 = nqs.NQS(
# #     nqs_repr="psi",
# #     backend=backend,
# #     log=True,
# #     logger_level="INFO",
# #     use_sr=False,  # Assuming you want to keep Stochastic Reconfiguration the same
# #     seed=seed,
# # )

# # system_omega_2.set_wf(
# #     wf_type,
# #     nparticles,
# #     dim,
# #     nhidden=nhidden,
# #     sigma2=1.0,
# # )

# # system_omega_2.set_sampler(mcmc_alg=mcmc_alg, scale=1.0)
# # system_omega_2.set_hamiltonian(type_="ho", int_type="Coulomb", omega=2.0)  # Changed omega to 2
# # system_omega_2.set_optimizer(
# #     optimizer=optimizer,
# #     eta=eta,
# #     beta1=0.9,
# #     beta2=0.999,
# #     epsilon=1e-8,
# # )

# # system_omega_2.train(
# #     max_iter=training_cycles[0],
# #     batch_size=batch_size,
# #     early_stop=False,
# #     seed=seed,
# # )

# # system_omega_2.sample(nsamples, nchains=nchains, seed=seed)

# # # Plotting psi2 for both wave functions
# # plt.figure(figsize=(10, 6))
# plot_psi2(system.wf, r_min=-4, r_max=4, num_points=300)
# # plot_psi2(system_omega_2.wf, r_min=-4, r_max=4, num_points=300)
# plt.legend()
# plt.xlabel("Position")
# plt.ylabel("Psi^2")
# plt.title("Comparison of Psi^2 for Different Omega Values")
# plt.show()
