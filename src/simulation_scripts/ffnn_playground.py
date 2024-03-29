import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.state import nqs

# import seaborn as sns
# import matplotlib.pyplot as plt

# from nqs.utils import plot_psi2

# Import nqs package


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "/Users/haas/Documents/Masters/NQS/data/playground.csv"
nparticles = 2
dim = 2


nsamples = int(2**18)  # 2**18 = 262144,
nchains = 1
eta = 0.001 / np.sqrt(nparticles * dim)  # 0.001  / np.sqrt(nparticles * dim)

training_cycles = 100  # this is cycles for the ansatz
mcmc_alg = "m"  # lmh is shit for ffnn
optimizer = "sr"
batch_size = 2000  # 2000
detailed = True
wf_type = "ffnn"
seed = 42

dfs_mean = []
df = []
df_all = []
import time

start = time.time()

system = nqs.NQS(
    nqs_repr="psi",
    backend="jax",
    log=True,
    logger_level="INFO",
    seed=seed,
)

layer_sizes = [nparticles * dim, 14, 9, 7, 5, 3, 1]
activations = ["elu", "gelu", "elu", "gelu", "elu", "linear"]
common_kwargs = {
    "layer_sizes": layer_sizes,
    "activations": activations,
    "correlation": None,  # or just j or None (default)
    "symmetry": "fermion",  # why does this change the pretrain? and should it?
}

system.set_wf("ffnn", nparticles, dim, **common_kwargs)  # all after this is kwargs.

system.set_sampler(mcmc_alg=mcmc_alg, scale=1 / np.sqrt(nparticles * dim))
system.set_hamiltonian(
    type_="ho", int_type="Coulomb", omega=1.0, r0_reg=1, training_cycles=training_cycles
)

system.set_optimizer(
    optimizer=optimizer,
    eta=eta,
    # gamma=0,
    # beta1=0.9,
    # beta2=0.999,
    # epsilon=1e-8,
)

system.pretrain(model="Gaussian", max_iter=1200, batch_size=2000, args=common_kwargs)
history = system.train(
    max_iter=training_cycles,
    batch_size=batch_size,
    early_stop=False,
    seed=seed,
    history=True,
    tune=False,
    grad_clip=0,
)


epochs = np.arange(len(history["energy"]))
for key, value in history.items():
    plt.plot(epochs, value, label=key)
    plt.legend()
    plt.show()


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
            "Opti",
        ]
    ]
    .iloc[0]
    .to_dict()
)
data = {**mean_data, **info_data}
df_mean = pd.DataFrame([data])
dfs_mean.append(df_mean)
end = time.time()
# print((end - start))

df_final = pd.concat(dfs_mean)

# Save results
df_final.to_csv(output_filename, index=False)

# plot energy convergence curve
# energy withour sr
df_all = pd.concat(df_all)
print(df_all)


# energy with sr
# if nchains > 1:
#     sns.lineplot(data=df_all, x="chain_id", y="energy", hue="sr")
# else:
#     sns.scatterplot(data=df_all, x="chain_id", y="energy", hue="sr")
# ylim
# plt.ylim(2.9, 3.6)

# plt.xlabel("Chain")
# plt.ylabel("Energy")
# plt.show()

# plot probability

# positions, one_body_density = system.sample(
#     2**12, nchains=1, seed=seed, one_body_density=True
# )

# plt.plot(positions, one_body_density)
# plt.show()


# plot_psi2(system.wf, num_points=300, r_min=-5, r_max=5)
