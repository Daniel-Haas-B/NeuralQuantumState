import sys

sys.path.append("/Users/haas/Documents/Masters/GANQS/src/")
import jax

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
import matplotlib.pyplot as plt

# from nqs.utils import plot_psi2

# Import nqs package


from nqs import nqs

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "../data/playground.csv"
nparticles = 2
dim = 2

nsamples = int(2**15)  # 2**18 = 262144
nchains = 2
eta = 0.1

training_cycles = 10000  # this is cycles for the ansatz
mcmc_alg = "lmh"
optimizer = "gd"
batch_size = 1000
detailed = True
wf_type = "ffnn"
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
        backend="jax",
        log=True,
        logger_level="INFO",
        use_sr=sr,
        seed=seed,
    )

    system.set_wf(
        "ffnn",
        nparticles,
        dim,  # all after this is kwargs.
        layer_sizes=[
            5,
            3,
            1,  # should always be this
        ],  # now includes input and output layers
        activations=["gelu", "gelu", "linear"],
        sigma2=1.0,
    )

    system.set_sampler(mcmc_alg=mcmc_alg, scale=1)
    system.set_hamiltonian(type_="ho", int_type="Coulomb", omega=1.0)
    system.set_optimizer(
        optimizer=optimizer,
        eta=eta,
        gamma=0,
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
        grad_clip=10,
    )

    epochs = np.arange(len(history["energy"]))
    plt.plot(epochs, history["energy"], label="energy")
    plt.legend()
    plt.show()
    plt.plot(epochs, history["grads"], label="gradient_norm")
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
                "sr",
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

positions, one_body_density = system.sample(
    2**12, nchains=1, seed=seed, one_body_density=True
)

plt.plot(positions, one_body_density)
plt.show()


# plot_psi2(system.wf, num_points=300, r_min=-5, r_max=5)
