import sys

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import nqs package
sys.path.append("/Users/haas/Documents/Masters/GANQS/src/")

from nqs import nqs

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "../data/playground.csv"
nparticles = 10
dim = 2
nhidden = 4
nsamples = int(2**14)  # 2**18 = 262144
nchains = 8
eta = 0.05

training_cycles = [50_000]  # this is cycles for the NN
mcmc_alg = "m"
backend = "numpy"
optimizer = "gd"
batch_size = 1_000
detailed = True

seed = 42

dfs_mean = []
df = []
df_all = []
import time

# for max_iter in training_cycles:
start = time.time()
# for i in range(5):
for sr in [False, True]:
    system = nqs.NQS(
        nqs_repr="psi",
        backend=backend,
        log=True,
        use_sr=sr,
    )

    system.set_wf("rbm", nparticles, dim, nhidden=nhidden, sigma2=1.0)
    system.set_sampler(mcmc_alg=mcmc_alg, scale=1.0)
    system.set_hamiltonian("HO", int_type="Coulomb")
    system.set_optimizer(
        optimizer=optimizer,
        eta=eta,
        use_sr=True,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )

    system.train(
        max_iter=training_cycles[0],
        batch_size=batch_size,  # 1_000
        early_stop=False,
        seed=seed,
    )

    df = system.sample(nsamples, nchains=nchains, seed=seed)

    df_all.append(df)
    # plt.plot(np.convolve(energies[0], np.ones((100,))/100, mode='valid'))
    # plt.show()
    # exit()
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
                "nvisible",
                "nhidden",
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

# print(df)

# energy with sr
sns.lineplot(data=df_all, x="chain_id", y="energy", hue="sr")
plt.xlabel("Chain")
plt.ylabel("Energy")
plt.show()
