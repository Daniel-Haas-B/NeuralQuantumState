import sys

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import nqs package
sys.path.append("/Users/haas/Documents/Masters/GANQS/src/")

from nqs import nqs

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "../data/playground.csv"
nparticles = 2
dim = 1
nhidden = 4
nsamples = int(2**18)  # 2**18 = 262144
nchains = 4
eta = 0.05

training_cycles = [100_000]  # this is cycles for the NN
mcmc_alg = "m"
backend = "numpy"
optimizer = "gd"
batch_size = 5_000
detailed = True

seed = 42

dfs = []

for max_iter in training_cycles:
    system = nqs.RBMNQS(
        nparticles,
        dim,
        nhidden=nhidden,
        interaction=False,  # TOTUNE True
        nqs_repr="psi",
        backend=backend,
        log=True,
        use_sr=False,
    )

    system.init(sigma2=1.0, seed=seed)  # 1.3 for lmh
    system.set_sampler(mcmc_alg=mcmc_alg, scale=3.0)
    system.set_optimizer(
        optimizer=optimizer, eta=eta, use_sr=True, beta1=0.9, beta2=0.999, epsilon=1e-8
    )

    system.train(
        max_iter=max_iter,
        batch_size=batch_size,  # 1_000
        early_stop=False,
        seed=seed,
    )

    df = system.sample(nsamples, nchains=nchains, seed=seed)

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
                "nsamples",
                "training_cycles",
                "training_batch",
            ]
        ]
        .iloc[0]
        .to_dict()
    )
    data = {**mean_data, **info_data}  # ** unpacks the dictionary
    df_mean = pd.DataFrame([data])
    dfs.append(df_mean)

df_final = pd.concat(dfs)
# Save results
df_final.to_csv(output_filename, index=False)

# plot energy convergence curve

plt.plot(df["energy"])
plt.xlabel("Chain")
plt.ylabel("Energy")
plt.show()
