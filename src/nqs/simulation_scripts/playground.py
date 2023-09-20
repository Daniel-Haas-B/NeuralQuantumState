import sys

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import nqs package
sys.path.insert(0, "../nqs/")
import nqs  # noqa

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "../data/playground.csv"
nparticles = 2  # particles # TOTUNE 1, 2, 3, 4
dim = 2  # dimensionality # TOTUNE 1, 2, 3, 4
nhidden = 2  # hidden neurons # TOTUNE 1, 2, 3, 4
nsamples = int(2**18)
nchains = 4
eta = 0.05  # TOTUNE 0.05, 0.005

training_cycles = [100_000]
mcmc_alg = "rwm"
backend = "numpy"
gradient_method = "adam"
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
        mcmc_alg=mcmc_alg,
        nqs_repr="psi",
        backend=backend,
        log=True,
    )

    system.init(sigma2=1.0, scale=3.0, seed=seed)  # 1.3 for lmh

    system.train(
        max_iter=max_iter,
        batch_size=batch_size,  # 1_000
        gradient_method=gradient_method,
        eta=eta,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
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
