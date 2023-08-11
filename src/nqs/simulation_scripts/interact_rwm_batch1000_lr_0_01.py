import sys

import jax
import numpy as np
import pandas as pd

# import nqs

# Import nqs package
sys.path.insert(0, "../nqs/")
import nqs  # noqa

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "../data/interact_rwm_batch1000_lr_0_01.csv"
nparticles = 2  # particles
dim = 2  # dimensionality
nhidden_lst = [1, 2, 3, 4, 5, 6, 7, 8]  # hidden neurons
nsamples = int(2**18)
nchains = 8
max_iter = 500_000
batch_size = 1_000
eta = 0.01


dfs = []

for nhidden in nhidden_lst:
    system = nqs.NQS(
        nparticles,
        dim,
        nhidden=nhidden,
        interaction=True,
        mcmc_alg="rwm",
        nqs_repr="psi",
        backend="numpy",
        log=True,
    )

    system.init(sigma2=1.0, scale=1.0)

    system.train(
        max_iter=max_iter,
        batch_size=batch_size,
        gradient_method="adam",
        eta=eta,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        early_stop=False,
    )

    df = system.sample(nsamples, nchains=nchains, seed=None)

    sem_factor = 1 / np.sqrt(len(df))
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
    data = {**mean_data, **info_data}
    df_mean = pd.DataFrame([data])
    dfs.append(df_mean)

df_final = pd.concat(dfs)
# Save results
df_final.to_csv(output_filename, index=False)
