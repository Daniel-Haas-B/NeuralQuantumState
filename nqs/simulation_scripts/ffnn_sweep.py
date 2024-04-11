import jax
import wandb

print(jax.devices())
from nqs.state import nqs

# import seaborn as sns
# import matplotlib.pyplot as plt
# from nqs.utils import plot_psi2

# import numpy as np
# import pandas as pd


# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "../data/playground.csv"


dfs_mean = []
df = []
df_all = []


def main():
    run = wandb.init(project="ffnn", entity="danihaa", reinit=True)
    run_name = wandb.run.name

    print(f"#### Run name: {run_name}")

    config = run.config

    nparticles = 2
    dim = 2
    nchains = 1

    config.nsamples = int(2**10)  # 2**18 = 262144
    config.eta = 0.1
    config.training_cycles = 10000  # this is cycles for the ansatz
    config.batch_proportion = 0.1
    batch_size = int(config.training_cycles * config.batch_proportion)
    config.mcmc_alg = "m"  # "lmh"
    config.optimizer = "adam"  # "gd", "rmsprop", "adagrad"
    config.sr = False
    config.scale = 1
    config.tune = False
    config.clip = 0
    config.gamma = 0
    config.num_layers = 3
    layer_sizes = [5] + [3] * (config.num_layers - 2) + [1]
    # config.schedule = (
    #    "CyclicLR"  # "StepLR" # "CosineAnnealingLR" # "WarmRestart" # "CyclicLR"
    # )
    seed = 142

    system = nqs.NQS(
        nqs_repr="psi",
        backend="jax",
        logger_level="INFO",
        use_sr=config.sr,
        seed=seed,
    )

    system.set_wf(
        "ffnn",
        nparticles,
        dim,  # all after this is kwargs.
        layer_sizes=layer_sizes,  # now includes input and output layers
        activations=["gelu"] * (len(layer_sizes) - 1) + ["linear"],
        sigma2=1.0,
    )

    system.set_sampler(mcmc_alg=config.mcmc_alg, scale=config.scale)
    system.set_hamiltonian(type_="ho", int_type="Coulomb", omega=1.0)
    system.set_optimizer(
        optimizer=config.optimizer,
        eta=config.eta,
        gamma=config.gamma,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )

    system.train(
        max_iter=config.training_cycles,
        batch_size=batch_size,
        early_stop=False,
        seed=seed,
        history=True,
        tune=config.tune,
        grad_clip=config.clip,
        agent=wandb,
    )

    system.sample(config.nsamples, nchains=nchains, seed=seed)

    run.finish()


if __name__ == "__main__":
    main()
