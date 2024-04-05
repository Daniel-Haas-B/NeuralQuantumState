import cProfile  # noqa
import pstats  # noqa

import jax
import numpy as np
import pandas as pd

from nqs.state import nqs
from nqs.state.utils import plot_obd


# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "/Users/haas/Documents/Masters/NQS/data/playground.csv"
nparticles = 2
dim = 2
save_positions = True

nsamples = int(2**17)  # 2**18 = 262144,
nchains = 1
eta = 0.001 / np.sqrt(nparticles * dim)  # 0.001  / np.sqrt(nparticles * dim)

training_cycles = 500  # this is cycles for the ansatz
mcmc_alg = "m"  # lmh is shit for ffnn
optimizer = "sr"
batch_size = 200  # 2000
detailed = True
wf_type = "ffnn"
seed = 42


save_positions = True

import time

start = time.time()

system = nqs.NQS(
    nqs_repr="psi",
    backend="jax",
    log=True,
    logger_level="INFO",
    seed=seed,
)

layer_sizes = [nparticles * dim, 14, 11, 9, 7, 3, 1]
activations = ["elu", "gelu", "elu", "gelu", "elu", "linear"]


common_kwargs = {
    "layer_sizes": layer_sizes,
    "activations": activations,
    "correlation": None,  # or just j or None (default)
    "symmetry": "none",  # why does this change the pretrain? and should it?
}

system.set_wf("ffnn", nparticles, dim, **common_kwargs)  # all after this is kwargs.

system.set_sampler(mcmc_alg=mcmc_alg, scale=1 / np.sqrt(nparticles * dim))
system.set_hamiltonian(
    type_="ho", int_type="none", omega=1.0, r0_reg=1, training_cycles=training_cycles
)

system.set_optimizer(
    optimizer=optimizer,
    eta=eta,
    # gamma=0,
    # beta1=0.9,
    # beta2=0.999,
    # epsilon=1e-8,
)


def main():

    dfs_mean = []
    df = []
    df_all = []
    system.pretrain(
        model="Gaussian",
        max_iter=100,
        batch_size=10000,
        args=common_kwargs,
    )
    history = system.train(
        max_iter=training_cycles,
        batch_size=batch_size,
        early_stop=False,
        seed=seed,
        history=True,
        tune=False,
        grad_clip=0,
    )

    epochs = np.arange(len(history["energy"]))  # noqa
    # for key, value in history.items():
    #     plt.plot(epochs, value, label=key)
    #     plt.legend()
    #     plt.show()

    df = system.sample(
        nsamples, nchains, seed, one_body_density=False, save_positions=save_positions
    )

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
    data = {**mean_data, **info_data}
    df_mean = pd.DataFrame([data])
    dfs_mean.append(df_mean)
    end = time.time()
    print((end - start))

    df_final = pd.concat(dfs_mean)

    # Save results
    df_final.to_csv(output_filename, index=False)

    df_all = pd.concat(df_all)
    print(df_all)

    if save_positions:
        plot_obd("positions_FFNN.h5", nsamples, dim)


if __name__ == "__main__":
    # Use cProfile to run the main function and save the stats to a file
    # main()
    cProfile.runctx("main()", globals(), locals(), "profile_stats.prof")

    # Create pstats object and sort by cumulative time
    p = pstats.Stats("profile_stats.prof")
    p.sort_stats("cumulative").print_stats(50)
