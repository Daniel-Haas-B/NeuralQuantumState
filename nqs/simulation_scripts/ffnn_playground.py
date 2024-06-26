import cProfile  # noqa
import pstats  # noqa

import jax
import numpy as np
import pandas as pd

from nqs.state import nqs
from nqs.state.utils import plot_3dobd  # noqa

# from nqs.state.utils import plot_pair_correlation  # noqa

# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
print(jax.devices())
# Config
output_filename = (
    "/Users/orpheus/Documents/Masters/NeuralQuantumState/data/playground.csv"
)
nparticles = 2
dim = 1
save_positions = True

nsamples = int(2**18)  # 2**18 = 262144,
nchains = 1
eta = 0.001 / np.sqrt(nparticles * dim)  # 0.001  / np.sqrt(nparticles * dim)

training_cycles = 100  # this is cycles for the ansatz
mcmc_alg = "m"  # lmh
optimizer = "sr"
batch_size = 300  # 2000
detailed = True
nqs_type = "ffnn"
seed = 42
logger_level = "info"
particle = "fermion_dots"
save_positions = True

import time

start = time.time()

system = nqs.NQS(
    nqs_repr="psi",
    backend="jax",
    logger_level=logger_level,
    seed=seed,
)

layer_sizes = [nparticles * dim, 9, 7, 3, 1]
activations = ["gelu", "elu", "gelu", "linear"]

common_kwargs = {
    "layer_sizes": layer_sizes,
    "activations": activations,
    "correlation": "none",  # or just j or None (default)
    "particle": particle,  # why does this change the pretrain? and should it?
}

system.set_wf("ffnn", nparticles, dim, **common_kwargs)  # all after this is kwargs.

system.set_sampler(mcmc_alg=mcmc_alg, scale=1 / np.sqrt(nparticles * dim))
system.set_hamiltonian(
    type_="ho",
    int_type="none",
    omega=1.0,
    # r0_reg=10,
    # training_cycles=training_cycles,
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
    import os

    file_name = "data/energies_and_pos_FFNN_ch0.h5"
    file_exists = os.path.isfile(file_name)
    overwrite = True
    if save_positions and file_exists:
        # prompt user saying which file is there and check if they want to overwrite
        print(f"File {file_name} already exists. Do you want to overwrite it?")
        ans = input("y/n: ")
        if ans != "y":
            overwrite = False

    if overwrite:
        df_all = []
        system.pretrain(
            model="Gaussian",
            max_iter=1000,
            batch_size=2000,
            logger_level=logger_level,
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

        df_all = system.sample(
            nsamples,
            nchains,
            seed,
            save_positions=save_positions,
        )

        # Mean values
        accept_rate_mean = df_all["accept_rate"].mean()

        # Extract means and standard errors
        means = df_all["energy"]
        std_errors = df_all["std_error"]

        # Calculate variances from standard errors
        variances = std_errors**2

        # Calculate weights based on variances
        weights = 1 / variances

        # Compute combined mean
        combined_mean = np.sum(weights * means) / np.sum(weights)

        # Compute combined variance
        combined_variance = 1 / np.sum(weights)

        # Compute combined standard error
        combined_std_error = np.sqrt(combined_variance)
        # Construct the combined DataFrame
        combined_data = {
            "energy": [combined_mean],
            "std_error": [combined_std_error],
            "variance": [
                np.mean(df_all["variance"])
            ],  # Keeping variance calculation for consistency
            "accept_rate": [accept_rate_mean],
        }

        df_mean = pd.DataFrame(combined_data)
        final_energy = df_mean["energy"].values[0]
        final_error = df_mean["std_error"].values[0]
        error_str = f"{final_error:.0e}"
        error_scale = int(
            error_str.split("e")[-1]
        )  # Extract exponent to determine error scale
        energy_decimal_places = -error_scale

        # Format energy to match the precision required by the error
        if energy_decimal_places > 0:
            energy_str = f"{final_energy:.{energy_decimal_places}f}"
        else:
            energy_str = (
                f"{int(final_energy)}"  # Convert to integer if no decimal places needed
            )

        # Get the first digit of the error for the parenthesis notation
        error_first_digit = error_str[0]

        # Remove trailing decimal point if it exists after formatting
        if energy_str[-1] == ".":
            energy_str = energy_str[:-1]

        formatted_energy = f"{energy_str}({error_first_digit})"
        df_mean["energy(error)"] = formatted_energy

        df_mean[
            [
                "nqs_type",
                "n_particles",
                "dim",
                "batch_size",
                "eta",
                "training_cycles",
                "nsamples",
                "Opti",
                "particle",
            ]
        ] = [
            nqs_type,
            nparticles,
            dim,
            batch_size,
            eta,
            training_cycles,
            nchains * nsamples,
            optimizer,
            particle,
        ]
        # # pretty display of df mean
        # dfs_mean.append(df_mean)
        print(df_mean)

    if save_positions:
        chain_id = 0  # TODO: make this general to get other chains
        file_name = f"energies_and_pos_FFNN_ch{chain_id}.h5"
        plot_3dobd(file_name, nsamples, dim)
        # plot_pair_correlation(file_name, nsamples, dr=0.1, max_range=5, dim=2)


if __name__ == "__main__":
    # Use cProfile to run the main function and save the stats to a file
    # main()
    cProfile.runctx("main()", globals(), locals(), "profile_stats.prof")

    # Create pstats object and sort by cumulative time
    p = pstats.Stats("profile_stats.prof")
    p.sort_stats("cumulative").print_stats(50)
