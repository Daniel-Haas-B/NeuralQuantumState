import jax
import matplotlib.pyplot as plt  # noqa
import numpy as np
import pandas as pd

from nqs.state.nqs import NQS
from nqs.state.utils import plot_3dobd

# print(jax.devices())
# import seaborn as sns
# import matplotlib.pyplot as plt

# from nqs.utils import plot_psi2


# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config

# output_filename = "/Users/haas/Documents/Masters/NQS/data/playground.csv"
output_filename = (
    "/Users/orpheus/Documents/Masters/NeuralQuantumState/data/playground.csv"
)

nparticles = 2
dim = 2
save_positions = True


nsamples = int(2**12)  # 2**18 = 262144
nchains = 1
eta = 0.001 / np.sqrt(nparticles * dim)  # 0.001  / np.sqrt(nparticles * dim)

training_cycles = 100  # this is cycles for the ansatz
mcmc_alg = "m"  # lmh is shit for ffnn
optimizer = "sr"
batch_size = 100
detailed = True
nqs_type = "dsffn"
seed = 42
latent_dimension = 10
particle = "boson"

system = NQS(
    nqs_repr="psi",
    backend="jax",
    logger_level="INFO",
    seed=seed,
)

common_layers_S0 = [14, 9, 7, 5, 3]
common_activations_S0 = ["gelu", "elu", "gelu", "elu", "gelu", "elu"]

layer_sizes = {
    "S0": [dim] + common_layers_S0 + [latent_dimension],
    "S1": [latent_dimension, 7, 5, 3, 1],
}

activations = {
    "S0": common_activations_S0,
    "S1": ["elu", "gelu", "elu", "linear"],
}

# Common kwargs for multiple function calls
common_kwargs = {
    "layer_sizes": layer_sizes,
    "activations": activations,
    "correlation": "none",  # or just j or None (default)
    "particle": particle,
}

# Initial function call with specific kwargs
system.set_wf("dsffn", nparticles, dim, **common_kwargs)

system.set_sampler(mcmc_alg=mcmc_alg, scale=1 / np.sqrt(nparticles * dim))
system.set_hamiltonian(
    type_="ho",
    int_type="coulomb",
    omega=0.1,
    r0_reg=10,
    training_cycles=training_cycles,
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
    system.pretrain(
        model="Gaussian", max_iter=1000, batch_size=1000, args=common_kwargs
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

    df_all = system.sample(nsamples, nchains, seed, save_positions=save_positions)

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

    df_mean.to_csv(output_filename, index=False)

    if save_positions:
        plot_3dobd("energies_and_pos_DS_ch0.h5", nsamples, dim)

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


if __name__ == "__main__":
    main()
