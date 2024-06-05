import argparse
import cProfile  # noqa
import os
import pstats  # noqa

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from nqs.state import nqs
from nqs.state.utils import plot_2dobd  # noqa
from nqs.state.utils import plot_3dobd
from nqs.state.utils import plot_density_profile  # noqa
from nqs.state.utils import plot_psi  # noqa

jax.config.update("jax_platform_name", "cpu")

# Print device
print(jax.devices())


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def initialize_system(config):
    system = nqs.NQS(
        nqs_repr="psi",
        backend=config["backend"],
        logger_level="INFO",
        seed=config["seed"],
    )

    if config["nqs_type"] == "ffnn":
        base_layer_sizes = config["base_layer_sizes"][config["nqs_type"]]
        layer_sizes = [config["nparticles"] * config["dim"]] + base_layer_sizes
        activations = config["activations"].get(config["nqs_type"], [])
        common_kwargs = {
            "layer_sizes": layer_sizes,
            "activations": activations,
            "correlation": config["correlation"],
            "particle": config["particle"],
        }
        system.set_wf(
            config["nqs_type"], config["nparticles"], config["dim"], **common_kwargs
        )
    elif config["nqs_type"] == "ds":
        common_kwargs = {
            "layer_sizes": {
                "S0": [config["dim"]]
                + config["base_layer_sizes"][config["nqs_type"]]["S0"]
                + [config["latent_dim"]],
                "S1": [config["latent_dim"]]
                + config["base_layer_sizes"][config["nqs_type"]]["S1"],
            },
            "activations": {
                "S0": config["activations"].get(config["nqs_type"], [])["S0"],
                "S1": config["activations"].get(config["nqs_type"], [])["S1"],
            },
            "correlation": config["correlation"],
            "particle": config["particle"],
        }
        print("layer_sizes", common_kwargs["layer_sizes"])
        print("activations", common_kwargs["activations"])
        system.set_wf(
            config["nqs_type"], config["nparticles"], config["dim"], **common_kwargs
        )
    elif config["nqs_type"] == "rbm":
        system.set_wf(
            config["nqs_type"],
            config["nparticles"],
            config["dim"],
            nhidden=config["nhidden"],
            sigma2=1.0 / np.sqrt(config["omega"]),
            particle=config["particle"],
            correlation=config["correlation"],
        )
    else:
        # Handle other methods like VMC that don't use layers
        system.set_wf(
            config["nqs_type"],
            config["nparticles"],
            config["dim"],
            particle=config["particle"],
            correlation=config["correlation"],
        )

    return system


def run_experiment(config):
    system = initialize_system(config)
    mcmc_alg = config["mcmc_alg"]
    system.set_sampler(
        mcmc_alg=mcmc_alg,
        scale=(
            1 / np.sqrt(config["nparticles"] * config["dim"])
            if mcmc_alg == "m"
            else 0.1 / np.sqrt(config["nparticles"])
        ),
    )
    system.set_hamiltonian(
        type_="ho",
        int_type=config["interaction_type"],
        sigma_0=config["sigma_0"],
        omega=config["omega"],  # needs to be fixed to 1 to compare to drissi et al
        v_0=config["v_0"],
        r0_reg=10,
        training_cycles=config["training_cycles"],
    )
    system.set_optimizer(
        optimizer=config["optimizer"],
        eta=config["eta"] / np.sqrt(config["nparticles"] * config["dim"]),
    )

    if config["nqs_type"] == "ffnn":
        common_kwargs = {
            "layer_sizes": [config["nparticles"] * config["dim"]]
            + config["base_layer_sizes"][config["nqs_type"]],
            "activations": config["activations"].get(config["nqs_type"], []),
            "correlation": config["correlation"],
            "particle": config["particle"],
        }
        system.pretrain(
            model="Gaussian",
            max_iter=1000,
            batch_size=2000,
            logger_level="INFO",
            args=common_kwargs,
        )

    if config["nqs_type"] == "ds":
        common_kwargs = {
            "layer_sizes": {
                "S0": [config["dim"]]
                + config["base_layer_sizes"][config["nqs_type"]]["S0"]
                + [config["latent_dim"]],
                "S1": [config["latent_dim"]]
                + config["base_layer_sizes"][config["nqs_type"]]["S1"],
            },
            "activations": {
                "S0": config["activations"].get(config["nqs_type"], [])["S0"],
                "S1": config["activations"].get(config["nqs_type"], [])["S1"],
            },
            "correlation": config["correlation"],
            "particle": config["particle"],
        }
        system.pretrain(
            model="Gaussian",
            max_iter=1000,
            batch_size=2000,
            logger_level="INFO",
            args=common_kwargs,
        )

    history = system.train(  # noqa
        max_iter=config["training_cycles"],
        batch_size=config["batch_size"],
        early_stop=False,
        history=True,
        tune=False,
        grad_clip=0,
        seed=config["seed"],
    )

    df_all = system.sample(
        config["nsamples"],
        config["nchains"],
        config["seed"],
        one_body_density=False,
        save_positions=config["save_positions"],
    )

    # Mean values
    accept_rate_mean = df_all["accept_rate"].mean()

    # Combined standard error of the mean for energy
    means = df_all["energy"]
    std_errors = df_all["std_error"]

    # Calculate variances from standard errors
    variances = std_errors**2

    weights = 1 / variances
    combined_mean = np.sum(weights * means) / np.sum(weights)

    # Compute combined variance
    combined_variance = 1 / np.sum(weights)

    # Compute combined standard error
    combined_std_error = np.sqrt(combined_variance)

    # Construct the combined DataFrame
    combined_data = {
        "energy": [combined_mean],
        "std_error": [combined_std_error],
        "variance": [np.mean(df_all["variance"])],
        "accept_rate": [accept_rate_mean],
    }

    df_mean = pd.DataFrame(combined_data)
    final_energy = df_mean["energy"].values[0]
    final_error = df_mean["std_error"].values[0]
    error_str = f"{final_error:.0e}"
    error_scale = int(error_str.split("e")[-1])
    energy_decimal_places = -error_scale

    # Format energy to match the precision required by the error
    if energy_decimal_places > 0:
        energy_str = f"{final_energy:.{energy_decimal_places}f}"
    else:
        energy_str = f"{int(final_energy)}"

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
        config["nqs_type"],
        config["nparticles"],
        config["dim"],
        config["batch_size"],
        config["eta"] / np.sqrt(config["nparticles"] * config["dim"]),
        config["training_cycles"],
        config["nchains"] * config["nsamples"],
        config["optimizer"],
        config["particle"],
    ]

    df_mean.to_csv(config["output_filename"], index=False)
    print(df_mean)

    if config["save_positions"]:
        chain_id = 0
        filename = f"energies_and_pos_{config['nqs_type'].upper()}_ch{chain_id}.h5"
        plot_3dobd(filename, config["nsamples"], config["dim"])

    sns.scatterplot(data=df_all, x="chain_id", y="energy")
    plt.xlabel("Chain")
    plt.ylabel("Energy")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NQS Experiment")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Ensure the config path is relative to the project root
    config_path = os.path.join(os.path.dirname(__file__), args.config)

    config = load_config(config_path)
    run_experiment(config)

# def main():
#     parser = argparse.ArgumentParser(description="Run NQS Experiment")
#     parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
#     args = parser.parse_args()

#     # Ensure the config path is relative to the project root
#     config_path = os.path.join(os.path.dirname(__file__), args.config)

#     config = load_config(config_path)
#     run_experiment(config)

# if __name__ == "__main__":
#     # Use cProfile to run the main function and save the stats to a file
#     cProfile.runctx("main()", globals(), locals(), "profile_stats.prof")

#     # Create pstats object and sort by cumulative time
#     p = pstats.Stats("profile_stats.prof")
#     p.sort_stats("cumulative").print_stats(50)
