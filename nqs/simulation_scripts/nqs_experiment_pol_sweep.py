import argparse
import cProfile  # noqa
import os
import pstats  # noqa

import jax
import numpy as np
import pandas as pd
import wandb
import yaml

from nqs.state import nqs
from nqs.state.utils import plot_2dobd  # noqa

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
    elif config["nqs_type"] == "dsffn":
        selected_arch = config["architecture"]
        print("selected_arch", selected_arch)
        # get values from local config
        base_layer_sizes = config["architectures"][selected_arch]["base_layer_sizes"][
            config["nqs_type"]
        ]
        base_activations = config["architectures"][selected_arch]["activations"][
            config["nqs_type"]
        ]
        common_kwargs = {
            "layer_sizes": {
                "S0": [config["dim"]] + base_layer_sizes["S0"] + [config["latent_dim"]],
                "S1": [config["latent_dim"]] + base_layer_sizes["S1"],
            },
            "activations": {
                "S0": base_activations["S0"],
                "S1": base_activations["S1"],
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
        print("running nhidden", config["nhidden"])
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
    print("running mcmc_alg", mcmc_alg)
    system.set_sampler(
        mcmc_alg=mcmc_alg,
        scale=(
            1 / np.sqrt(config["nparticles"] * config["dim"])
            if mcmc_alg == "m"
            else 0.1 / np.sqrt(config["nparticles"])
        ),
    )
    print("running v0", config["v_0"])
    print("running nqs", config["nqs_type"])

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
    common_kwargs = {}
    if config["nqs_type"] == "ffnn":
        common_kwargs = {
            "layer_sizes": [config["nparticles"] * config["dim"]]
            + config["base_layer_sizes"][config["nqs_type"]],
            "activations": config["activations"].get(config["nqs_type"], []),
            "correlation": config["correlation"],
            "particle": config["particle"],
        }

    if config["nqs_type"] == "dsffn":
        selected_arch = config["architecture"]
        print("selected_arch", selected_arch)
        # get values from local config
        base_layer_sizes = config["architectures"][selected_arch]["base_layer_sizes"][
            config["nqs_type"]
        ]
        base_activations = config["architectures"][selected_arch]["activations"][
            config["nqs_type"]
        ]

        common_kwargs = {
            "layer_sizes": {
                "S0": [config["dim"]] + base_layer_sizes["S0"] + [config["latent_dim"]],
                "S1": [config["latent_dim"]] + base_layer_sizes["S1"],
            },
            "activations": {
                "S0": base_activations["S0"],
                "S1": base_activations["S1"],
            },
            "correlation": config["correlation"],
            "particle": config["particle"],
        }
    if config["pretrain"]:
        system = pretrain(
            system,
            max_iter=1000,
            batch_size=500,
            args=common_kwargs,
        )

    history = system.train(  # noqa
        max_iter=config["training_cycles"],
        batch_size=config["batch_size"],
        early_stop=False,
        history=True,
        tune=True,  # just for sweep
        grad_clip=0,
        seed=config["seed"],
        agent=wandb,
    )

    # # save energies and stds to csv, together
    # df_hist = pd.DataFrame({"energy": history["energy"], "std_error": history["std"], "nqs_type": config["nqs_type"], "n_particles": config["nparticles"], "dim": config["dim"], "batch_size": config["batch_size"], "eta": config["eta"], "training_cycles": config["training_cycles"], "nsamples": config["nsamples"], "Opti": config["optimizer"], "particle": config["particle"]} )
    # if not os.path.exists(config["output_filename"] + f"energies_and_stds_v0_{config['v_0']}.csv"):
    #     df_hist.to_csv(config["output_filename"] + f"energies_and_stds_v0_{config['v_0']}.csv", index=False)
    # else:
    #     df_hist.to_csv(config["output_filename"] + f"energies_and_stds_v0_{config['v_0']}.csv", mode='a', header=False, index=False)

    df_all = system.sample(
        config["nsamples"],
        config["nchains"],
        config["seed"],
        save_positions=False,
        foldername=config["output_filename"],
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
        "v_0": [config["v_0"]],
        "omega": [config["omega"]],
        "pretrain": [config["pretrain"]],
        "sigma_0": [config["sigma_0"]],
        "correration": [config["correlation"]],
        "optimizer": [config["optimizer"]],
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

    print(df_mean)

    # if config["save_positions"]:
    #     chain_id = 0
    #     filename = f"energies_and_pos_{config['nqs_type'].upper()}_ch{chain_id}.h5"
    #     plot_3dobd(filename, config["nsamples"], config["dim"])

    # sns.scatterplot(data=df_all, x="chain_id", y="energy")
    # plt.xlabel("Chain")
    # plt.ylabel("Energy")
    # plt.show()


def pretrain(system, max_iter, batch_size, args):
    print("Pretraining with GAUSSIAN model first")
    if system.nqs_type == "ffnn" or system.nqs_type == "dsffn":
        system.pretrain(
            model="Gaussian",
            max_iter=max_iter,
            batch_size=batch_size,
            logger_level="INFO",
            args=args,
        )

    if config["v_0"] != 0:  # pretrain with v_0 = 0
        print("Pretraining with v_0 = 0 FIRST")
        system.set_hamiltonian(
            type_="ho",
            int_type=config["interaction_type"],
            sigma_0=config["sigma_0"],
            omega=config["omega"],  # will be fixed to 1 to compare to drissi et al
            v_0=0,
            r0_reg=10,
            training_cycles=config["training_cycles"],
        )
        system.train(  # noqa
            max_iter=config["training_cycles"],
            batch_size=config["batch_size"],
            early_stop=False,
            history=True,
            tune=False,
            grad_clip=0,
            seed=config["seed"],
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
    return system


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NQS Experiment")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Ensure the config path is relative to the project root

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    base_config = load_config(config_path)

    wandb.init(project="dsff_1d_fermions_N2_V0=-2")

    wand_config = wandb.config
    # from wanb.config get which architecture to run

    # combine the dictionaries
    config = {**base_config, **wand_config}

    # wandb config

    # if os.path.exists(f'{config["output_filename"]}final_results_{config["v_0"]}.csv'):
    #     os.remove(f'{config["output_filename"]}final_results_{config["v_0"]}.csv')
    # if os.path.exists(f'{config["output_filename"]}energies_and_stds_{config["v_0"]}.csv'):
    #     os.remove(f'{config["output_filename"]}energies_and_stds_{config["v_0"]}.csv')

    run_experiment(config)
