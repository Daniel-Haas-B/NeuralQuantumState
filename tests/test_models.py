import os

import jax
import numpy as np
import pandas as pd
import pytest

from . import test_utils
from nqs.state.nqs import NQS

# import test_utils

# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


@pytest.fixture(params=["vmc", "rbm", "ffnn", "ds"])
def config(request):
    return test_utils.get_config_from_yml(f"tests/config_{request.param}.yaml")


@pytest.fixture
def system(config):
    system = NQS(
        nqs_repr="psi",
        backend="jax",
        log=True,
        logger_level="INFO",
        seed=config["seed"],
    )
    system.set_wf(
        config["wf_type"],
        config["nparticles"],
        config["dim"],
        **config["common_kwargs"],
    )
    system.set_sampler(
        mcmc_alg=config["mcmc_alg"],
        scale=1 / np.sqrt(config["nparticles"] * config["dim"]),
    )
    system.set_hamiltonian(
        type_="ho",
        int_type="Coulomb",
        omega=1.0,
        r0_reg=1,
        training_cycles=config["training_cycles"],
    )
    system.set_optimizer(
        optimizer=config["optimizer"],
        eta=config["eta"],
        # Additional optimizer params can be set here if needed
    )
    return system


def test_nqs_system(system, config):
    df_all = []

    # Pretrain if ds or ffnn
    if config["wf_type"] in ["ds", "ffnn"]:
        system.pretrain(
            model="Gaussian", max_iter=100, batch_size=100, args=config["common_kwargs"]
        )

    system.train(
        max_iter=config["training_cycles"],
        batch_size=config["batch_size"],
        early_stop=False,
        seed=config["seed"],
        history=False,
        tune=False,
        grad_clip=0,
    )

    df = system.sample(
        config["nsamples"], nchains=config["nchains"], seed=config["seed"]
    )
    df_all.append(df)

    df_all = pd.concat(df_all)
    temp_csv = f"tests/test_temp_{config['wf_type']}.csv"
    df_all.to_csv(temp_csv, index=False)
    df_all_temp = pd.read_csv(temp_csv)
    df_all_test = pd.read_csv(f"tests/test_{config['wf_type']}.csv")

    assert df_all_temp.equals(df_all_test)
    os.remove(temp_csv)
