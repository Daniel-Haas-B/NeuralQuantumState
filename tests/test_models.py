import os

import jax
import numpy as np
import pandas as pd

from . import test_utils
from src.state.nqs import NQS

# import test_utils

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def test_deepset():
    df = []
    df_all = []

    config = test_utils.get_config_from_yml("tests/config_deepset.yaml")

    system = NQS(
        nqs_repr="psi",
        backend="jax",
        log=True,
        logger_level="INFO",
        seed=config["seed"],
    )
    system.set_wf("ds", config["nparticles"], config["dim"], **config["common_kwargs"])
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
        # gamma=0,
        # beta1=0.9,
        # beta2=0.999,
        # epsilon=1e-8,
    )

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
    # to csv to assure they have the same types and precision
    df_all.to_csv("tests/test_temp_deepset.csv", index=False)
    df_all_temp = pd.read_csv("tests/test_temp_deepset.csv")
    df_all_test = pd.read_csv("tests/test_deepset.csv")

    assert df_all_temp.equals(df_all_test)
    # delete temporary file
    os.remove("tests/test_temp_deepset.csv")


def test_ffnn():
    df = []
    df_all = []

    config = test_utils.get_config_from_yml("tests/config_ffnn.yaml")

    system = NQS(
        nqs_repr="psi",
        backend="jax",
        log=True,
        logger_level="INFO",
        seed=config["seed"],
    )
    system.set_wf(
        "ffnn", config["nparticles"], config["dim"], **config["common_kwargs"]
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
        # gamma=0,
        # beta1=0.9,
        # beta2=0.999,
        # epsilon=1e-8,
    )

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
    # to csv to assure they have the same types and precision
    df_all.to_csv("tests/test_temp_ffnn.csv", index=False)
    df_all_temp = pd.read_csv("tests/test_temp_ffnn.csv")
    df_all_test = pd.read_csv("tests/test_ffnn.csv")

    assert df_all_temp.equals(df_all_test)
    # delete temporary file
    os.remove("tests/test_temp_ffnn.csv")


def test_rbm():
    pass


def test_vmc():
    pass


if __name__ == "__main__":
    test_ffnn()
    test_deepset()
    test_rbm()
    test_vmc()
    print("All tests passed!")
