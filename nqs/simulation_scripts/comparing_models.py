import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.state import nqs
from src.state.utils import plot_psi2  # noqa
from src.state.utils import plot_style

# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "../data/comparing_models.csv"
nparticles = 2
dim = 2
nsamples = int(2**10)  # 2**18 = 262144
nchains = 1
eta = 0.1

training_cycles = 10  # this is cycles for the ansatz
mcmc_alg = "m"
backend = "jax"
optimizer = "sr"
batch_size = 100
detailed = True
wf_types = ["rbm"]
seed = 42

dfs_mean = []
df = []
df_all = pd.DataFrame()
df_training_energies = pd.DataFrame()
for wf_type in wf_types:
    system = nqs.NQS(
        nqs_repr="psi",
        backend=backend,
        log=True,
        logger_level="INFO",
        seed=seed,
    )

    kwargs = {}
    if wf_type == "rbm":
        kwargs = {
            "nhidden": 4,
        }
    elif wf_type == "ffnn":
        kwargs = {
            "layer_sizes": [
                nparticles * dim,  # should always be this
                5,
                3,
                1,  # should always be this
            ],
            "activations": ["gelu", "elu", "linear"],
            "jastrow": False,
        }

    system.set_wf(wf_type, nparticles, dim, symmetry="none", **kwargs)

    system.set_sampler(mcmc_alg=mcmc_alg, scale=1.0 / np.sqrt(nparticles * dim))
    system.set_hamiltonian(
        type_="ho",
        int_type="Coulomb",
        omega=1.0,
        r0_reg=3,
        training_cycles=training_cycles,
    )
    if wf_type == "ffnn":
        # it does not make sense to pretrain an rbm in a regression problem, neither a vmc
        system.pretrain(model="Gaussian", max_iter=1000, batch_size=1000, args=kwargs)

    system.set_optimizer(
        optimizer=optimizer,
        eta=eta,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )

    history = system.train(
        max_iter=training_cycles,
        batch_size=batch_size,
        early_stop=False,
        seed=seed,
        history=True,
        tune=False,
    )
    epochs = np.arange(training_cycles)
    # for key, value in history.items():
    #     plt.plot(epochs, value, label=key)
    #     plt.legend()
    #     plt.show()

    # make sure they have the same length
    if len(history["energy"]) < training_cycles:
        history["energy"] = np.concatenate(
            [
                history["energy"],
                np.full(training_cycles - len(history["energy"]), np.nan),
            ]
        )
        history["std"] = np.concatenate(
            [history["std"], np.full(training_cycles - len(history["std"]), np.nan)]
        )

    z = 1.96  # 95% confidence interval
    df_training_energies = pd.concat(
        [
            df_training_energies,
            pd.DataFrame(
                {
                    "energy": history["energy"],
                    "std_error": np.array(history["std"]) * z / np.sqrt(batch_size),
                    "epoch": epochs,
                    "NQS": wf_type,
                }
            ),
        ]
    )

    df = system.sample(nsamples, nchains=nchains, seed=seed)
    df_all = pd.concat([df_all, df])

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

    data = {**mean_data, **info_data}  # ** unpacks the dictionary
    df_mean = pd.DataFrame([data])
    dfs_mean.append(df_mean)

    df_final = pd.concat(dfs_mean)

    # Save results
    df_final.to_csv(output_filename, index=False)

    # plot energy convergence curve
    # energy withour sr
    df_all = df_all.sort_values(by="chain_id")
    print(df_all)
    # # energy with sr
    # if nchains > 1:
    #     sns.lineplot(data=df_all, x="chain_id", y="energy")
    # else:
    #     sns.scatterplot(data=df_all, x="chain_id", y="energy")
    # # ylim
    # # plt.ylim(2.9, 3.6)

    # plt.xlabel("Chain")
    # plt.ylabel("Energy")
    # plt.show()

sns.lineplot(data=df_training_energies, x="epoch", y="energy", hue="NQS")
df_training_energies["upper"] = (
    df_training_energies["energy"] + df_training_energies["std_error"]
)
df_training_energies["lower"] = (
    df_training_energies["energy"] - df_training_energies["std_error"]
)
# Now, for each unique value in "NQS", plot the fill_between
unique_NQS = df_training_energies["NQS"].unique()

for nqs_ in unique_NQS:
    subset = df_training_energies[df_training_energies["NQS"] == nqs_]
    plt.fill_between(subset["epoch"], subset["lower"], subset["upper"], alpha=0.3)

plot_style.save("training_energies")
plt.show()
