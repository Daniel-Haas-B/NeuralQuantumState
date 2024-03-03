import sys

sys.path.append("/Users/haas/Documents/Masters/GANQS/src/")
# sys.path.append("/home/daniel/home/daniel/test/GANQS/src/")

import jax

# import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd

# import seaborn as sns
import matplotlib.pyplot as plt

# from nqs.utils import plot_psi2

# Import nqs package


from nqs import anqs, nqs

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "../data/playground.csv"
nparticles = 2
dim = 2


nsamples = int(2**15)  # 2**18 = 262144
nchains = 1
eta_g = 0.001
eta_d = 0.01

training_cycles = 2000  # this is cycles for the ansatz
mcmc_alg = "m"  # lmh is shit for ffnn
opti_g = "adam"
opti_d = "adam"
batch_size = 1000

detailed = True
wf_type = "ffnn"
seed = 42

dfs_mean = []
df = []
df_all = []
import time

# for max_iter in training_cycles:
start = time.time()
# for i in range(5):


# Disc Setup

system_d = anqs.DNQS(
    backend="jax", batch_size=batch_size, log=True, logger_level="INFO", seed=seed
)

system_d.set_net(
    "ffnn",
    nparticles,
    dim,  # all after this is kwargs.
    layer_sizes=[
        nparticles * dim,  # should always be this
        5,
        3,
        1,  # should always be this
    ],
    activations=["relu", "relu", "sigmoid"],
)

system_d.set_optimizer(
    optimizer=opti_d,  # for disc, adam is just fine
    eta=eta_d,
    gamma=0,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
)

#######################################

system_g = anqs.GNQS(
    backend="jax",
    log=True,
    logger_level="INFO",
    seed=seed,
)

system_g.set_wf(
    "gffnn",
    nparticles,
    dim,  # all after this is kwargs.
    layer_sizes=[
        nparticles * dim,  # should always be this
        5,
        3,
        5,
        nparticles
        * dim,  # 1,#nparticles * dim # NOTICE how this is different from the ffnn case.
    ],
    activations=["elu", "elu", "elu", "linear"],
    symmetry="boson",
)

system_g.set_sampler(mcmc_alg=mcmc_alg, scale=1)
system_g.set_hamiltonian(
    type_="ho", int_type=None, omega=1.0, r0_reg=1, training_cycles=training_cycles
)

system_g.set_optimizer(
    optimizer=opti_g,
    eta=eta_g,
    gamma=0,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
)

#######################################
# train the truth generator

truth = nqs.NQS(
    nqs_repr="psi",
    backend="jax",
    log=True,
    logger_level="INFO",
    seed=seed,
)

truth.set_wf(
    "vmc",
    nparticles,
    dim,
    symmetry=None,
)

truth.set_sampler(mcmc_alg=mcmc_alg, scale=1.0 / np.sqrt(nparticles * dim))
truth.set_hamiltonian(
    type_="ho", int_type=None, omega=1.0, r0_reg=3, training_cycles=training_cycles
)
truth.set_optimizer(
    optimizer=opti_g,
    eta=0.1,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
)

history = truth.train(
    max_iter=100,
    batch_size=2000,
    early_stop=False,
    seed=seed,
    history=True,
)

df = truth.sample(nsamples, nchains=nchains, seed=seed)
print(df)


#######################################
system = anqs.ANQS(system_g, system_d)

system.set_truth(truth)


# system.pretrain(model="Gaussian", max_iter=1000, batch_size=1000)
loss_g, loss_d = system.train(
    max_iter=training_cycles,
    batch_size=batch_size,
    discriminator_updates=1,
    # early_stop=False,
    # seed=seed,
    # history=True,
    # tune=False,
    # grad_clip=0,
)

plt.plot(loss_g, label="Generator $log(D(G(z_i)))$")
plt.plot(loss_d, label="Discriminator $log D(x_i) + log(1 - D(G(z_i)))$")
plt.legend()
plt.show()

# generate fake and real samples and plot
gen_sampl, data_sampl = system.test_gen(samples=1000, particles=nparticles, dim=dim)

gen_sampl = gen_sampl.reshape(-1, dim)
data_sampl = data_sampl.reshape(-1, dim)

if dim == 1:
    plt.hist(gen_sampl, bins=50, alpha=0.5, label="Generated", color="orange")
    plt.hist(data_sampl, bins=50, alpha=0.5, label="Real", color="blue")

if dim == 2:
    plt.scatter(
        data_sampl[:, 0], data_sampl[:, 1], alpha=0.5, label="Real", color="blue"
    )
    plt.scatter(
        gen_sampl[:, 0],
        gen_sampl[:, 1],
        alpha=0.5,
        label="Generated",
        color="orange",
        marker="x",
    )


plt.legend()
plt.show()


# testing the discriminator
gen_sampl_probas, data_sampl_probas = system.test_disc(
    samples=1000, particles=nparticles, dim=dim
)

plt.hist(gen_sampl_probas, bins=50, alpha=0.5, label="Generated")
plt.hist(data_sampl_probas, bins=50, alpha=0.5, label="Real")

# add annotations
plt.legend()
plt.show()

# confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true = np.concatenate(
    [np.ones_like(data_sampl_probas), np.zeros_like(gen_sampl_probas)]
)
y_pred = np.concatenate([data_sampl_probas, gen_sampl_probas])

cm = confusion_matrix(y_true, y_pred > 0.5)
sns.heatmap(cm, annot=True, fmt="d")
# add true positive and true negative texts
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()


# epochs = np.arange(len(history["energy"]))
# for key, value in history.items():
#     plt.plot(epochs, value, label=key)
#     plt.legend()
#     plt.show()


# df = system.sample(nsamples, nchains=nchains, seed=seed)
# df_all.append(df)

# sem_factor = 1 / np.sqrt(len(df))  # sem = standard error of the mean
# mean_data = df[["energy", "std_error", "variance", "accept_rate"]].mean().to_dict()
# mean_data["sem_energy"] = df["energy"].std() * sem_factor
# mean_data["sem_std_error"] = df["std_error"].std() * sem_factor
# mean_data["sem_variance"] = df["variance"].std() * sem_factor
# mean_data["sem_accept_rate"] = df["accept_rate"].std() * sem_factor
# info_data = (
#     df[
#         [
#             "nparticles",
#             "dim",
#             "eta",
#             "scale",
#             # "nvisible",
#             # "nhidden",
#             "mcmc_alg",
#             "nqs_type",
#             "nsamples",
#             "training_cycles",
#             "training_batch",
#             "Opti",
#         ]
#     ]
#     .iloc[0]
#     .to_dict()
# )
# data = {**mean_data, **info_data}
# df_mean = pd.DataFrame([data])
# dfs_mean.append(df_mean)
# end = time.time()
# # print((end - start))

# df_final = pd.concat(dfs_mean)

# # Save results
# df_final.to_csv(output_filename, index=False)

# # plot energy convergence curve
# # energy withour sr
# df_all = pd.concat(df_all)
# print(df_all)
