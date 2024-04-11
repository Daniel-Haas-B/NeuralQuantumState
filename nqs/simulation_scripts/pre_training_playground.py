import jax  # noqa
import matplotlib.pyplot as plt
import numpy as np

from nqs.state import pretrain

print(jax.devices())
# import seaborn as sns
# import matplotlib.pyplot as plt

# from nqs.utils import plot_psi2

# Import nqs package


# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "../data/playground.csv"
nparticles = 2
dim = 2

nchains = 1
eta = 0.1

training_cycles = 5000  # this is cycles for the ansatz
mcmc_alg = "m"
optimizer = "adam"
batch_size = 10000
detailed = True
wf_type = "ffnn"
seed = 42

system = pretrain.Gaussian(logger_level="INFO", seed=seed, symmetry="fermion")

system.set_wf(
    "ffnn",
    nparticles,
    dim,  # all after this is kwargs.
    layer_sizes=[
        nparticles * dim,  # should always be this
        7,
        5,
        3,
        1,  # should always be this
    ],
    activations=["gelu", "gelu", "gelu", "linear"],
    symmetry="none",
)

system.set_sampler(mcmc_alg=mcmc_alg, scale=1)

system.set_optimizer(
    optimizer=optimizer,
    eta=eta,
    gamma=0,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
)

params, history = system.pretrain(
    max_iter=training_cycles,
    batch_size=batch_size,
    seed=seed,
    tune=False,
    grad_clip=0,
    pretrain_sampler=False,
    history=True,
)

# epochs = np.arange(len(history["loss"]))
# for key, value in history.items():
#     plt.plot(epochs, value, label=key)
#     plt.legend()
#     plt.show()


# sample from the ffnn
sigma = 2
inputs = np.random.standard_normal(
    size=(10000, nparticles * dim)
)  # np.random.uniform(-3*sigma, 3*sigma, size=(10000, nparticles * dim))


# if particles times dims is 2:
if nparticles * dim == 2:
    outputs = system.wf.pdf(inputs)
    truth = np.exp(system.multivar_gaussian_pdf(inputs))
    plt.plot(inputs, outputs, "o")
    # lineplot truth
    # sort inputs

    plt.plot(inputs, truth, "*")
    # from -2 to 2
    plt.xlim(-5, 5)

    plt.show()

# if particles times dims is 4 (need to plot 3d)
if nparticles * dim == 4:
    outputs = system.wf.pdf(inputs)
    truth = np.exp(system.multivar_gaussian_pdf(inputs))
    plt.plot(inputs, outputs, "o")

    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(inputs[:, 0], inputs[:, 1], outputs, c="r", marker="o")
    ax.scatter(inputs[:, 0], inputs[:, 1], truth, c="b", marker="o")

    # add legend
    ax.legend(["ffnn", "truth"])
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()
