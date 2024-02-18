import sys

sys.path.append("/Users/haas/Documents/Masters/GANQS/src/")
# sys.path.append("/home/daniel/home/daniel/test/GANQS/src/")

import jax

# import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
import matplotlib.pyplot as plt

# from nqs.utils import plot_psi2

# Import nqs package


from nqs import pretrain

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Config
output_filename = "../data/playground.csv"
nparticles = 1
dim = 1

nchains = 1
eta = 0.1

training_cycles = 500  # this is cycles for the ansatz
mcmc_alg = "m"  # lmh is shit for ffnn
optimizer = "adam"
batch_size = 10000
detailed = True
wf_type = "ffnn"
seed = 42

system = pretrain.Gaussian(
    nqs_repr="psi",
    log=True,
    logger_level="INFO",
    seed=seed,
)

system.set_wf(
    "ffnn",
    nparticles,
    dim,  # all after this is kwargs.
    layer_sizes=[
        nparticles * dim,  # should always be this
        5,
        3,
        1,  # should always be this
    ],
    activations=["gelu", "gelu", "linear"],
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
)

epochs = np.arange(len(history["loss"]))
for key, value in history.items():
    plt.plot(epochs, value, label=key)
    plt.legend()
    plt.show()
