import sys

import jax


# Import nqs package
sys.path.insert(0, "../nqs/")
import nqs  # noqa

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

nparticles = 2  # particles
dim = 2  # dimensionality
nhidden = 5  # hidden neurons

system = nqs.RBMNQS(
    nparticles,
    dim,
    nhidden=nhidden,
    interaction=True,
    mcmc_alg="lmh",
    nqs_repr="psi",
    backend="numpy",
    log=True,
)

# system.init(sigma2=1.0, scale=3.0)
system.init(sigma2=1.0, scale=1.0)
"""
system.tune(tune_iter=10_000,
            tune_interval=500,
            early_stop=False,  # set to True later
            rtol=1e-05,
            atol=1e-08,
            seed=None,
            mcmc_alg='rwm'
            )
"""
system.train(
    max_iter=50_000,
    batch_size=5_000,
    gradient_method="adam",
    eta=0.05,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    early_stop=True,  # set to True later
    rtol=1e-05,
    atol=1e-08,
    seed=None,
)

df = system.sample(10_000, nchains=4, seed=None, mcmc_alg=None)

# print(df)
print(df[["energy", "std_error", "variance", "accept_rate"]])
