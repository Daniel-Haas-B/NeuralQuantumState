import nqs

# Config
nparticles = 1  # number of particles
dim = 1  # dimensionality
nhidden = 2  # hidden neurons
scale = 3.0  # Scale of proposal distribution
sigma2 = 1.0  # Variance of Gaussian layer in the RBM

# Instantiate the model
system = nqs.RBM(
    nparticles,
    dim,
    nhidden=nhidden,
    interaction={False, True},  # Repulsion or not
    mcmc_alg={"m", "mh"},  # MCMC algorithm
    nqs_repr={"psi", "psi2"},  # RBM representation
    backend={"numpy", "jax"},  # Closed-form or AD
    log=True,  # Show logger?
    logger_level="INFO",  # Logging level (see docs)
    rng=None,  # Set RNG engine (optional)
)

# Initialize parameters; biases and weight matrix are set automatically
system.init(sigma2=sigma2, scale=scale)

# Train the model
system.train(
    max_iter=500_000,  # No. of training iterations
    batch_size=1_000,  # No. samples used in one update
    optimizer={"gd", "adam"},  # Optimization algorithm
    eta=0.05,  # Learning rate
    beta1=0.9,  # ADAM hyperparameter
    beta2=0.999,  # ADAM hyperparameter
    epsilon=1e-8,  # ADAM hyperparameter
    mcmc_alg=None,  # Set MCMC algo. (optional)
    seed=None,  # Set for reproducibility
)

# Sample variational energy
df = system.sample(
    int(2**18),  # No. of energy samples
    nchains=4,  # No. of Markov chains
    seed=None,  # Set for reproducibility
)

# Results are returned in a pandas.DataFrame. Save directly with
system.to_csv("filename.csv")
