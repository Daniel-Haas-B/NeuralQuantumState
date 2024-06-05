import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nqs.state import nqs
from nqs.state.utils import plot_3dobd
from nqs.state.utils import plot_tbd  # noqa


# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# print device
print(jax.devices())

# Config
# output_filename = "/Users/haas/Documents/Masters/NQS/data/playground.csv"
output_filename = (
    "/Users/orpheus/Documents/Masters/NeuralQuantumState/data/playground.csv"
)

nparticles = 2
dim = 2
nsamples = int(2**19)  # 2**18 = 262144
nchains = 2
eta = 0.1  # / np.sqrt(nparticles * dim)  # 0.001  / np.sqrt(nparticles * dim)

training_cycles = 100  # this is cycles for the ansatz
mcmc_alg = "m"
backend = "jax"
optimizer = "adam"  # reminder: for adam, use bigger learning rate
batch_size = 500
detailed = True
nqs_type = "vmc"
seed = 42
save_positions = False
particle = "boson"
scale = (
    1.0 / np.sqrt(nparticles * dim) if mcmc_alg == "m" else 0.1 / np.sqrt(nparticles)
)
dfs_mean = []
df = []
df_all = []
import time

start = time.time()

system = nqs.NQS(
    nqs_repr="psi",
    backend=backend,
    logger_level="INFO",
    seed=seed,
)

system.set_wf(
    nqs_type,
    nparticles,
    dim,
    particle=particle,
    correlation="none",
)

system.set_sampler(mcmc_alg=mcmc_alg, scale=scale)
system.set_hamiltonian(
    type_="ho",
    int_type="coulomb",
    omega=1.0,
    r0_reg=10,
    training_cycles=training_cycles,
)
system.set_optimizer(
    optimizer=optimizer,
    eta=eta,
    # beta1=0.9,
    # beta2=0.999,
    # epsilon=1e-8,
)

history = system.train(
    max_iter=training_cycles,
    batch_size=batch_size,
    early_stop=False,
    seed=seed,
    history=True,
    tune=False,
)

df_all = system.sample(
    nsamples, nchains, seed, one_body_density=False, save_positions=save_positions
)

# Mean values
print(df_all)

accept_rate_mean = df_all["accept_rate"].mean()

# Extract means and standard errors
means = df_all["energy"]
std_errors = df_all["std_error"]

# Calculate variances from standard errors
variances = std_errors**2

# Calculate weights based on variances
weights = 1 / variances

# Compute combined mean
combined_mean = np.sum(weights * means) / np.sum(weights)

# Compute combined variance
combined_variance = 1 / np.sum(weights)

# Compute combined standard error
combined_std_error = np.sqrt(combined_variance)
# Construct the combined DataFrame
combined_data = {
    "energy": [combined_mean],
    "std_error": [combined_std_error],
    "variance": [
        np.mean(df_all["variance"])
    ],  # Keeping variance calculation for consistency
    "accept_rate": [accept_rate_mean],
}

df_mean = pd.DataFrame(combined_data)
final_energy = df_mean["energy"].values[0]
final_error = df_mean["std_error"].values[0]
error_str = f"{final_error:.0e}"
error_scale = int(error_str.split("e")[-1])  # Extract exponent to determine error scale
energy_decimal_places = -error_scale

# Format energy to match the precision required by the error
if energy_decimal_places > 0:
    energy_str = f"{final_energy:.{energy_decimal_places}f}"
else:
    energy_str = (
        f"{int(final_energy)}"  # Convert to integer if no decimal places needed
    )

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
    nqs_type,
    nparticles,
    dim,
    batch_size,
    eta,
    training_cycles,
    nchains * nsamples,
    optimizer,
    particle,
]
# # pretty display of df mean
# dfs_mean.append(df_mean)
print(df_mean)

end = time.time()
# print((end - start))
epochs = np.arange(training_cycles)

for key, value in history.items():
    plt.plot(epochs[: len(value)], value, label=key)
    plt.legend()
    plt.show()


# Save results
df_mean.to_csv(output_filename, index=False)

# plot energy convergence curve
# energy with sr
if save_positions:
    chain_id = 0  # TODO: make this general to get other chains
    plot_3dobd(f"energies_and_pos_VMC_ch{chain_id}.h5", nsamples, dim)


if nchains > 1:
    sns.lineplot(data=df_all, x="chain_id", y="energy")
else:
    sns.scatterplot(data=df_all, x="chain_id", y="energy")
# ylim
# plt.ylim(2.9, 3.6)

plt.xlabel("Chain")
plt.ylabel("Energy")
plt.show()
