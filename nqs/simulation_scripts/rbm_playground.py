import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nqs.state import nqs
from nqs.state.utils import plot_2dobd
from nqs.state.utils import plot_3dobd
from nqs.state.utils import plot_density_profile
from nqs.state.utils import plot_psi


# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
print(jax.devices())
# Config
# output_filename = "/Users/haas/Documents/Masters/NQS/data/playground.csv"
output_filename = (
    "/Users/orpheus/Documents/Masters/NeuralQuantumState/data/playground.csv"
)
nparticles = 2
dim = 2
nhidden = 6

nsamples = int(2**12)
nchains = 1
eta = 0.1  # / np.sqrt(nparticles * dim)

training_cycles = 100  # this is cycles for the NN
mcmc_alg = "m"
backend = "jax"
optimizer = "adam"
batch_size = 500
detailed = True
nqs_type = "rbm"
particle = "boson"
trap_freq = 0.28
seed = 42
int_type = "coulomb"  # "None" "gaussian", "coulomb"
save_positions = True

dfs_mean = []
df = []
df_all = []
import time

# for max_iter in training_cycles:
start = time.time()
# for i in range(5):

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
    nhidden=nhidden,  # all after this is kwargs. In this example it is RBM dependent
    sigma2=1.0 / np.sqrt(trap_freq),
    particle=particle,
    correlation="none",
)

system.set_sampler(mcmc_alg=mcmc_alg, scale=1 / np.sqrt(nparticles * dim))
system.set_hamiltonian(
    type_="ho",
    int_type=int_type,
    omega=trap_freq,
    r0_reg=10,
    training_cycles=training_cycles,
    sigma_0=0.5,
    v0=0,
)
system.set_optimizer(
    optimizer=optimizer,
    eta=eta,
    # gamma=0.7,
    # beta1=0.9,
    # beta2=0.999,
    # epsilon=1e-8,
)

history = system.train(
    max_iter=training_cycles,
    batch_size=batch_size,
    early_stop=False,
    history=True,
    tune=False,
    grad_clip=0,
    seed=seed,
)

epochs = np.arange(len(history["energy"]))

for key, value in history.items():
    plt.plot(epochs, value, label=key)
    plt.legend()
    plt.show()

df_all = system.sample(
    nsamples, nchains, seed, one_body_density=False, save_positions=save_positions
)

# Mean values
energy_mean = df_all["energy"].mean()
accept_rate_mean = df_all["accept_rate"].mean()

# Combined standard error of the mean for energy
# https://stats.stackexchange.com/questions/231027/combining-samples-based-off-mean-and-standard-error
# i think we should instead block the combined chains
combined_std_error = np.mean(df_all["std_error"]) / np.sqrt(
    nchains
)  # this might be wrong

# Construct the combined DataFrame
combined_data = {
    "energy": [energy_mean],
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

# Save results
df_mean.to_csv(output_filename, index=False)

# plot energy convergence curve
# energy withour sr
print(df_all)

if save_positions:
    chain_id = 0  # TODO: make this general to get other chains
    filename = f"energies_and_pos_RBM_ch{chain_id}.h5"
    if dim == 2:
        plot_3dobd(filename, nsamples, dim)
    elif dim == 1:
        plot_2dobd(filename, nsamples, dim)
        plot_density_profile(filename, nsamples, dim)

if nchains > 1:
    sns.lineplot(data=df_all, x="chain_id", y="energy")
else:
    sns.scatterplot(data=df_all, x="chain_id", y="energy")

plt.xlabel("Chain")
plt.ylabel("Energy")
plt.show()

plot_psi(system, nparticles, dim)
