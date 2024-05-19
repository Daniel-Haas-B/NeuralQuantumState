import jax
import numpy as np

from nqs import nqs

# import matplotlib.pyplot as plt
# import seaborn as sns
# Import nqs package

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")
print(jax.devices())
# Config
output_filename = "../data/playground.csv"
nparticles = 2
dim = 2


nsamples = int(2**16)  # 2**18 = 262144
nchains = 1
eta = 0.01

training_cycles = 100  # this is cycles for the ansatz
mcmc_alg = "m"  # lmh is shit for ffnn
optimizer = "sr"
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

system = nqs.NQS(
    nqs_repr="psi",
    backend="jax",
    logger_level="INFO",
    seed=seed,
)


def test_wf(particle, r, r_ex):
    print(f"\nReinitializing wave function with {particle} particle")
    latent_dimension = 3
    system.set_wf(
        "deepset",
        nparticles,
        dim,  # all after this is kwargs.
        layer_sizes={
            "S0": [
                dim,  # should always be this
                3,
                latent_dimension,  # should always be this
            ],
            "S1": [
                latent_dimension,
                2,
                1,  # should always be this
            ],
        },
        activations={
            "S0": ["gelu", "elu"],
            "S1": ["gelu", "linear"],
        },
    )
    print("Output without particle exchange: ", system.wf.ds(r))
    print("Output with particle exchange: ", system.wf.ds(r_ex))
    # system.set_wf(
    #     "ffnn",
    #     nparticles,
    #     dim,
    #     layer_sizes=[nparticles * dim, 5, 3, 1],
    #     activations=["gelu", "elu", "linear"],
    #     particle=particle,
    # )

    # print("Output without particle exchange: ", system.wf.ffnn(r, system.wf.params))
    # print("Output with particle exchange: ", system.wf.ffnn(r_ex, system.wf.params))


# Test case for two particles
print("=== Test Case: One Particle ===")
nparticles = 1
dim = 2
r = np.array([[1, 2], [3, 4]])
r_ex = np.array([[1, 2], [3, 4]])

print("Fake input:", r)
print("Exchanged r:", r_ex)


# # Test with boson
# test_wf("boson", r, r_ex)

# # Test with fermion
# test_wf("fermion", r, r_ex)

test_wf("deepset", r, r_ex)

print("=== Test Case: Two Particles ===")
nparticles = 2
dim = 2
r = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
r_ex = np.array([[3, 4, 1, 2], [7, 8, 5, 6]])

print("Fake input:", r)
print("Exchanged r:", r_ex)


# # Test with boson symmetry
# test_wf("boson", r, r_ex)

# # Test with fermion symmetry
# test_wf("fermion", r, r_ex)

test_wf("deepset", r, r_ex)

print("\n=== Test Case: Three Particles ===")
# Test case with three particles
nparticles = 3
dim = 2
r = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
r_ex = np.array([[3, 4, 1, 2, 5, 6], [9, 10, 7, 8, 11, 12]])


print("Fake input:", r)
print("Exchanged r:", r_ex)

# Repeating tests for three particles

# # Test with boson symmetry
# test_wf("boson", r, r_ex)

# # Test with fermion symmetry
# test_wf("fermion", r, r_ex)

test_wf("deepset", r, r_ex)


print("\n=== Test Case: Three Particles, ONE DIM     ===")
# Test case with three particles
nparticles = 3
dim = 1
r = np.array([[1, 2, 3], [7, 8, 9]])
r_ex = np.array([[3, 2, 1], [9, 8, 7]])


print("Fake input:", r)
print("Exchanged r:", r_ex)

# Repeating tests for three particles

# # Test with boson symmetry
# test_wf("boson", r, r_ex)

# # Test with fermion symmetry
# test_wf("fermion", r, r_ex)

test_wf("deepset", r, r_ex)


print("\n=== Test Case: Three Particles batch size 1===")
# Test case with three particles
nparticles = 3
dim = 2
r = np.array([1, 2, 3, 4, 5, 6])
r_ex = np.array([3, 4, 1, 2, 5, 6])


print("Fake input:", r)
print("Exchanged r:", r_ex)

# Repeating tests for three particles

# # Test with boson symmetry
# test_wf("boson", r, r_ex)

# # Test with fermion symmetry
# test_wf("fermion", r, r_ex)

test_wf("deepset", r, r_ex)
