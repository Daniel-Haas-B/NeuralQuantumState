# This will be the exact diagonalization for
# the system of interacting bosons in a harmonic
# oscilator with coulomb interaction.
# We will here construct the Hamiltonian matrix
# and diagonalize it.
import sys

import jax.numpy as jnp
import numpy as np
import seaborn as sns

# import time
# import matplotlib.pyplot as plt
# import pandas as pd

sys.path.append("/Users/haas/Documents/Masters/GANQS/src/")
# from nqs import nqs

sns.set_theme()
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})


def hamiltonian(nparticles, dim, int_type, backend):
    """Get the exact Hamiltonian matrix for the system"""
    if backend == "numpy":
        la = np.linalg
    elif backend == "jax":
        la = jnp.linalg
    else:
        raise ValueError("Invalid backend:", backend)

    # Construct the Hamiltonian matrix
    H = np.zeros((2**nparticles, 2**nparticles))
    for i in range(2**nparticles):
        for j in range(2**nparticles):
            # Construct the basis states
            basis_i = np.array([int(b) for b in bin(i)[2:].zfill(nparticles)])
            basis_j = np.array([int(b) for b in bin(j)[2:].zfill(nparticles)])

            # Construct the interaction term
            v_int = 0.0
            if int_type == "Coulomb":
                r_dist = la.norm(basis_i[None, :] - basis_j[:, None], axis=-1)
                v_int = np.sum(np.triu(1 / r_dist, k=1))

            # Construct the trap term
            v_trap = 0.5 * np.sum(basis_i * basis_i)

            # Construct the full Hamiltonian matrix
            H[i, j] = v_trap + v_int

    return H


def get_exact_ground_state(nparticles, dim, int_type, backend):
    """Get the exact ground state of the system"""
    H = hamiltonian(nparticles, dim, int_type, backend)
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvals[0], eigvecs[:, 0]
