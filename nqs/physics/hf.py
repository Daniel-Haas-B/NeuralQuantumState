import numpy as np
import scipy.linalg
import scipy.special
from itertools import combinations
import math
import tqdm
def harmonic_oscillator_basis_2d(nx, ny, x, y):
    """
    Compute the 2D harmonic oscillator wave function for quantum numbers nx and ny at positions x and y in natural units.
    """
    normalization_x = 1.0 / np.sqrt(2**nx * math.factorial(nx) * np.sqrt(np.pi))
    normalization_y = 1.0 / np.sqrt(2**ny * math.factorial(ny) * np.sqrt(np.pi))
    return (normalization_x * scipy.special.hermite(nx)(x) * np.exp(-x**2 / 2)) * \
           (normalization_y * scipy.special.hermite(ny)(y) * np.exp(-y**2 / 2))

def kinetic_energy_integral_2d(nx, ny, basis_functions):
    """
    Compute the kinetic energy integral for the (nx, ny)-th basis function in 2D.
    """
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    psi_nx_ny = basis_functions[nx, ny](X, Y)
    psi_nx_ny_xx = np.gradient(np.gradient(psi_nx_ny, x, axis=0), x, axis=0)
    psi_nx_ny_yy = np.gradient(np.gradient(psi_nx_ny, y, axis=1), y, axis=1)
    integrand = -0.5 * psi_nx_ny * (psi_nx_ny_xx + psi_nx_ny_yy)
    return np.trapz(np.trapz(integrand, x), y)

def potential_energy_integral_2d(nx, ny, basis_functions):
    """
    Compute the potential energy integral for the (nx, ny)-th basis function in 2D.
    """
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    psi_nx_ny = basis_functions[nx, ny](X, Y)
    integrand = 0.5 * (X**2 + Y**2) * psi_nx_ny**2
    return np.trapz(np.trapz(integrand, x), y)

def coulomb_integral_2d(nx1, ny1, nx2, ny2, nx3, ny3, nx4, ny4, basis_functions):
    """
    Compute the Coulomb integral between the (nx1, ny1) and (nx2, ny2) with (nx3, ny3) and (nx4, ny4) basis functions in 2D.
    """
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    psi_nx1_ny1 = basis_functions[nx1, ny1](X, Y)
    psi_nx2_ny2 = basis_functions[nx2, ny2](X, Y)
    psi_nx3_ny3 = basis_functions[nx3, ny3](X, Y)
    psi_nx4_ny4 = basis_functions[nx4, ny4](X, Y)
    integrand = psi_nx1_ny1 * psi_nx2_ny2 * psi_nx3_ny3 * psi_nx4_ny4 / np.sqrt((X - X.T)**2 + (Y - Y.T)**2 + 1e-9)
    return np.trapz(np.trapz(integrand, x), y)

def hartree_fock(basis_size_x, basis_size_y, n_particles, include_interaction=True, max_iter=100, tol=1e-6):
    """
    Perform Hartree-Fock calculation for a 2D harmonic oscillator system with or without Coulomb interaction.
    """
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    basis_functions = {(nx, ny): lambda X, Y, nx=nx, ny=ny: harmonic_oscillator_basis_2d(nx, ny, X, Y) for nx in range(basis_size_x) for ny in range(basis_size_y)}
    
    n_basis = basis_size_x * basis_size_y
    H_core = np.zeros((n_basis, n_basis))
    for nx1 in range(basis_size_x):
        for ny1 in range(basis_size_y):
            for nx2 in range(basis_size_x):
                for ny2 in range(basis_size_y):
                    if (nx1 == nx2) and (ny1 == ny2):
                        H_core[nx1 * basis_size_y + ny1, nx2 * basis_size_y + ny2] = (
                            kinetic_energy_integral_2d(nx1, ny1, basis_functions) +
                            potential_energy_integral_2d(nx1, ny1, basis_functions)
                        )

    # Initialize the density matrix
    P = np.zeros((n_basis, n_basis))

    # Initial guess for P using orthonormal basis
    _, coeffs = scipy.linalg.eigh(H_core)
    for i in range(n_particles):
        P += 2 * np.outer(coeffs[:, i], coeffs[:, i])

    def build_fock_matrix(P, include_interaction):
        F = H_core.copy()
        if include_interaction:
            for nx1 in range(basis_size_x):
                for ny1 in range(basis_size_y):
                    for nx2 in range(basis_size_x):
                        for ny2 in range(basis_size_y):
                            idx1 = nx1 * basis_size_y + ny1
                            idx2 = nx2 * basis_size_y + ny2
                            F[idx1, idx2] += sum(
                                P[nx3 * basis_size_y + ny3, nx4 * basis_size_y + ny4] *
                                (
                                    coulomb_integral_2d(nx1, ny1, nx2, ny2, nx3, ny3, nx4, ny4, basis_functions) -
                                    0.5 * coulomb_integral_2d(nx1, ny1, nx3, ny3, nx4, ny4, nx2, ny2, basis_functions)
                                )
                                for nx3 in range(basis_size_x) for ny3 in range(basis_size_y)
                                for nx4 in range(basis_size_x) for ny4 in range(basis_size_y)
                            )
        return F

    # Iterative Hartree-Fock procedure
    #use tqdm to show progress bar
    for iteration in tqdm.tqdm(range(max_iter)):
        F = build_fock_matrix(P, include_interaction)
        energies, C = scipy.linalg.eigh(F)
        P_new = np.zeros((n_basis, n_basis))
        for i in range(n_particles):
            P_new += 2 * np.outer(C[:, i], C[:, i])

        if np.linalg.norm(P_new - P) < tol:
            print(f"Converged after {iteration + 1} iterations")
            break

        P = P_new

    return energies, C

def main():
    raise NotImplementedError("This script is incorrect as is.")
    basis_size_x = 10
    basis_size_y = 10
    n_particles = 4

    include_interaction = False  # Set to False to compute energy without interaction

    energies, coeffs = hartree_fock(basis_size_x, basis_size_y, n_particles, include_interaction)
    
    print("Computed Hartree-Fock E_0:", energies[0])

if __name__ == "__main__":
    main()
