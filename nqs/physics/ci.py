import numpy as np
import scipy.linalg
import scipy.special
from itertools import combinations
import math

def harmonic_oscillator_basis_2d(nx, ny, x, y):
    """
    Compute the 2D harmonic oscillator wave function for quantum numbers nx and ny at positions x and y in natural units.
    """
    normalization_x = 1.0 / np.sqrt(2**nx * math.factorial(nx) * np.sqrt(np.pi))
    normalization_y = 1.0 / np.sqrt(2**ny * math.factorial(ny) * np.sqrt(np.pi))
    return (normalization_x * scipy.special.hermite(nx)(x) * np.exp(-x**2 / 2)) * \
           (normalization_y * scipy.special.hermite(ny)(y) * np.exp(-y**2 / 2))

def construct_slater_determinants_2d(basis_size_x, basis_size_y, n_particles):
    """
    Construct all possible Slater determinants for a given basis size in 2D and number of particles.
    Each spatial orbital can be occupied by a spin-up or spin-down fermion.
    """
    orbitals = [(nx, ny, spin) for nx in range(basis_size_x) for ny in range(basis_size_y) for spin in ['up', 'down']]
    return list(combinations(orbitals, n_particles))

def compute_matrix_element_2d(det_i, det_j, basis_functions, n_particles):
    """
    Compute the Hamiltonian matrix element between two Slater determinants in 2D.
    """
    if det_i == det_j:
        # Diagonal elements: same Slater determinant
        kinetic_energy = sum(kinetic_energy_integral_2d(nx, ny, basis_functions) for (nx, ny, spin) in det_i)
        potential_energy = sum(potential_energy_integral_2d(nx, ny, basis_functions) for (nx, ny, spin) in det_i)
        coulomb_energy = sum(coulomb_integral_2d(nx1, ny1, nx2, ny2, basis_functions) 
                             for i, (nx1, ny1, spin1) in enumerate(det_i) 
                             for j, (nx2, ny2, spin2) in enumerate(det_i) if i < j)
        return kinetic_energy + potential_energy + coulomb_energy
    else:
        # Off-diagonal elements: not considered in this simple model
        return 0

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

def coulomb_integral_2d(nx1, ny1, nx2, ny2, basis_functions):
    """
    Compute the Coulomb integral between the (nx1, ny1) and (nx2, ny2)-th basis functions in 2D.
    """
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X1, Y1 = np.meshgrid(x, y)
    X2, Y2 = np.meshgrid(x, y)
    psi_nx1_ny1 = basis_functions[nx1, ny1](X1, Y1)
    psi_nx2_ny2 = basis_functions[nx2, ny2](X2, Y2)
    integrand = psi_nx1_ny1**2 * psi_nx2_ny2**2 / np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2 + 1e-9)
    return np.trapz(np.trapz(integrand, x), y)

def main():
    basis_size_x = 10
    basis_size_y = 10
    n_particles = 2

    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    basis_functions = {(nx, ny): lambda X, Y, nx=nx, ny=ny: harmonic_oscillator_basis_2d(nx, ny, X, Y) for nx in range(basis_size_x) for ny in range(basis_size_y)}
    det_basis = construct_slater_determinants_2d(basis_size_x, basis_size_y, n_particles)

    H = np.zeros((len(det_basis), len(det_basis)))

    for i, det_i in enumerate(det_basis):
        for j, det_j in enumerate(det_basis):
            H[i, j] = compute_matrix_element_2d(det_i, det_j, basis_functions, n_particles)
    
    energies, coeffs = scipy.linalg.eigh(H)
    
    print("Computed ground state energy:", energies[0])

if __name__ == "__main__":
    main()
