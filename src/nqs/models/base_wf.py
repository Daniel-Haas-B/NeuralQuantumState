import math
from abc import abstractmethod
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as ss
from numpy import polynomial as P
from scipy.special import expit


class WaveFunction:
    def __init__(
        self,
        nparticles,
        dim,
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
        backend="numpy",
        seed=None,
        pade=False,
    ):
        self.params = None
        self.nparticles = nparticles
        self.dim = dim
        self._log = log
        self.logger = logger
        self._logger_level = logger_level
        self.backend = backend
        self._seed = seed
        self.symmetry = None
        self.jastrow = False
        self.pade_jastrow = False
        self.sqrt_omega = 1  # will be reset in the set_hamiltonian

        self.slater_fact = np.log(
            ss.factorial(self.nparticles)
        )  # move this to be after backend is set later

        if logger:
            self.logger = logger
        else:
            import logging

            self.logger = logging.getLogger(__name__)

        self.log = log
        self.rng = rng
        self.r0 = self.rng.standard_normal(size=self.nparticles * self.dim)

    def _reinit_positions(self):
        self.r0 = self.rng.standard_normal(size=self.nparticles * self.dim)
        print("====== reinitiated positions to", self.r0)

    def _jit_functions(self):
        functions_to_jit = [
            "log_wf",
            # "log_wf0",
            "log_wfc_jastrow",
            "logprob_closure",
            "set_0",
            "set_1",
            "wf",
            "compute_sr_matrix",
            # "grad_wf_closure", ## give diff results, strange
            # "grads_closure",
            # "laplacian_closure",
            "_precompute",
            "_softplus",
        ]
        for func in functions_to_jit:
            if hasattr(self, func):
                setattr(self, func, jax.jit(getattr(self, func)))

    def configure_backend(self, backend):
        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
            self.sigmoid = expit
        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg
            self.sigmoid = expit  # jax.nn.sigmoid
            self.grad_wf_closure = self.grad_wf_closure_jax
            self.grads_closure = self.grads_closure_jax
            self.laplacian_closure = self.laplacian_closure_jax
            self._jit_functions()  # maybe should be inside the child class
        else:
            raise ValueError("Invalid backend:", backend)

    def configure_symmetry(self, symmetry):
        self.symmetry = symmetry  # boson, fermion, none
        if self.symmetry == "fermion":  # then uses slater determinant
            self.log_wf = self.log_wf_slater_det
        else:
            self.log_wf = self.log_wf0

    def configure_correlation(self, correlation):
        """
        Note:
            Will be called after configure symmetry
        """
        if correlation == "pj":
            self.pade_jastrow = True
            self.pade_aij = jnp.zeros((self.nparticles, self.nparticles))
            for i in range(self.nparticles):
                for j in range(i + 1, self.nparticles):
                    # first N//2 particles are spin up, the rest are spin down
                    # there is a more efficient way to do this for sure
                    if i < self.nparticles // 2 and j < self.nparticles // 2:
                        self.pade_aij = self.pade_aij.at[i, j].set(1 / (self.dim + 1))
                    elif i >= self.nparticles // 2 and j >= self.nparticles // 2:
                        self.pade_aij = self.pade_aij.at[i, j].set(1 / (self.dim + 1))
                    else:
                        self.pade_aij = self.pade_aij.at[i, j].set(1 / (self.dim - 1))

            self.log_wf = self.log_wf_pade_jastrow

        elif correlation == "j":
            self.jastrow = True
            if self.symmetry == "fermion":
                self.log_wf = self.log_wf_slater_jastrow
            elif self.symmetry != "fermion":
                self.log_wf = self.log_wf_jastrow
            else:
                self.log_wf = self.log_wf0

        if self.logger is not None:
            self.logger.info(f"Using correlation factor {correlation}")

    def log_wf_slater_det(self, r, params):
        """
        this is the non interacting case, with only the slater determinant
        """
        return self.log_wf0(r, params) + self.log_slater(r)

    def log_wf_jastrow(self, r, params):
        """
        this is the jastraw factor only, without the slater determinant
        """
        return self.log_wf0(r, params) + self.log_jastrow(r, params)

    def log_wf_slater_jastrow(self, r, params):
        return (
            self.log_wf0(r, params) + self.log_slater(r) + self.log_jastrow(r, params)
        )

    def log_wf_pade_jastrow(self, r, params):
        return (
            self.log_wf0(r, params)
            + self.log_slater(r)
            + self.log_pade_jastrow(r, params)
        )

    def log_jastrow(self, r, params):
        """
        Only used when jastrow is True, and it is kinda irrespective of
        the NN architecture.
        TODO: One future idea is to make a NN for the "learning" of the Jastrow factor.
        """

        epsilon = 1e-10  # Small epsilon value was 10^-8 before
        r_cpy = r.reshape(-1, self._N, self._dim)
        r_diff = r_cpy[:, None, :, :] - r_cpy[:, :, None, :]
        r_dist = self.la.norm(r_diff + epsilon, axis=-1)  # Add epsilon to avoid nan

        rij = jnp.triu(r_dist, k=1)

        x = jnp.einsum("nij,ij->n", rij, params["WJ"])

        return x.squeeze(-1)

    def log_pade_jastrow(self, r, params):
        """ """

        epsilon = 1e-10  # Small epsilon value was 10^-8 before
        r_cpy = r.reshape(-1, self._N, self._dim)
        r_diff = r_cpy[:, None, :, :] - r_cpy[:, :, None, :]
        r_dist = self.la.norm(r_diff + epsilon, axis=-1)  # Add epsilon to avoid nan

        rij = jnp.triu(r_dist, k=1)

        num = self.pade_aij * rij  # elemntwise multiplication
        den = 1 + params["CPJ"] * rij  # elementwise addition

        x = jnp.einsum("nij,nij->n", num, 1 / den)

        return x.squeeze(-1)

    def generate_degrees(self):
        max_comb = self.nparticles // 2
        combinations = [[0] * self.dim]
        seen = {tuple(combinations[0])}

        while len(combinations) < max_comb:
            new_combinations = []
            for comb in combinations:
                for i in range(self.dim):
                    # Try incrementing each dimension by 1
                    new_comb = comb.copy()
                    new_comb[i] += 1
                    new_comb_tuple = tuple(new_comb)
                    if new_comb_tuple not in seen:
                        seen.add(new_comb_tuple)
                        new_combinations.append(new_comb)
                        if len(seen) == max_comb:
                            return np.array(combinations + new_combinations)
            combinations += new_combinations

        return np.array(combinations)

    def log_slater(self, r):
        """
        Decomposed spin Slater determinant in log domain.
        ln psi = ln det (D(up)) + ln det (D(down))
        In our ground state, half of the particles are spin up and half are spin down.
        We will also add the 1/sqrt(N!) normalization factor here.

        D = |phi_1(r_1) phi_2(r_1) ... phi_n(r_1)|
            |phi_1(r_2) phi_2(r_2) ... phi_n(r_2)|
            |   ...         ...          ...     |
            |phi_1(r_n) phi_2(r_n) ... phi_n(r_n)|

        where phi_i is the i-th single particle wavefunction, in our case it is a hermite polynomial.
        """
        A = self.nparticles // 2
        r = r.reshape(-1, self.nparticles, self.dim)

        r_up = r[:, :A, :]
        r_down = r[:, A:, :]

        # Compute the Slater determinant for the spin up particles
        D_up = self.backend.zeros((r.shape[0], A, A))
        D_down = self.backend.zeros((r.shape[0], A, A))

        degree_combs = self.generate_degrees()
        # print("r in log_slater", r)
        for part in range(A):
            for j in range(A):
                degrees = degree_combs[j]
                # print("degrees", degrees)
                # TODO: breaks here because hermite is giving an array
                D_up = D_up.at[:, part, j].set(self.hermite(r_up[:, part, :], degrees))
                D_down = D_down.at[:, part, j].set(
                    self.hermite(r_down[:, part, :], degrees)
                )

        # print("D_up", D_up)
        # print("D_down", D_down)

        # Compute the Slater determinant for the spin down particles
        log_slater_up = jnp.linalg.slogdet(D_up)[1].squeeze(-1)
        log_slater_down = jnp.linalg.slogdet(D_down)[1].squeeze(-1)

        return (
            log_slater_up + log_slater_down - 0.5 * self.slater_fact
        )  # the factor does not matter for energy but maybe important for the gradient?

    def hermite(self, r, degs):
        """
        Compute the product of Hermite polynomials for the given values, degree, and dimension.

        Parameters:
        - vals: Array-like of values for which to compute the Hermite polynomial product. It should be of shape (nbatch, dim)
        - degs: The degrees of the Hermite polynomials.
        - dim: The dimension, indicating how many values and subsequent polynomials to consider.

        Returns:
        - The product of Hermite polynomials for the given inputs.

        #TODO: move this to a helper function
        """
        # Error handling for input parameters
        # if not isinstance(vals, list) or not isinstance(deg, int) or not isinstance(self.dim, int):
        #    raise ValueError("Invalid input types for vals, deg, or dim.")
        # if len(vals) != dim:
        #    raise ValueError("Dimension mismatch between 'vals' and 'dim'.")

        # Compute the product of Hermite polynomials across the given dimensions
        hermite_product = 1

        for batch in range(r.shape[0]):
            # cartesian of r[batch] and degs

            for i in range(len(degs)):
                deg = degs[i]
                r_ = r[batch][i]

                # print(f"print(P.Hermite([0] * {deg} + [1])({r_}))")
                hermite_poly = P.Hermite([0] * deg + [1])(r_ * self.sqrt_omega)
                hermite_product *= hermite_poly

        return hermite_product

    @staticmethod
    def symmetry(func):
        """
        #TODO: ensure that receiving args are size (bacth, nparticles, dim)
        """

        def wrapper(self, r, *args, **kwargs):
            n = self.nparticles

            # first permutation is always the identity

            r_reshaped = r.reshape(-1, n, self.dim)

            def symmetrize_r(r):
                # print("r.shape[0]", r.shape[0])
                # first permutation is always the identity
                total = self.backend.zeros_like(
                    func(self, r, *args, **kwargs)
                )  # inneficient

                for sigma in permutations(range(n)):
                    permuted_r = r_reshaped[:, sigma, :]
                    # print("permuted r", permuted_r)
                    permuted_r = permuted_r.reshape(r.shape)
                    total += func(self, permuted_r, *args, **kwargs)

                return total / math.factorial(n)

            def antisymmetrize(func, *args):
                total = self.backend.zeros_like(
                    func(self, r, *args, **kwargs)
                )  # inneficient
                for sigma in permutations(range(n)):
                    permuted_r = r_reshaped[:, sigma, :]

                    inversions = 0
                    for i in range(len(sigma)):
                        for j in range(i + 1, len(sigma)):
                            if sigma[i] > sigma[j]:
                                inversions += 1
                    sign = (-1) ** inversions

                    permuted_r = permuted_r.reshape(r.shape)

                    total += sign * func(self, permuted_r, *args, **kwargs)
                return total / math.factorial(n)

            if self.symmetry == "boson":
                return symmetrize_r(r)

            elif self.symmetry == "fermion":
                return antisymmetrize(func, *args)
            else:
                return func(self, r, *args, **kwargs)

        return wrapper

    @abstractmethod
    def laplacian(self, r):
        """
        to be overwritten by the inheriting class
        """
        pass

    @abstractmethod
    def laplacian_closure(self, r):
        """
        to be overwritten by the inheriting class
        """
        pass

    @abstractmethod
    def laplacian_closure_jax(self, r):
        """
        to be overwritten by the inheriting class
        """
        pass

    @abstractmethod
    def pdf(self, r):
        """
        to be overwritten by the inheriting class
        """
        pass

    @abstractmethod
    def compute_sr_matrix(self, expval_grads, grads, shift=1e-6):
        """
        to be overwritten by the inheriting class
        """
        pass

    @abstractmethod
    def grads(self, r):
        """
        to be overwritten by the inheriting class
        """
        pass

    @abstractmethod
    def grad_wf(self, r):
        """
        to be overwritten by the inheriting class
        """
        pass

    @abstractmethod
    def grad_wf_closure(self, r):
        """
        to be overwritten by the inheriting class
        """
        pass

    @abstractmethod
    def grad_wf_closure_jax(self, r):
        """
        to be overwritten by the inheriting class
        """
        pass

    @abstractmethod
    def logprob(self, r):
        """
        to be overwritten by the inheriting class
        """
        pass

    @abstractmethod
    def logprob_closure(self, r):
        """
        to be overwritten by the inheriting class
        """
        pass

    @abstractmethod
    def wf(self, r):
        """
        Ouputs the wavefunction.
        to be overwritten by the inheriting class
        """
        pass
