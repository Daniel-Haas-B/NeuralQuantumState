import math  # noqa
from abc import abstractmethod
from itertools import permutations  # noqa

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as ss
from numpy import polynomial as P
from scipy.special import expit


class WaveFunction:
    """
    Base class for various wave functions used in quantum simulations.

    Attributes:
        params (dict): Parameters of the wave function.
        N (int): Number of particles.
        dim (int): Dimensionality of the system.
        _log (bool): Whether to enable logging.
        logger (logging.Logger): Logger for debugging and info messages.
        _logger_level (str): Level of the logger, defaults to 'INFO'.
        backend (str): Backend used for calculations, 'numpy' or 'jax'.
        _seed (int or None): Seed for random number generation.
        symmetry (str or None): Symmetry of the wave function, 'boson', 'fermion', or 'none'.
        jastrow (bool): Whether to use Jastrow factors.
        pade_jastrow (bool): Whether to use Pade-Jastrow factors.
        sqrt_omega (float): Square root of the oscillator frequency.
        slater_fact (float): Logarithm of the factorial of N, for normalization.

    Parameters:
        nparticles (int): The number of particles in the system.
        dim (int): The dimensionality of the system.
        rng (Generator, optional): Random number generator instance.
        log (bool, optional): Flag to enable logging. Defaults to False.
        logger (logging.Logger, optional): Logger instance. Defaults to None.
        logger_level (str, optional): Logging level. Defaults to 'INFO'.
        backend (str, optional): Specifies the computation backend. Defaults to 'numpy'.
        seed (int, optional): Seed for the random number generator. Defaults to None.
    """

    def __init__(
        self,
        nparticles,
        dim,
        rng=None,
        logger=None,
        logger_level="INFO",
        backend="numpy",
        seed=None,
    ):
        self.params = None
        self.N = nparticles
        self.dim = dim
        self.logger = logger
        self.logger_level = logger_level
        self.backend = backend
        self._seed = seed
        self.symmetry = None
        self.jastrow = False
        self.pade_jastrow = False
        self.sqrt_omega = 1  # will be reset in the set_hamiltonian

        self.slater_fact = np.log(
            ss.factorial(self.N)
        )  # move this to be after backend is set later

        if logger:
            self.logger = logger
        else:
            import logging

            self.logger = logging.getLogger(__name__)

        self.rng = rng
        self.r0 = self.rng.standard_normal(size=self.N * self.dim)

    def reinit_positions(self):
        """
        Reinitialize the particle positions with a standard normal distribution.
        """
        self.r0 = self.rng.standard_normal(size=self.N * self.dim)
        print("====== reinitiated positions to", self.r0)

    def _jit_functions(self):
        """
        JIT compile the wave function methods for performance using JAX, if available.
        """
        functions_to_jit = [
            "log_wf",
            # "log_wf0", # dont use this with deepset at least
            "log_wfc_jastrow",
            "logprob_closure",
            "set_0",
            "set_1",
            "compute_sr_matrix",
            "grad_wf_closure",
            "grad_params_closure",
            "laplacian_closure",
            "_precompute",
            "_softplus",
        ]
        for func in functions_to_jit:
            if hasattr(self, func):
                setattr(self, func, jax.jit(getattr(self, func)))

    def configure_backend(self, backend):
        """
        Configure the computation backend for the wave function operations.

        Parameters:
            backend (str): The backend to use, 'numpy' or 'jax'.

        Raises:
            ValueError: If an invalid backend is specified.
        """

        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
            self.sigmoid = expit
        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg
            self.sigmoid = expit  # jax.nn.sigmoid
            self.grad_wf_closure = self.grad_wf_closure_jax
            self.grad_params_closure = self.grad_params_closure_jax
            self.laplacian_closure = self.laplacian_closure_jax
            self.r0 = jnp.array(self.r0)
            self._jit_functions()
        else:
            raise ValueError("Invalid backend:", backend)

    def configure_symmetry(self, symmetry):
        """
        Configure the symmetry of the wave function.

        Parameters:
            symmetry (str): The symmetry to configure, 'boson', 'fermion', or 'none'.
        """
        self.symmetry = symmetry  # boson, fermion, none
        if self.symmetry == "fermion":  # then uses slater determinant
            self.log_wf = self.log_wf_slater_det
        else:
            self.log_wf = self.log_wf0
        if self.logger_level != "SILENT":
            self.logger.info(f"Using symmetry {symmetry}")

    def configure_correlation(self, correlation):
        """
        Configure the correlation factor for the wave function.

        Parameters:
            correlation (str): The correlation factor to use, 'pj' for Pade-Jastrow, 'j' for Jastrow, or 'none'.

        Notes:
            Will be called after configure_symmetry.
        """
        if correlation == "pj":
            self.pade_jastrow = True
            self.pade_aij = jnp.zeros((self.N, self.N))
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    # first N//2 particles are spin up, the rest are spin down
                    # there is a more efficient way to do this for sure
                    if i < self.N // 2 and j < self.N // 2:
                        self.pade_aij = self.pade_aij.at[i, j].set(1 / (self.dim + 1))
                    elif i >= self.N // 2 and j >= self.N // 2:
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

        if self.logger_level != "SILENT":
            self.logger.info(f"Using correlation factor {correlation}")

    def log_wf_slater_det(self, r, params):
        """
        Calculate the log of the wave function value considering only the Slater determinant,
        which represents the non-interacting case.

        Parameters:
            r (ndarray): An array of particle positions.
            params (dict): Parameters of the wave function.

        Returns:
            ndarray: Logarithm of the Slater determinant component of the wave function.
        """
        return self.log_wf0(r, params) + self.log_slater(r)

    def log_wf_jastrow(self, r, params):
        """
        Calculate the log of the wave function value considering only the Jastrow factor.

        Parameters:
            r (ndarray): An array of particle positions.
            params (dict): Parameters of the wave function, including Jastrow factor parameters.

        Returns:
            ndarray: Logarithm of the Jastrow factor component of the wave function.
        """
        return self.log_wf0(r, params) + self.log_jastrow(r, params)

    def log_wf_slater_jastrow(self, r, params):
        """
        Calculate the log of the wave function value using both the Slater determinant and Jastrow factor.

        Parameters:
            r (ndarray): An array of particle positions.
            params (dict): Parameters of the wave function, including Slater and Jastrow parameters.

        Returns:
            ndarray: Logarithm of the combined Slater and Jastrow wave function.
        """

        return (
            self.log_wf0(r, params) + self.log_slater(r) + self.log_jastrow(r, params)
        )

    def log_wf_pade_jastrow(self, r, params):
        """
        Calculate the log of the wave function value using both the Slater determinant and Pade-Jastrow factor.

        Parameters:
            r (ndarray): An array of particle positions.
            params (dict): Parameters of the wave function, including Pade-Jastrow factor parameters.

        Returns:
            ndarray: Logarithm of the combined Slater and Pade-Jastrow wave function.
        """
        return (
            self.log_wf0(r, params)
            + self.log_slater(r)
            + self.log_pade_jastrow(r, params)
        )

    def log_jastrow(self, r, params):
        """
        Compute the Jastrow factor component of the wave function.

        Parameters:
            r (ndarray): An array of particle positions reshaped into (-1, N, dim) dimensions.
            params (dict): Parameters of the wave function including the Jastrow weight matrix 'WJ'.

        Returns:
            ndarray: Computed Jastrow factor.
        """

        epsilon = 1e-10  # Small epsilon value was 10^-8 before
        r_cpy = r.reshape(-1, self.N, self.dim)
        r_diff = r_cpy[:, None, :, :] - r_cpy[:, :, None, :]
        r_dist = self.la.norm(r_diff + epsilon, axis=-1)  # Add epsilon to avoid nan

        rij = jnp.triu(r_dist, k=1)

        x = jnp.einsum("nij,ij->n", rij, params["WJ"])

        return x.squeeze(-1)

    def log_pade_jastrow(self, r, params):
        """
        Compute the Pade-Jastrow factor for the wave function, which is a specific type of Jastrow factor.

        Parameters:
            r (ndarray): An array of particle positions reshaped into (-1, N, dim) dimensions.
            params (dict): Parameters of the wave function including the Pade-Jastrow coefficient 'CPJ'.

        Returns:
            ndarray: Computed Pade-Jastrow factor.
        """

        epsilon = 1e-10  # Small epsilon value was 10^-8 before
        r_cpy = r.reshape(-1, self.N, self.dim)
        r_diff = r_cpy[:, None, :, :] - r_cpy[:, :, None, :]
        r_dist = self.la.norm(r_diff + epsilon, axis=-1)  # Add epsilon to avoid nan

        rij = jnp.triu(r_dist, k=1)

        num = self.pade_aij * rij  # elemntwise multiplication
        den = 1 + params["CPJ"] * rij  # elementwise addition

        x = jnp.einsum("nij,nij->n", num, 1 / den)

        return x.squeeze(-1)

    def generate_degrees(self):
        """
        Generate all possible combinations of degrees for the single-particle wave functions
        used in the Slater determinant.

        Returns:
            ndarray: Array of combinations of degrees with the shape (N//2, dim).
        """
        max_comb = self.N // 2
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
        Compute the logarithm of the Slater determinant for a system of particles, considering
        the spin decomposition in the ground state.

        Parameters:
            r (ndarray): An array of particle positions reshaped into (-1, N, dim) dimensions.

        Returns:
            ndarray: Computed logarithm of the Slater determinant for the given positions.

        The determinant D is calculated as follows, where phi_i is the i-th single particle wavefunction,
        which in our case is a Hermite polynomial:

        .. math::
            D = \\begin{vmatrix}
            \\phi_1(r_1) & \\phi_2(r_1) & \\cdots & \\phi_n(r_1) \\\\
            \\phi_1(r_2) & \\phi_2(r_2) & \\cdots & \\phi_n(r_2) \\\\
            \\vdots      & \\vdots      & \\ddots & \\vdots      \\\\
            \\phi_1(r_n) & \\phi_2(r_n) & \\cdots & \\phi_n(r_n)
            \\end{vmatrix}
        """
        A = self.N // 2
        r = r.reshape(-1, self.N, self.dim)

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
        Compute the product of Hermite polynomials for a given set of positions and degrees.

        Parameters:
            r (ndarray): An array of particle positions.
            degs (list of int): Degrees of the Hermite polynomials for each dimension.

        Returns:
            float: Product of Hermite polynomials for the given positions and degrees.
        """

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

    # @staticmethod
    # def symmetry(func):
    #     """
    #     #TODO: ensure that receiving args are size (bacth, nparticles, dim)
    #     """

    #     def wrapper(self, r, *args, **kwargs):
    #         n = self.N

    #         # first permutation is always the identity

    #         r_reshaped = r.reshape(-1, n, self.dim)

    #         def symmetrize_r(r):
    #             # print("r.shape[0]", r.shape[0])
    #             # first permutation is always the identity
    #             total = self.backend.zeros_like(
    #                 func(self, r, *args, **kwargs)
    #             )  # inneficient

    #             for sigma in permutations(range(n)):
    #                 permuted_r = r_reshaped[:, sigma, :]
    #                 # print("permuted r", permuted_r)
    #                 permuted_r = permuted_r.reshape(r.shape)
    #                 total += func(self, permuted_r, *args, **kwargs)

    #             return total / math.factorial(n)

    #         def antisymmetrize(func, *args):
    #             total = self.backend.zeros_like(
    #                 func(self, r, *args, **kwargs)
    #             )  # inneficient
    #             for sigma in permutations(range(n)):
    #                 permuted_r = r_reshaped[:, sigma, :]

    #                 inversions = 0
    #                 for i in range(len(sigma)):
    #                     for j in range(i + 1, len(sigma)):
    #                         if sigma[i] > sigma[j]:
    #                             inversions += 1
    #                 sign = (-1) ** inversions

    #                 permuted_r = permuted_r.reshape(r.shape)

    #                 total += sign * func(self, permuted_r, *args, **kwargs)
    #             return total / math.factorial(n)

    #         if self.symmetry == "boson":
    #             return symmetrize_r(r)

    #         elif self.symmetry == "fermion":
    #             return antisymmetrize(func, *args)
    #         else:
    #             return func(self, r, *args, **kwargs)

    #     return wrapper

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
    def grad_params(self, r):
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

    def activation(self, activation_str):
        match activation_str:
            case "tanh":
                return jnp.tanh
            case "sigmoid":
                return jax.nn.sigmoid
            case "relu":
                return jax.nn.relu
            case "softplus":
                return jax.nn.softplus
            case "gelu":
                return jax.nn.gelu
            case "linear":
                return lambda x: x
            case "exp":
                return jnp.exp
            case "elu":
                return jax.nn.elu
            case _:  # default
                raise ValueError(f"Invalid activation function {activation_str}")

    def rescale_parameters(self, factor):
        """
        Rescale the parameters of the wave function.
        """
        self.params.rescale(factor)

    def compute_sr_matrix(self, expval_grad_params, grad_params, shift=1e-3):
        """
        Compute the matrix for the stochastic reconfiguration algorithm.
        The expval_grad_params and grad_params should be dictionaries with keys
        "v_bias", "h_bias", "kernel" in the case of RBM. In the case of FFNN,
        we have "weights" and "biases", and "kernel" is not present.

        The expression here is for kernel element :math:`W_{ij}`:

        .. math::
            S_{ij,kl} = \\left\\langle \\left( \\frac{\\partial}{\\partial W_{ij}} \\log(\\psi) \\right)
            \\left( \\frac{\\partial}{\\partial W_{kl}} \\log(\\psi) \\right) \\right\\rangle
            - \\left\\langle \\frac{\\partial}{\\partial W_{ij}} \\log(\\psi) \\right\\rangle
            \\left\\langle \\frac{\\partial}{\\partial W_{kl}} \\log(\\psi) \\right\\rangle

        For bias (V or H) we have:

        .. math::
            :nowrap:

            S_{i,j} = \\left\\langle \\left( \\frac{\\partial}{\\partial V_{i}} \\log(\\psi) \\right) \\left( \\frac{\\partial}{\\partial V_{j}} \\log(\\psi) \\right) \\right\\rangle - \\left\\langle \\frac{\\partial}{\\partial V_{i}} \\log(\\psi) \\right\\rangle \\left\\langle \\frac{\\partial}{\\partial V_{j}} \\log(\\psi) \\right\\rangle

        Steps:

        1. Compute the gradient :math:`\\frac{\\partial}{\\partial W} \\log(\\psi)` using the ``_grad_kernel`` function.
        2. Compute the outer product of the gradient with itself: :math:`\\frac{\\partial}{\\partial W} \\log(\\psi) \\otimes \\frac{\\partial}{\\partial W} \\log(\\psi)`
        3. Compute the expectation value of the outer product over all the samples.
        4. Compute the expectation value of the gradient :math:`\\frac{\\partial}{\\partial W} \\log(\\psi)` over all the samples.
        5. Compute the outer product of the expectation value of the gradient with itself: :math:`\\langle \\frac{\\partial}{\\partial W} \\log(\\psi) \\rangle \\otimes \\langle \\frac{\\partial}{\\partial W} \\log(\\psi) \\rangle`

        Observation:

        :math:`\\langle \\frac{\\partial}{\\partial W_{ij}} \\log(\\psi) \\rangle` is already computed inside the training of the WF class, but we still need :math:`\\left\\langle \\left( \\frac{\\partial}{\\partial W_{ij}} \\log(\\psi) \\right) \\left( \\frac{\\partial}{\\partial W_{kl}} \\log(\\psi) \\right) \\right\\rangle`.

        Parameters:
            expval_grad_params (dict): Expected values of the gradients.
            grad_params (dict): Gradients of the parameters.
            shift (float): Shift used in the stochastic reconfiguration to avoid singularities.

        Returns:
            dict: The computed stochastic reconfiguration matrices.
        """
        sr_matrices = {}

        for key, grad_value in grad_params.items():

            if "W" in key:  # means it is a matrix
                grad_params_outer = self.backend.einsum(
                    "nij,nkl->nijkl", grad_value, grad_value
                )  # this is ∂_W log(ψ) ⊗ ∂_W log(ψ) for the batch
            else:  # means it is a (bias) vector
                grad_params_outer = self.backend.einsum(
                    "ni,nj->nij", grad_value, grad_value
                )

            # this is < (d/dW_i log(psi)) (d/dW_j log(psi)) > over the batch
            expval_outer_grad = self.backend.mean(grad_params_outer, axis=0)
            outer_expval_grad = self.backend.outer(
                expval_grad_params[key], expval_grad_params[key]
            )  # this is <∂_W log(ψ)> ⊗ <∂_W log(ψ)>

            sr_mat = (
                expval_outer_grad.reshape(outer_expval_grad.shape) - outer_expval_grad
            )
            sr_matrices[key] = sr_mat + shift * self.backend.eye(sr_mat.shape[0])

        return sr_matrices
