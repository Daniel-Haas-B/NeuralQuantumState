import math
from abc import abstractmethod
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
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
        symmetry=None,
    ):
        self.params = None
        self.nparticles = nparticles
        self.dim = dim
        self._log = log
        self.logger = logger
        self._logger_level = logger_level
        self.backend = backend
        self._seed = seed
        self.symmetry = symmetry

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
            "_log_wf",
            "log_wf",
            # "log_wf0",
            # "log_wfi",
            "logprob_closure",
            "wf",
            "compute_sr_matrix",
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
