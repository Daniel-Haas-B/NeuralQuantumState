from abc import abstractmethod

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
        print("rng", rng)
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

    def _jit_functions(self):
        functions_to_jit = [
            "_log_wf",
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
            self._jit_functions()
        else:
            raise ValueError("Invalid backend:", backend)

    @staticmethod
    def symmetry(func):
        def wrapper(self, r, *args, **kwargs):
            # Assuming `symmetry` is an attribute that determines if sorting should occur
            if self.symmetry == "boson":
                r_reshaped = r.reshape(-1, self.nparticles, self.dim)
                sort_indices = np.argsort(
                    r_reshaped.sum(axis=2), axis=1
                )  # sorts axis 1 over sum of axis 2
                r_sorted = np.array(
                    [
                        sample[indices]
                        for sample, indices in zip(r_reshaped, sort_indices)
                    ]
                )
                r_sorted_reshaped = r_sorted.reshape(-1, self.nparticles * self.dim)
                r_sorted_reshaped
            elif self.symmetry == "fermion":
                raise NotImplementedError  # TODO
            else:
                r_sorted_reshaped = r

            return func(self, r_sorted_reshaped, *args, **kwargs)

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
    def compute_sr_matrix(self, expval_grads, grads, shift=1e-3):
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
