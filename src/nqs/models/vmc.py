import jax.numpy as jnp
import numpy as np
from nqs.utils import Parameter


class VMC:
    def __init__(
        self,
        nparticles,
        dim,
        sigma2=1.0,
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
        backend="numpy",
    ):
        self.configure_backend(backend)
        self._initialize_vars(nparticles, dim, sigma2, rng, log, logger, logger_level)

        if logger:
            self.logger = logger
        else:
            import logging

            self.logger = logging.getLogger(__name__)

        self.log = log
        self.rng = rng if rng else np.random.default_rng()
        r = rng.standard_normal(size=self._N * self._dim).reshape(  # noqa
            self._N, self._dim
        )  # noqa

        self._initialize_variational_params()

    def configure_backend(self, backend):
        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg
        else:
            raise ValueError("Invalid backend:", backend)

    def wf(self, r):
        """
        Ψ(r)=exp(- ∑_{i=1}^{N} alpha_i r_i.T * r_i) but in log domain
        r: (N, dim) array so that r_i is a dim-dimensional vector
        alpha: (N, dim) array so that alpha_i is a dim-dimensional vector
        """
        alpha = self.params.get("alpha")  #
        r_2 = r.T @ r  # (dim, dim)
        alpha_r_2 = alpha * r_2  # (dim, dim). this is element-wise multiplication
        return -self.backend.sum(alpha_r_2, axis=-1)  # axis=-1 means sum over dim

    def logprob_closure(self, r):
        """
        Return a function that computes the log of the wavefunction squared
        """
        return self.wf(r).sum()  # maybe there should be a factor of 2 here?

    def logprob(self, r):
        """
        Compute the log of the wavefunction squared
        """
        return self.logprob_closure(r)

    def grad_wf_closure(self, r, alpha):
        """
        Return a function that computes the gradient of the wavefunction
        """
        return -2 * alpha * r  # again, element-wise multiplication

    def grad_wf(self, r):
        """
        Compute the gradient of the wavefunction
        """
        alpha = self.params.get("alpha")
        return self.grad_wf_closure(r, alpha)

    def _initialize_vars(self, nparticles, dim, sigma2, rng, log, logger, logger_level):
        self._N = nparticles
        self._dim = dim
        self._sigma2 = sigma2
        self._log = log
        self._logger = logger
        self._logger_level = logger_level

    def _initialize_variational_params(self):
        self.params = Parameter()
        self.params.set(
            "alpha", self.rng.uniform(low=0.2, high=0.8, size=(self._dim, self._dim))
        )

    def laplacian_closure(self, r, alpha):
        """
        Return a function that computes the laplacian of the wavefunction
        """
        return

    def laplacian(self, r):
        """
        Compute the laplacian of the wavefunction
        """
        alpha = self.params.get("alpha")  # noqa
        return
