import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from nqs.utils import Parameter
from nqs.utils import State


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
        r = rng.standard_normal(size=(self._N * self._dim))

        self._initialize_variational_params(rng)

        logp = self.logprob(r)  # log of the (absolute) wavefunction squared
        self.state = State(r, logp, 0, 0)

        if self.log:
            msg = f"""VMC initialized with {self._N} particles in {self._dim} dimensions with {
                    self.params.get("alpha").size
                    } parameters"""
            self.logger.info(msg)

    def configure_backend(self, backend):
        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg
            self.grad_wf_closure = self.grad_wf_closure_jax
            self.grads_closure = self.grads_closure_jax
            self.laplacian_closure = self.laplacian_closure_jax
            self._jit_functions()
        else:
            raise ValueError("Invalid backend:", backend)

    def _jit_functions(self):
        functions_to_jit = [
            "logprob_closure",
            "wf",
            "grad_wf_closure",
            "laplacian_closure",
            "grads_closure",
        ]
        for func in functions_to_jit:
            setattr(self, func, jax.jit(getattr(self, func)))
        return self

    def wf(self, r, alpha):
        """
        Ψ(r)=exp(- ∑_{i=1}^{N} alpha_i r_i * r_i) but in log domain
        r: (batch_size, N * dim) array so
        alpha: (batch_size, N * dim) array
        """

        r_2 = r * r
        alpha_r_2 = alpha * r_2

        return -self.backend.sum(alpha_r_2, axis=-1)

    def logprob_closure(self, r, alpha):
        """
        Return a function that computes the log of the wavefunction squared
        """
        return self.wf(r, alpha).sum()  # maybe there should be a factor of 2 here?

    def logprob(self, r):
        """
        Compute the log of the wavefunction squared
        """
        alpha = self.params.get("alpha")  #
        return self.logprob_closure(r, alpha)

    def grad_wf_closure(self, r, alpha):
        """
            Return a function that computes the gradient of the wavefunction
        # TODO: check if this is correct CHECK DIMS
        """
        return -2 * alpha * r  # again, element-wise multiplication

    def grad_wf_closure_jax(self, r, alpha):
        """
        Returns a function that computes the gradient of the wavefunction with respect to r
        for each configuration in the batch.
        r: (batch_size, N*dim) array where each row is a flattened array of all particle positions.
        alpha: (batch_size, N*dim) array for the parameters.
        """
        batch_size = np.shape(r)[0] if np.ndim(r) > 1 else 1

        def scalar_wf(r_, alpha, i):
            wf_values = self.wf(r_, alpha)[i]
            return wf_values

        grad_wf = vmap(lambda i: jax.grad(scalar_wf, argnums=0)(r, alpha, i))(
            np.arange(batch_size)
        )

        return grad_wf

    def grad_wf(self, r):
        """
        Compute the gradient of the wavefunction
        """
        alpha = self.params.get("alpha")
        grads_alpha = self.grad_wf_closure(r, alpha)

        return grads_alpha

    def grads(self, r):
        """
        Compute the gradient of the log of the wavefunction squared
        """
        alpha = self.params.get("alpha")
        grads_alpha = self.grads_closure(r, alpha)

        grads_dict = {"alpha": grads_alpha}

        return grads_dict

    def grads_closure_jax(self, r, alpha):
        """
        Return a function that computes the gradient of the log of the wavefunction squared
        """
        batch_size = np.shape(r)[0] if np.ndim(r) > 1 else 1

        def scalar_wf(r_, alpha, i):
            wf_values = self.wf(r_, alpha)[i]
            return wf_values

        grads = vmap(lambda i: jax.grad(scalar_wf, argnums=1)(r, alpha, i))(
            np.arange(batch_size)
        )

        return grads

    def _initialize_vars(self, nparticles, dim, sigma2, rng, log, logger, logger_level):
        self._N = nparticles
        self._dim = dim
        self._sigma2 = sigma2
        self._log = log
        self._logger = logger
        self._logger_level = logger_level

    def _initialize_variational_params(self, rng):
        self.params = Parameter()
        self.params.set("alpha", rng.uniform(size=(self._N * self._dim)))

    def laplacian(self, r):
        """
        Compute the laplacian of the wavefunction
        """
        alpha = self.params.get("alpha")  # noqa
        return self.laplacian_closure(r, alpha)

    def laplacian_closure_jax(self, r, alpha):
        """
        Return a function that computes the laplacian of the wavefunction
        """

        def wrapped_wf(r_):
            return self.wf(r_, alpha)

        hessian_wf = vmap(jax.hessian(wrapped_wf))
        # Compute the Hessian for each element in the batch
        hessian_at_r = hessian_wf(r)

        def trace_fn(x):
            return jnp.trace(x)

        laplacian = vmap(trace_fn)(hessian_at_r)
        return laplacian

    def pdf(self, r):
        """
        Compute the probability distribution function
        """

        return self.backend.exp(self.logprob(r)) ** 2
