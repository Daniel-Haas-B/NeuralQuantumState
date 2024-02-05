import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from nqs.utils import Parameter
from nqs.utils import State


class Dummy:
    def __init__(
        self,
        nparticles,
        dim,
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
        backend="numpy",
    ):
        """
        this wave function is supposed to be used for testing purposes
        It is simply an identity function and the parameters have no effect
        """
        self.configure_backend(backend)
        self._initialize_vars(nparticles, dim, rng, log, logger, logger_level)

        if logger:
            self.logger = logger
        else:
            import logging

            self.logger = logging.getLogger(__name__)

        self.log = log
        self.rng = rng if rng else np.random.default_rng()
        r = rng.standard_normal(size=self._N * self._dim)

        self._initialize_variational_params(rng)

        logp = self.logprob(r)  # log of the (absolute) wavefunction squared
        self.state = State(r, logp, 0, 0)

        if self.log:
            msg = f"""Dummy initialized with {self._N} particles in {self._dim} dimensions.
            WARNING: this is a dummy wavefunction and the parameters have no effect"""
            self.logger.info(msg)

    def configure_backend(self, backend):
        if backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg
            self.grad_wf_closure = self.grad_wf_closure_jax
            self.grads_closure = self.grads_closure_jax
            self.laplacian_closure = self.laplacian_closure_jax
            # self._jit_functions()
        else:
            raise ValueError(f"Invalid backend: {backend}, only jax is supported")

    def _jit_functions(self):
        functions_to_jit = [
            "wf",
            "grad_wf_closure",
            "laplacian_closure",
            "logprob_closure",
            "grads_closure",
        ]
        for func in functions_to_jit:
            setattr(self, func, jax.jit(getattr(self, func)))
        return self

    def wf(self, r, alpha):
        """
        Ψ(r)= sum r
        r: (N, dim) array so that r_i is a dim-dimensional vector
        """
        return self.backend.sum(r, axis=0)  # r.sum(axis=0)

    def logprob_closure(self, r, alpha):
        return self.backend.log(self.backend.abs(self.wf(r, alpha)) ** 2)

    def logprob(self, r):
        """
        Return a function that computes the log of the wavefunction squared
        """
        alpha = self.params.get("alpha")
        return self.logprob_closure(r, alpha)

    def grad_wf_closure_jax(self, r, alpha):
        """
        Return a function that computes the gradient of the wavefunction
        """
        grad_wf_closure = jax.grad(self.wf, argnums=0)

        # either vmap(grad_wf_closure, in_axes=(0, None))(r, alpha) or we loop ourselves over ba

        return vmap(grad_wf_closure, in_axes=(0, None))(r, alpha)

    def grad_wf(self, r):
        """
        Compute the gradient of the wavefunction
        """
        alpha = self.params.get("alpha")
        return self.grad_wf_closure(r, alpha)

    def grads(self, r):
        """
        Compute the gradient of the log of the wavefunction squared
        """
        alpha = self.params.get("alpha")
        grads_alpha = self.grads_closure(r, alpha)  # note it does not depend on alpha

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
            self.backend.arange(batch_size)
        )

        return grads

    def _initialize_vars(self, nparticles, dim, rng, log, logger, logger_level):
        self._N = nparticles
        self._dim = dim
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

        hessian_at_r = hessian_wf(r)

        def trace_fn(x):
            return jnp.trace(x)

        return vmap(trace_fn)(hessian_at_r)

    def pdf(self, r):
        """
        Compute the probability distribution function
        """
        alpha = self.params.get("alpha")
        return self.backend.abs(self.wf(r, alpha)) ** 2

    def compute_sr_matrix(self, expval_grads, grads, shift=1e-4):
        """ """
        sr_matrices = {}

        for key, grad_value in grads.items():
            grad_value = self.backend.array(
                grad_value
            )  # this should be done outside of the function

            if self.backend.ndim(grad_value[0]) == 2:
                # if key == "kernel":
                grads_outer = self.backend.einsum(
                    "nij,nkl->nijkl", grad_value, grad_value
                )
            elif self.backend.ndim(grad_value[0]) == 1:
                # else:
                grads_outer = self.backend.einsum("ni,nj->nij", grad_value, grad_value)

            expval_outer_grad = self.backend.mean(grads_outer, axis=0)
            outer_expval_grad = self.backend.outer(expval_grads[key], expval_grads[key])

            sr_mat = (
                expval_outer_grad.reshape(outer_expval_grad.shape) - outer_expval_grad
            )
            sr_matrices[key] = sr_mat + shift * self.backend.eye(sr_mat.shape[0])

        return sr_matrices
