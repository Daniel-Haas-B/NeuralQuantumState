import jax
import jax.numpy as jnp
import numpy as np
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
            self._jit_functions()
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
        return r.sum(axis=-1)

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
        grad_wf = jax.grad(self.wf, argnums=0)

        return grad_wf(r, alpha)

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
        grads_wf = jax.grad(
            self.wf, argnums=1
        )  # argnums=1 means we take the gradient wrt alpha

        return grads_wf(r, alpha)

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

        grad_wf = jax.grad(wrapped_wf)

        hessian_wf = jax.jacfwd(grad_wf)
        hessian_at_r = hessian_wf(r)
        laplacian = jnp.trace(hessian_at_r)

        return laplacian

    def pdf(self, r):
        """
        Compute the probability distribution function
        """
        alpha = self.params.get("alpha")
        return self.backend.abs(self.wf(r, alpha)) ** 2

    def compute_sr_matrix(self, expval_grads, grads, shift=1e-4):
        """
        expval_grads and grads should be dictionaries with keys "v_bias", "h_bias", "kernel" in the case of RBM
        in the case of FFNN we have "weights" and "biases" and "kernel" is not present
        WIP: for now this does not involve the averages because r will be a single sample
        Compute the matrix for the stochastic reconfiguration algorithm
            for now we do it only for the kernel
            The expression here is for kernel element W_ij:
                S_ij,kl = < (d/dW_ij log(psi)) (d/dW_kl log(psi)) > - < d/dW_ij log(psi) > < d/dW_kl log(psi) >

            For bias (V or H) we have:
                S_i,j = < (d/dV_i log(psi)) (d/dV_j log(psi)) > - < d/dV_i log(psi) > < d/dV_j log(psi) >


            1. Compute the gradient ∂_W log(ψ) using the _grad_kernel function.
            2. Compute the outer product of the gradient with itself: ∂_W log(ψ) ⊗ ∂_W log(ψ)
            3. Compute the expectation value of the outer product over all the samples
            4. Compute the expectation value of the gradient ∂_W log(ψ) over all the samples
            5. Compute the outer product of the expectation value of the gradient with itself: <∂_W log(ψ)> ⊗ <∂_W log(ψ)>

            OBS: < d/dW_ij log(psi) > is already done inside train of the RBM class but we need still the < (d/dW_ij log(psi)) (d/dW_kl log(psi)) >
        """
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
