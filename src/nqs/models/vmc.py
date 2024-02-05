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
        r = rng.standard_normal(size=self._N * self._dim)

        self._initialize_variational_params(rng)

        logp = self.logprob(r)  # log of the (absolute) wavefunction squared
        self.state = State(r, logp, 0, 0)

        if self.log:
            msg = f"""VMC initialized with {self._N} particles in {self._dim} dimensions with {
                    self.params.get("alpha").size
                    } parameters"""
            self.logger.info(msg)
        self.grads_performed = 0

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

        print("shape alpha", alpha.shape)
        print("shape r_2", r_2.shape)
        print("shape alpha_r_2", alpha_r_2.shape)

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
        alpha: (N*dim) array for the parameters.
        self.wf output is of size (batch_size, )
        """
        grad_wf_closure = jax.grad(self.wf, argnums=0)

        return vmap(grad_wf_closure, in_axes=(0, None))(
            r, alpha
        )  # 0, none will broadcast alpha to the batch size

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
        self.grads_performed += 1
        return grads_dict

    def grads_closure(self, r, alpha):
        """
        Return a function that computes the gradient of the log of the wavefunction squared
        """
        r2 = r * r  # element-wise multiplication

        return -2 * r2.sum(axis=-1)  # sum over particles

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
        laplacian = self.laplacian_closure(r, alpha)

        return laplacian

    def laplacian_closure(self, r, alpha):
        """
        Return a function that computes the laplacian of the wavefunction
        Remember in log domain, the laplacian is
        ∇^2 Ψ(r) = ∇^2 - ∑_{i=1}^{N} alpha_i r_i.T * r_i = -2 * alpha
        """
        # check if this is correct!
        return -2 * alpha.sum(axis=-1)

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

        return vmap(trace_fn)(hessian_at_r)

    def pdf(self, r):
        """
        Compute the probability distribution function
        """

        return self.backend.exp(self.logprob(r)) ** 2

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
