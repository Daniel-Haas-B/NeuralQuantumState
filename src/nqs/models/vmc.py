import jax
import numpy as np
from jax import vmap
from nqs.models.base_wf import WaveFunction
from nqs.utils import Parameter
from nqs.utils import State


class VMC(WaveFunction):
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
        super().__init__(  # i know this looks weird
            nparticles,
            dim,
            rng=rng,
            log=log,
            logger=logger,
            logger_level=logger_level,
            backend=backend,
        )

        self.configure_backend(backend)
        self._initialize_variational_params(rng)

        logp = self.logprob(self.r0)  # log of the (absolute) wavefunction squared
        self.state = State(self.r0, logp, 0, 0)

        if self.log:
            msg = f"""VMC initialized with {self.nparticles} particles in {self.dim} dimensions with {
                    self.params.get("alpha").size
                    } parameters"""
            self.logger.info(msg)

    def wf(self, r, alpha):
        """
        Ψ(r)=exp(- ∑_{i=1}^{N*DIM} alpha_i r_i * r_i) but in log domain
        r: (N * dim) array so that r_i is a dim-dimensional vector
        alpha: (N * dim) array so that alpha_i is a dim-dimensional vector
        """

        r_2 = r * r
        alpha_r_2 = alpha * r_2

        return -self.backend.sum(alpha_r_2, axis=-1)

    def logprob_closure(self, r, alpha):
        """
        Return a function that computes the log of the wavefunction squared
        """
        return 2 * self.wf(r, alpha).sum()

    def logprob(self, r):
        """
        Compute the log of the wavefunction squared
        """
        alpha = self.params.get("alpha")
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
        Compute the gradient of the log of the wavefunction squared (why squared?)
        """
        alpha = self.params.get("alpha")
        grads_alpha = self.grads_closure(r, alpha)

        return {"alpha": grads_alpha}

    def grads_closure(self, r, alpha):
        """
        Return a function that computes the gradient of the log of the wavefunction squared
        """
        r2 = r * r  # element-wise multiplication

        return -r2

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

    def _initialize_variational_params(self, rng):
        self.params = Parameter()
        self.params.set("alpha", rng.uniform(size=(self.nparticles * self.dim)))

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
            return self.backend.trace(x)

        return vmap(trace_fn)(hessian_at_r)

    def pdf(self, r):
        """
        Compute the probability distribution function
        """

        return self.backend.exp(self.logprob(r)) ** 2

    def compute_sr_matrix(self, expval_grads, grads, shift=1e-3):
        """

        Compute the matrix for the stochastic reconfiguration algorithm

            For alpha vector, we have:
                S_i,j = < (d/dalpha_i log(psi)) (d/dalpha_j log(psi)) > - < d/dalpha_i log(psi) > < d/dalpha_j log(psi) >


            1. Compute the gradient ∂_alpha log(ψ) using the grads function.
            2. Compute the outer product of the gradient with itself: ∂_W log(ψ) ⊗ ∂_W log(ψ) )
            3. Compute the expectation value of the outer product over all the samples
            4. Compute the expectation value of the gradient ∂_W log(ψ) over all the samples
            5. Compute the outer product of the expectation value of the gradient with itself: <∂_W log(ψ)> ⊗ <∂_W log(ψ)>

            OBS: < d/dW_ij log(psi) > is already done inside train of the NQS class (expval_grads) but we need still the < (d/dW_i log(psi)) (d/dW_j log(psi)) >
        """
        sr_matrices = {}

        for key, grad_value in grads.items():
            grad_value = self.backend.array(grad_value)[
                0
            ]  # this is annoying, but first we need to convert the grads to a numpy or jax array. This zero index is also annoying, but it is because the grads are returned as a list of arrays due to the .items() method.

            grads_outer = self.backend.einsum(
                "ni,nj->nij", grad_value, grad_value
            )  # this is ∂_W log(ψ) ⊗ ∂_W log(ψ) for the batch
            expval_outer_grad = self.backend.mean(
                grads_outer, axis=0
            )  # this is < (d/dW_i log(psi)) (d/dW_j log(psi)) > over the batch
            outer_expval_grad = self.backend.einsum(
                "i,j->ij", expval_grads[key], expval_grads[key]
            )  # this is <∂_W log(ψ)> ⊗ <∂_W log(ψ)>

            sr_mat = expval_outer_grad - outer_expval_grad

            sr_matrices[key] = sr_mat + shift * self.backend.eye(sr_mat.shape[0])

        return sr_matrices
