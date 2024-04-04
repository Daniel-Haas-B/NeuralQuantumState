import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from src.models.base_wf import WaveFunction
from src.state.utils import Parameter
from src.state.utils import State


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
        symmetry=None,
        correlation=None,
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

        self.configure_symmetry(symmetry)  # need to be before correlation
        self.configure_correlation(correlation)  # NEED TO BE BEFORE CONFIGURE_BACKEND
        self.configure_backend(backend)
        self._initialize_variational_params(rng)
        self._N = nparticles
        self._dim = dim

        logp = self.logprob(self.r0)  # log of the (absolute) wavefunction squared
        self.state = State(self.r0, logp, 0, 0)

        if self.log:
            msg = f"""VMC initialized with {self._N} particles in {self.dim} dimensions with {
                    self.params.get("alpha").size
                    } parameters"""
            self.logger.info(msg)

    # @WaveFunction.symmetry
    def log_wf0(self, r, params):
        """
        Ψ(r)=exp(- ∑_{i=1}^{N*DIM} alpha_i r_i * r_i) but in log domain
        r: (N * dim) array so that r_i is a dim-dimensional vector
        alpha: (N * dim) array so that alpha_i is a dim-dimensional vector
        """

        alpha = params.get("alpha")
        alpha_r_2 = alpha * r * r
        return -self.backend.sum(alpha_r_2, axis=-1)

    def logprob_closure(self, r, alpha):
        """
        Return a function that computes the log of the wavefunction squared
        """
        return 2 * self.log_wf(r, alpha)  # .sum()

    def logprob(self, r):
        """
        Compute the log of the wavefunction squared
        """
        return self.logprob_closure(r, self.params)

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

        return vmap(jax.grad(self.log_wf, argnums=0), in_axes=(0, None))(
            r, alpha
        )  # 0, none will broadcast alpha to the batch size

    def grad_wf(self, r):
        """
        Compute the gradient of the wavefunction
        """

        return self.grad_wf_closure(r, self.params)

    def grad_params(self, r):

        return self.grad_params_closure(r, self.params)

    def grad_params_closure(self, r, alpha):
        """
        Return a function that computes the gradient of the log of the wavefunction squared
        """
        return -r * r  # element-wise multiplication

    def grad_params_closure_jax(self, r, alpha):
        """
        Return a function that computes the gradient of the log of the wavefunction squared
        """

        grad_fn = vmap(jax.grad(self.log_wf, argnums=1), in_axes=(0, None))
        grad_eval = grad_fn(r, alpha)  # still a parameter type

        return grad_eval

    def _initialize_variational_params(self, rng):
        self.params = Parameter()
        self.params.set("alpha", rng.uniform(size=(self._N * self.dim)))
        if self.jastrow:
            input_j_size = self._N * (self._N - 1) // 2
            limit = np.sqrt(2 / (input_j_size))
            self.params.set(
                "WJ", np.array(rng.uniform(-limit, limit, (self._N, self._N)))
            )
        if self.pade_jastrow:
            assert not self.jastrow, "Pade Jastrow requires Jastrow to be false"
            self.params.set("CPJ", np.array(rng.uniform(-limit, limit, 1)))

    def laplacian(self, r):
        """
        Compute the laplacian of the wavefunction
        """

        laplacian = self.laplacian_closure(r, self.params)

        return laplacian

    def laplacian_closure(self, r, params):
        """
        Return a function that computes the laplacian of the wavefunction
        Remember in log domain, the laplacian is
        ∇^2 Ψ(r) = ∇^2 - ∑_{i=1}^{N} alpha_i r_i.T * r_i = -2 * alpha
        """
        # check if this is correct!
        alpha = params.get("alpha")
        lap = -2 * alpha.sum()
        lap_batch = self.backend.ones(r.shape[0]) * lap

        return lap_batch

    def laplacian_closure_jax(self, r, params):
        """
        Return a function that computes the laplacian of the wavefunction
        """

        hessian_wf = vmap(jax.hessian(self.log_wf), in_axes=(0, None))

        trace_hessian = vmap(jnp.trace)

        return trace_hessian(hessian_wf(r, params))

    def pdf(self, r):
        """
        Compute the probability distribution function
        """

        return self.backend.exp(self.logprob(r)) ** 2

    def compute_sr_matrix(self, expval_grad_params, grad_params, shift=1e-3):
        """
        Compute the matrix for the stochastic reconfiguration algorithm

            For alpha vector, we have:
                S_i,j = < (d/dalpha_i log(psi)) (d/dalpha_j log(psi)) > - < d/dalpha_i log(psi) > < d/dalpha_j log(psi) >


            1. Compute the gradient ∂_alpha log(ψ) using the grad_params function.
            2. Compute the outer product of the gradient with itself: ∂_W log(ψ) ⊗ ∂_W log(ψ) )
            3. Compute the expectation value of the outer product over all the samples
            4. Compute the expectation value of the gradient ∂_W log(ψ) over all the samples
            5. Compute the outer product of the expectation value of the gradient with itself: <∂_W log(ψ)> ⊗ <∂_W log(ψ)>

            OBS: < d/dW_ij log(psi) > is already done inside train of the NQS class (expval_grad_params) but we need still the < (d/dW_i log(psi)) (d/dW_j log(psi)) >
        """
        sr_matrices = {}

        for key, grad_value in grad_params.items():
            grad_value = self.backend.array(grad_value)

            grad_params_outer = self.backend.einsum(
                "ni,nj->nij", grad_value, grad_value
            )  # this is ∂_W log(ψ) ⊗ ∂_W log(ψ) for the batch
            expval_outer_grad = self.backend.mean(
                grad_params_outer, axis=0
            )  # this is < (d/dW_i log(psi)) (d/dW_j log(psi)) > over the batch
            outer_expval_grad = self.backend.einsum(
                "i,j->ij", expval_grad_params[key], expval_grad_params[key]
            )  # this is <∂_W log(ψ)> ⊗ <∂_W log(ψ)>

            sr_mat = expval_outer_grad - outer_expval_grad

            sr_matrices[key] = sr_mat + shift * self.backend.eye(sr_mat.shape[0])

        return sr_matrices
