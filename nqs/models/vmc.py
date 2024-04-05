import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from nqs.models.base_wf import WaveFunction
from nqs.state.utils import Parameter
from nqs.state.utils import State


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
        self.N = nparticles
        self.dim = dim

        logp = self.logprob(self.r0)  # log of the (absolute) wavefunction squared
        self.state = State(self.r0, logp, 0, 0)

        if self.log:
            msg = f"""VMC initialized with {self.N} particles in {self.dim} dimensions with {
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
        self.params.set("alpha", rng.uniform(size=(self.N * self.dim)))
        if self.jastrow:
            input_j_size = self.N * (self.N - 1) // 2
            limit = np.sqrt(2 / (input_j_size))
            self.params.set(
                "WJ", np.array(rng.uniform(-limit, limit, (self.N, self.N)))
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
