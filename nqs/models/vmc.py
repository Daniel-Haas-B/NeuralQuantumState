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
        logger=None,
        logger_level="INFO",
        backend="numpy",
        particle=None,
        correlation=None,
    ):
        super().__init__(
            nparticles,
            dim,
            rng=rng,
            logger=logger,
            logger_level=logger_level,
            backend=backend,
        )

        self.configure_particle(particle)  # need to be before correlation
        self.configure_correlation(correlation)  # NEED TO BE BEFORE CONFIGURE_BACKEND
        self.configure_backend(backend)
        self._initialize_variational_params(rng)
        self.N = nparticles
        self.dim = dim

        sign, logp = self.logprob(self.r0)  # log of the (absolute) wavefunction squared
        self.state = State(self.r0, logp, 0, 0)

        if self.logger_level != "SILENT":
            msg = f"""VMC initialized with {self.N} particles in {self.dim} dimensions with {
                self.params.get("alpha").size
            } parameters"""
            self.logger.info(msg)

    # @WaveFunction.particle
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
        sign, value = self.log_wf(r, alpha)
        return sign, 2 * value

    def logprob(self, r):
        """
        Compute the log of the wavefunction squared
        """
        return self.logprob_closure(r, self.params)

    def grad_wf_closure(self, r, params):
        """ """

        return -2 * r * params.get("alpha")

    def grad_wf_closure_jax(self, r, params):
        """ """

        def value_log_wf(r, alpha):
            return self.log_wf(r, alpha)[1]

        grad_wf_closure = jax.grad(value_log_wf, argnums=0)
        return vmap(grad_wf_closure, in_axes=(0, None))(r, params)

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
        # this needs to be a parameter type, so that we can update the parameters

        grad = -r * r
        param_type_grad = Parameter()
        param_type_grad.set("alpha", grad)
        return param_type_grad

    def grad_params_closure_jax(self, r, alpha):
        """
        Return a function that computes the gradient of the log of the wavefunction squared
        """

        def value_log_wf(r, alpha):
            return self.log_wf(r, alpha)[1]

        grad_fn = vmap(jax.grad(value_log_wf, argnums=1), in_axes=(0, None))

        return grad_fn(r, alpha)

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
            limit = np.sqrt(2 / (100000))
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

        return self.backend.ones(r.shape[0]) * lap

    def laplacian_closure_jax(self, r, params):
        """
        Return a function that computes the laplacian of the wavefunction
        """

        def value_log_wf(r, alpha):
            return self.log_wf(r, alpha)[1]

        hessian_wf = vmap(jax.hessian(value_log_wf), in_axes=(0, None))
        trace_hessian = vmap(jnp.trace)

        return trace_hessian(hessian_wf(r, params))

    def pdf(self, r):
        """
        Compute the probability distribution function
        """

        return self.backend.exp(self.logprob(r)) ** 2
