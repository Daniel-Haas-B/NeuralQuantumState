import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from src.models.base_wf import WaveFunction
from src.state.utils import Parameter
from src.state.utils import State


class RBM(WaveFunction):
    def __init__(
        self,
        nparticles,
        dim,
        nhidden=1,
        factor=1.0,
        sigma2=1.0,
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
        backend="numpy",
        symmetry=None,
        correlation=None,
    ):
        """
        Initializes the RBM Neural Network Quantum State.

        Args:
        - nparticles (int): Number of particles.
        - dim (int): Dimensionality.
        ...
        """
        super().__init__(
            nparticles,
            dim,
            rng=rng,
            log=log,
            logger=logger,
            logger_level=logger_level,
            backend=backend,
        )

        self._initialize_vars(nparticles, dim, nhidden, factor, sigma2)
        self.configure_symmetry(symmetry)  # need to be before correlation
        self.configure_correlation(correlation)  # NEED TO BE BEFORE CONFIGURE_BACKEND
        self.configure_backend(backend)

        self._initialize_bias_and_kernel(rng)

        logp = self.logprob(self.r0)
        self.state = State(self.r0, logp, 0, 0)

        if self.log:
            neuron_str = "neurons" if self.Nhidden > 1 else "neuron"
            msg = (
                f"Neural Network Quantum State initialized as RBM with "
                f"{self.Nhidden} hidden {neuron_str}"
            )
            self.logger.info(msg)

    def __call__(self, r):
        return self.wf(
            r,
            self.params.get("v_bias"),
            self.params.get("h_bias"),
            self.params.get("W_kernel"),
        )

    def _initialize_bias_and_kernel(self, rng):
        v_bias = rng.standard_normal(size=self.Nvisible) * 0.01
        h_bias = rng.standard_normal(size=self.Nhidden) * 0.01
        kernel = rng.standard_normal(size=(self.Nvisible, self.Nhidden))
        kernel *= np.sqrt(1 / self.Nvisible)
        self.params = Parameter()
        self.params.set(["v_bias", "h_bias", "W_kernel"], [v_bias, h_bias, kernel])
        if self.jastrow:
            input_j_size = self.N * (self.N - 1) // 2
            limit = np.sqrt(2 / (input_j_size))
            self.params.set(
                "WJ", np.array(rng.uniform(-limit, limit, (self.N, self.N)))
            )
        if self.pade_jastrow:
            assert not self.jastrow, "Pade Jastrow requires Jastrow to be false"
            self.params.set("CPJ", np.array(rng.uniform(-limit, limit, 1)))

    def _initialize_vars(self, nparticles, dim, nhidden, factor, sigma2):
        self._sigma2 = sigma2
        self._factor = factor
        self._rbm_psi_repr = 2 * self._factor
        self.N = nparticles
        self.dim = dim
        self.Nvisible = self.N * self.dim
        self.Nhidden = nhidden
        self._precompute()

    def _precompute(self):
        self._sigma4 = self._sigma2 * self._sigma2
        self._sigma2_factor = 1.0 / self._sigma2
        self._sigma2_factor2 = 0.5 / self._sigma2

    def _softplus(self, x):
        """Softplus activation function.

        Computes the element-wise function
                softplus(x) = log(1 + e^x)
        """
        return self.backend.logaddexp(x, 0)

    def log_wf0(self, r, params):
        """Logarithmic gaussian-binary RBM"""
        # visible layer
        x_v = self.la.norm(
            r - params.get("v_bias"), axis=-1
        )  # axis = -1 means the last axis which is correct no matter the dimension of r

        x_v *= -x_v * self._sigma2_factor2  # this will also be broadcasted correctly

        # hidden layer
        x_h = self._softplus(
            params.get("h_bias") + (r @ params.get("W_kernel")) * self._sigma2_factor
        )  # potential failure point
        x_h = self.backend.sum(x_h, axis=-1)
        # print("x_h + x_v", x_h + x_v)

        return x_v + x_h

    def wf(self, r, params):
        """Evaluate the wave function
        This factor is 2 because we are evaluating the wave function squared.
        So p_rbm = |Ψ(r)|^2 = exp(-2 * log(Ψ(r))) which in log domain is -2 * log(Ψ(r))
        """

        return self._factor * self.log_wf(r, params)

    def pdf(self, r):
        """
        Probability amplitude
        """
        return self.backend.exp(self.logprob(r))

    def logprob_closure(self, r, params):
        """Log probability amplitude"""
        return self._rbm_psi_repr * self.log_wf(r, params).sum()

    def logprob(self, r):
        """Log probability amplitude"""
        params = self.params

        return self.logprob_closure(r, params)

    def grad_wf_closure(self, r, params):
        """
        This is the gradient of the logarithm of the wave function w.r.t. the coordinates
        """

        _expit = self.sigmoid(
            params.get("h_bias") + (r @ params.get("W_kernel")) * self._sigma2_factor
        )

        einsum_str = "ij,bj->bi"  # if r.ndim == 2 else "ij,j->i"  # TODO: CHANGE THIS
        gr = -(r - params.get("v_bias")) + self.backend.einsum(
            einsum_str, params.get("W_kernel"), _expit
        )
        gr *= self._sigma2 * self._factor
        return gr

    def grad_wf_closure_jax(self, r, params):
        """
        This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the coordinates
        """
        grad_wf_closure = jax.grad(self.wf, argnums=0)
        return vmap(grad_wf_closure, in_axes=(0, None))(r, params)

    def grad_wf(self, r):
        """
        grad of the wave function w.r.t. the coordinates
        """

        return self.grad_wf_closure(r, self.params)

    def laplacian_closure(self, r, params):
        _expit = self.sigmoid(
            params.get("h_bias") + (r @ params.get("W_kernel")) * self._sigma2_factor
        )  # r @ kernel is the r1 * W1 + r2 * W2 + ...
        _expos = self.sigmoid(
            -params.get("h_bias") - (r @ params.get("W_kernel")) * self._sigma2_factor
        )

        kernel2 = self.backend.square(params.get("W_kernel"))  # shape: (4, 4) if 2d

        # Element-wise multiplication, results in shape: (batch_size, 4)
        exp_prod = _expos * _expit

        # Use einsum for the batched matrix-vector product, kernel2 @ exp_prod for each item in the batch
        # This computes the dot product for each vector in exp_prod with kernel2, resulting in shape: (batch_size, 4)
        gr = -self._sigma2 + self._sigma4 * np.einsum("ij,bj->bi", kernel2, exp_prod)
        gr *= self._factor
        return gr.sum(axis=-1)  # sum over the coordinates

    def laplacian_closure_jax(self, r, params):
        """
        nabla^2 of the wave function w.r.t. the coordinates
        """

        def wrapped_wf(r_):
            return self.wf(r_, params)

        hessian_wf = vmap(jax.hessian(wrapped_wf))

        def trace_fn(x):
            return jnp.trace(x)

        laplacian = vmap(trace_fn)(hessian_wf(r))
        return laplacian

    def laplacian(self, r):
        params = self.params
        laplacian = self.laplacian_closure(r, params)

        return laplacian

    def grad_params_closure(self, r, params):
        _expit = self.sigmoid(
            params.get("h_bias") + (r @ params.get("W_kernel")) * self._sigma2_factor
        )

        grad_h_bias = self._factor * _expit

        # grad_kernel calculation needs to handle the outer product for each pair in the batch
        # r[:, None] adds an extra dimension making it (batch_size, previous_len, 1)
        # _expit[:, None] changes _expit to have shape (batch_size, 1, hidden_units),
        # enabling broadcasting for batch-wise outer product
        grad_kernel = (
            self._sigma2 * (r[:, :, None] @ _expit[:, None, :])
        ) * self._factor

        grad_v_bias = (r - params.get("v_bias")) * self._sigma2 * self._factor

        grads_dict = {
            "v_bias": grad_v_bias,
            "h_bias": grad_h_bias,
            "W_kernel": grad_kernel,
        }

        return grads_dict

    def grad_params_closure_jax(self, r, params):
        """
        This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the parameters
        """

        return vmap(jax.grad(self.wf, argnums=1), in_axes=(0, None))(r, params)

    def grad_params(self, r):
        """Gradients of the wave function w.r.t. the parameters"""

        return self.grad_params_closure(r, self.params)

    @property
    def sigma2(self):
        return self._sigma2

    @sigma2.setter
    def sigma2(self, value):
        self._sigma2 = value
        self._precompute()
