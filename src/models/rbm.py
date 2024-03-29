import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from src.models.base_wf import WaveFunction
from src.state.utils import Parameter
from src.state.utils import State

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


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
            neuron_str = "neurons" if self._nhidden > 1 else "neuron"
            msg = (
                f"Neural Network Quantum State initialized as RBM with "
                f"{self._nhidden} hidden {neuron_str}"
            )
            self.logger.info(msg)

    def __call__(self, r):
        return self.wf(
            r,
            self.params.get("v_bias"),
            self.params.get("h_bias"),
            self.params.get("kernel"),
        )

    def _initialize_bias_and_kernel(self, rng):
        v_bias = rng.standard_normal(size=self._nvisible) * 0.01
        h_bias = rng.standard_normal(size=self._nhidden) * 0.01
        kernel = rng.standard_normal(size=(self._nvisible, self._nhidden))
        kernel *= np.sqrt(1 / self._nvisible)
        self.params = Parameter()
        self.params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])
        if self.jastrow:
            input_j_size = self._N * (self._N - 1) // 2
            limit = np.sqrt(2 / (input_j_size))
            self.params.set(
                "WJ", np.array(rng.uniform(-limit, limit, (self._N, self._N)))
            )
        if self.pade_jastrow:
            assert not self.jastrow, "Pade Jastrow requires Jastrow to be false"
            self.params.set("CPJ", np.array(rng.uniform(-limit, limit, 1)))

    def _initialize_vars(self, nparticles, dim, nhidden, factor, sigma2):
        self._sigma2 = sigma2
        self._factor = factor
        self._rbm_psi_repr = 2 * self._factor
        self._N = nparticles
        self._dim = dim
        self._nvisible = self._N * self._dim
        self._nhidden = nhidden
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
            params.get("h_bias") + (r @ params.get("kernel")) * self._sigma2_factor
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
            params.get("h_bias") + (r @ params.get("kernel")) * self._sigma2_factor
        )

        einsum_str = "ij,bj->bi"  # if r.ndim == 2 else "ij,j->i"  # TODO: CHANGE THIS
        gr = -(r - params.get("v_bias")) + self.backend.einsum(
            einsum_str, params.get("kernel"), _expit
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
        params = self.params
        grads = self.grad_wf_closure(r, params)

        return grads

    def laplacian_closure(self, r, params):
        _expit = self.sigmoid(
            params.get("h_bias") + (r @ params.get("kernel")) * self._sigma2_factor
        )  # r @ kernel is the r1 * W1 + r2 * W2 + ...
        _expos = self.sigmoid(
            -params.get("h_bias") - (r @ params.get("kernel")) * self._sigma2_factor
        )

        kernel2 = self.backend.square(params.get("kernel"))  # shape: (4, 4) if 2d

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

    def grads_closure(self, r, params):
        _expit = self.sigmoid(
            params.get("h_bias") + (r @ params.get("kernel")) * self._sigma2_factor
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
            "kernel": grad_kernel,
        }

        return grads_dict

    def grads_closure_jax(self, r, params):
        """
        This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the parameters
        """
        grads = vmap(jax.grad(self.wf, argnums=1), in_axes=(0, None))(r, params)

        return grads

    def grads(self, r):
        """Gradients of the wave function w.r.t. the parameters"""
        params = self.params

        grads_dict = self.grads_closure(r, params)
        return grads_dict  # grad_v_bias, grad_h_bias, grad_kernel

    # @partial(jax.jit, static_argnums=(0,)) only if backend is jax
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
            grad_value = self.backend.array(grad_value)

            # if self.backend.ndim(grad_value[0]) == 2:
            if key == "kernel" or key == "WJ":
                grads_outer = self.backend.einsum(
                    "nij,nkl->nijkl", grad_value, grad_value
                )
            # elif self.backend.ndim(grad_value[0]) == 1:
            else:
                grads_outer = self.backend.einsum("ni,nj->nij", grad_value, grad_value)

            expval_outer_grad = self.backend.mean(grads_outer, axis=0)
            outer_expval_grad = self.backend.outer(expval_grads[key], expval_grads[key])

            sr_mat = (
                expval_outer_grad.reshape(outer_expval_grad.shape) - outer_expval_grad
            )
            sr_matrices[key] = sr_mat + shift * self.backend.eye(sr_mat.shape[0])

        return sr_matrices

    @property
    def sigma2(self):
        return self._sigma2

    @sigma2.setter
    def sigma2(self, value):
        self._sigma2 = value
        self._precompute()
