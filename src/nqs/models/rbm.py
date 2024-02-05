import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from nqs.utils import Parameter
from nqs.utils import State
from scipy.special import expit


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class RBM:
    def __init__(
        self,
        nparticles,
        dim,
        nhidden=1,
        factor=1.0,  # TODO: verify this
        sigma2=1.0,
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
        backend="numpy",
    ):
        """
        Initializes the RBM Neural Network Quantum State.

        Args:
        - nparticles (int): Number of particles.
        - dim (int): Dimensionality.
        ...
        """
        self._configure_backend(backend)
        self._initialize_vars(nparticles, dim, nhidden, factor, sigma2)

        if logger:
            self.logger = logger
        else:
            import logging

            self.logger = logging.getLogger(__name__)

        self.log = log
        self.rng = rng if rng else np.random.default_rng()
        r = rng.standard_normal(size=self._nvisible)

        self._initialize_bias_and_kernel(rng)

        logp = self.logprob(r)
        self.state = State(r, logp, 0, 0)

        if self.log:
            neuron_str = "neurons" if self._nhidden > 1 else "neuron"
            msg = (
                f"Neural Network Quantum State initialized as RBM with "
                f"{self._nhidden} hidden {neuron_str}"
            )
            self.logger.info(msg)

    def _initialize_bias_and_kernel(self, rng):
        v_bias = rng.standard_normal(size=self._nvisible) * 0.01
        h_bias = rng.standard_normal(size=self._nhidden) * 0.01
        kernel = rng.standard_normal(size=(self._nvisible, self._nhidden))
        kernel *= np.sqrt(1 / self._nvisible)
        self.params = Parameter()
        self.params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])

    def _initialize_vars(self, nparticles, dim, nhidden, factor, sigma2):
        self._sigma2 = sigma2
        self._factor = factor
        self._rbm_psi_repr = 2 * self._factor
        self._N = nparticles
        self._dim = dim
        self._nvisible = self._N * self._dim
        self._nhidden = nhidden
        self._precompute()

    def _configure_backend(self, backend):
        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
            self.sigmoid = expit

        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg
            self.sigmoid = expit  # jax.nn.sigmoid
            self.grad_wf_closure = self.grad_wf_closure_jax
            self.grads_closure = self.grads_closure_jax
            self.laplacian_closure = self.laplacian_closure_jax
            # self._convert_constants_to_jnp()
            self._jit_functions()
        else:
            raise ValueError(f"Invalid backend: {backend}")

    def _jit_functions(self):
        functions_to_jit = [
            "_log_wf",
            "wf",
            "logprob_closure",
            "grad_wf_closure",
            "grads_closure",
            "laplacian_closure",
            "_precompute",
            "_softplus",
        ]
        for func_name in functions_to_jit:
            setattr(self, func_name, jax.jit(getattr(self, func_name)))
        return self  # for chaining

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

    def _log_wf(self, r, v_bias, h_bias, kernel):
        """Logarithmic gaussian-binary RBM"""

        # visible layer
        x_v = self.la.norm(r - v_bias)
        x_v *= -x_v * self._sigma2_factor2

        # hidden layer
        x_h = self._softplus(h_bias + (r.T @ kernel) * self._sigma2_factor)
        x_h = self.backend.sum(x_h, axis=-1)

        return x_v + x_h

    def wf(self, r, v_bias, h_bias, kernel):
        """Evaluate the wave function"""
        return (
            self._factor * self._log_wf(r, v_bias, h_bias, kernel).sum()
        )  # we sum because we are in the log domain

    def pdf(self, r):
        """
        Probability amplitude
        """
        return self.backend.exp(self.logprob(r))

    def logprob_closure(self, r, v_bias, h_bias, kernel):
        """Log probability amplitude"""

        return self._rbm_psi_repr * self._log_wf(r, v_bias, h_bias, kernel).sum()

    def logprob(self, r):
        """Log probability amplitude"""
        v_bias, h_bias, kernel = (
            self.params.get("v_bias"),
            self.params.get("h_bias"),
            self.params.get("kernel"),
        )

        return self.logprob_closure(r, v_bias, h_bias, kernel)

    def grad_wf_closure(self, r, v_bias, h_bias, kernel):
        """
        This is the gradient of the logarithm of the wave function w.r.t. the coordinates
        """
        print("hbias shape", h_bias.shape)
        print("kernel shape", kernel.shape)
        print("r shape", r.shape)
        _expit = self.sigmoid(h_bias + (r @ kernel) * self._sigma2_factor)
        gr = -(r - v_bias) + kernel @ _expit
        gr *= self._sigma2 * self._factor
        return gr

    # @partial(jax.jit, static_argnums=(0,))
    def grad_wf_closure_jax(self, r, v_bias, h_bias, kernel):
        """
        This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the coordinates
        """

        grad_wf = jax.grad(self.wf, argnums=0)
        # grad_wf(r, v_bias, h_bias, kernel)

        grad_wf = vmap(
            grad_wf,
            in_axes=(0, None, None, None),
        )(r, v_bias, h_bias, kernel)

        return grad_wf

    def grad_wf(self, r):
        """
        grad of the wave function w.r.t. the coordinates
        """
        v_bias, h_bias, kernel = (
            self.params.get("v_bias"),
            self.params.get("h_bias"),
            self.params.get("kernel"),
        )
        # v_bias, h_bias, kernel = self.params.get(["v_bias", "h_bias", "kernel"])
        return self.grad_wf_closure(r, v_bias, h_bias, kernel)

    def laplacian_closure(self, r, v_bias, h_bias, kernel):
        _expit = self.sigmoid(h_bias + (r @ kernel) * self._sigma2_factor)
        _expos = self.sigmoid(-h_bias - (r @ kernel) * self._sigma2_factor)
        kernel2 = self.backend.square(kernel)
        exp_prod = _expos * _expit
        gr = -self._sigma2 + self._sigma4 * kernel2 @ exp_prod
        gr *= self._factor
        return gr

    # @partial(jax.jit, static_argnums=(0,))
    def laplacian_closure_jax(self, r, v_bias, h_bias, kernel):
        """
        nabla^2 of the wave function w.r.t. the coordinates
        """

        def wrapped_wf(r_):
            return self.wf(r_, v_bias, h_bias, kernel)

        hessian_wf = vmap(jax.hessian(wrapped_wf))

        # If you want the Laplacian, you'd sum the diagonal of the Hessian.
        # This assumes r is a vector and you want the Laplacian w.r.t. each element.
        def trace_fn(x):
            return jnp.trace(x)

        laplacian = vmap(trace_fn)(hessian_wf(r))

        return laplacian

    def laplacian(self, r):
        v_bias, h_bias, kernel = (
            self.params.get("v_bias"),
            self.params.get("h_bias"),
            self.params.get("kernel"),
        )
        laplacian = self.laplacian_closure(r, v_bias, h_bias, kernel)

        return laplacian

    def grads_closure(self, r, v_bias, h_bias, kernel):
        _expit = self.sigmoid(h_bias + (r @ kernel) * self._sigma2_factor)
        grad_h_bias = self._factor * _expit
        grad_kernel = (
            self._sigma2
            * r[:, self.backend.newaxis]
            @ _expit[:, self.backend.newaxis].T
        ) * self._factor
        grad_v_bias = (r - v_bias) * self._sigma2 * self._factor
        return grad_v_bias, grad_h_bias, grad_kernel

    # @partial(jax.jit, static_argnums=(0,))
    def grads_closure_jax(self, r, v_bias, h_bias, kernel):
        """
        This is the autograd version of the gradient of the logarithm of the wave function w.r.t. the parameters
        """

        batch_size = r.shape[0] if np.ndim(r) > 1 else 1

        def scalar_wf(r_, v_bias, h_bias, kernel, i):
            wf_values = self.wf(r_, v_bias, h_bias, kernel)
            return wf_values[i]

        grads = (
            vmap(
                lambda i: jax.grad(scalar_wf, argnums=1)(r, v_bias, h_bias, kernel, i)
            )(np.arange(batch_size)),
            vmap(
                lambda i: jax.grad(scalar_wf, argnums=2)(r, v_bias, h_bias, kernel, i)
            )(np.arange(batch_size)),
            vmap(
                lambda i: jax.grad(scalar_wf, argnums=3)(r, v_bias, h_bias, kernel, i)
            )(np.arange(batch_size)),
        )

        return grads

    def grads(self, r):
        """Gradients of the wave function w.r.t. the parameters"""
        v_bias, h_bias, kernel = (
            self.params.get("v_bias"),
            self.params.get("h_bias"),
            self.params.get("kernel"),
        )

        grad_v_bias, grad_h_bias, grad_kernel = self.grads_closure(
            r, v_bias, h_bias, kernel
        )
        grads_dict = {
            "v_bias": grad_v_bias,
            "h_bias": grad_h_bias,
            "kernel": grad_kernel,
        }
        return grads_dict  # grad_v_bias, grad_h_bias, grad_kernel

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

            # if self.backend.ndim(grad_value[0]) == 2:
            if key == "kernel":
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

    # def tune(
    #     self,
    #     tune_iter=20_000,
    #     tune_interval=500,
    #     early_stop=False,  # set to True later
    #     rtol=1e-05,
    #     atol=1e-08,
    #     seed=None,
    #     mcmc_alg=None,
    # ):
    #     """
    #     !! BROKEN NOW due to self.scale
    #     Tune proposal scale so that the acceptance rate is around 0.5.
    #     """

    #     state = self.state
    #     v_bias, h_bias, kernel = self.wf.params.get(["v_bias", "h_bias", "kernel"])

    #     scale = self.scale

    #     if mcmc_alg is not None:
    #         self._sampler = Sampler(self.mcmc_alg, self.rbm, self.rng, self._log)

    #     # Used to throw warnings if tuned alg mismatch chosen alg
    #     # in other procedures
    #     self._tuned_mcmc_alg = self.mcmc_alg

    #     # Config
    #     # did_early_stop = False
    #     seed_seq = generate_seed_sequence(seed, 1)[0]

    #     # Reset n_accepted
    #     state = State(state.positions, state.logp, 0, state.delta)

    #     if self._log:
    #         t_range = tqdm(
    #             range(tune_iter),
    #             desc="[Tuning progress]",
    #             position=0,
    #             leave=True,
    #             colour="green",
    #         )
    #     else:
    #         t_range = range(tune_iter)

    #     steps_before_tune = tune_interval

    #     for i in t_range:
    #         state = self._sampler.step(state, v_bias, h_bias, kernel, seed_seq)
    #         steps_before_tune -= 1

    #         if steps_before_tune == 0:
    #             # Tune proposal scale
    #             old_scale = scale
    #             accept_rate = state.n_accepted / tune_interval
    #             scale = self._sampler.tune_scale(old_scale, accept_rate)

    #             # Reset
    #             steps_before_tune = tune_interval
    #             state = State(state.positions, state.logp, 0, state.delta)

    #     # Update shared values
    #     self.state = state
    #     self.wf.params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])
    #     self.scale = scale
    #     self._is_tuned_ = True
