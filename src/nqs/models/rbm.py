from abc import abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
from nqs.utils import Parameter
from nqs.utils import State
from scipy.special import expit

# from functools import partial

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class RBM:
    def __init__(
        self,
        nparticles,
        dim,
        nhidden=1,
        factor=1.0,  # not sure about this value
        sigma2=1.0,
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
        backend="numpy",
    ):
        """
        RBM Neural Network Quantum State
        the wave function is represented by a gaussian-binary restricted Boltzmann machine.

        The implementation assumes a logarithmic wave function.
        """
        self._sigma2 = sigma2
        self._factor = factor
        self._rbm_psi_repr = 2 * self._factor
        self._N = nparticles
        self._dim = dim
        self._nvisible = self._N * self._dim
        self._nhidden = nhidden
        self.logger = logger
        self._precompute()

        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
            self.sigmoid = expit
        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg
            self.sigmoid = jax.nn.sigmoid
            # convert constants to jnp 64
            self._factor = jnp.float64(self._factor)
            self._rbm_psi_repr = jnp.float64(self._rbm_psi_repr)
            self._sigma2 = jnp.float64(self._sigma2)
            self._sigma4 = jnp.float64(self._sigma4)
            self._sigma2_factor = jnp.float64(self._sigma2_factor)
            # jit functions
            self._log_wf = jax.jit(self._log_wf)
            self.wf = jax.jit(self.wf)
            self.logprob_closure = jax.jit(self.logprob_closure)
            self.grad_closure = jax.jit(self.grad_closure)
            self._precompute = jax.jit(self._precompute)
            self._softplus = jax.jit(self._softplus)
        else:
            raise ValueError("Invalid backend:", backend)

        r = rng.standard_normal(size=self._nvisible)

        # Initialize visible bias
        v_bias = rng.standard_normal(size=self._nvisible) * 0.01
        h_bias = rng.standard_normal(size=self._nhidden) * 0.01
        kernel = rng.standard_normal(size=(self._nvisible, self._nhidden))
        kernel *= np.sqrt(1 / self._nvisible)

        self.params = Parameter()
        self.params.set(["v_bias", "h_bias", "kernel"], [v_bias, h_bias, kernel])

        self.log = log
        logp = self.logprob(r)
        self.state = State(r, logp, 0, 0)

        if self.log:
            neuron_str = "neurons" if self._nhidden > 1 else "neuron"
            msg = (
                f"Neural Network Quantum State initialized as RBM with "
                f"{self._nhidden} hidden {neuron_str}"
            )
            self.logger.info(msg)

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
        return self._factor * self._log_wf(r, v_bias, h_bias, kernel).sum()

    @abstractmethod
    def potential(self):
        """Potential energy function.
        To be overwritten by subclass.
        """
        raise NotImplementedError

    def pdf(self, r, v_bias, h_bias, kernel):
        """
        Probability amplitude
        """
        return self.backend.exp(self.logprob(r, v_bias, h_bias, kernel))

    def logprob_closure(self, r, v_bias, h_bias, kernel):
        """Log probability amplitude"""
        return self._rbm_psi_repr * self._log_wf(r, v_bias, h_bias, kernel).sum()

    def logprob(self, r):
        """Log probability amplitude"""
        v_bias, h_bias, kernel = self.params.get(["v_bias", "h_bias", "kernel"])
        return self.logprob_closure(r, v_bias, h_bias, kernel)

    def grad_closure(self, r, v_bias, h_bias, kernel):
        _expit = self.sigmoid(h_bias + (r @ kernel) * self._sigma2_factor)
        gr = -(r - v_bias) + kernel @ _expit
        gr *= self._sigma2 * self._factor
        return gr

    def jax_grad_closure(self, r, v_bias, h_bias, kernel):
        """
        Here we will use jaxgrad to compute the gradient
        """

    def grad_wf(self, r):
        """
        grad of the wave function w.r.t. the coordinates
        """
        v_bias, h_bias, kernel = self.params.get(["v_bias", "h_bias", "kernel"])
        return self.grad_closure(r, v_bias, h_bias, kernel)

    def laplacian_closure(self, r, v_bias, h_bias, kernel):
        _expit = self.sigmoid(h_bias + (r @ kernel) * self._sigma2_factor)
        _expos = self.sigmoid(-h_bias - (r @ kernel) * self._sigma2_factor)
        kernel2 = self.backend.square(kernel)
        exp_prod = _expos * _expit
        gr = -self._sigma2 + self._sigma4 * kernel2 @ exp_prod
        gr *= self._factor
        return gr

    def laplacian(self, r):
        v_bias, h_bias, kernel = self.params.get(["v_bias", "h_bias", "kernel"])
        return self.laplacian_closure(r, v_bias, h_bias, kernel)

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

    def grads(self, r):
        """Gradients of the wave function w.r.t. the parameters"""
        v_bias, h_bias, kernel = self.params.get(["v_bias", "h_bias", "kernel"])
        return self.grads_closure(r, v_bias, h_bias, kernel)

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
