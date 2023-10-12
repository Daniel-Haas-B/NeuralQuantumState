from abc import abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

# from functools import partial

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class BaseRBM:
    """
    Base class for creating a quantum system where the wave function is
    represented by a gaussian-binary restricted Boltzmann machine.

    The implementation assumes a logarithmic wave function.
    """

    def __init__(self, sigma2=1.0, factor=0.5, backend="numpy"):
        self._sigma2 = sigma2
        self._factor = factor
        self._rbm_psi_repr = 2 * self._factor
        self._precompute()
        self.params = None

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

        else:
            raise ValueError("Invalid backend:", backend)

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
        """Probability amplitude"""
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
