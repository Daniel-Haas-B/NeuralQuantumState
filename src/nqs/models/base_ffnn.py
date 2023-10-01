from abc import abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# from jax import grad
# from jax import jit
# from jax import lax
# from jax import vmap
# from scipy.special import expit

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class BaseFFNN:
    """Base class for creating a quantum system where the wave function is
    represented by a FFNN.

    The implementation assumes a logarithmic wave function.
    """

    def __init__(self, sigma2=1.0, factor=0.5):
        self._sigma2 = sigma2
        self._factor = factor
        self._ffnn_psi_repr = 2 * self._factor
        # self._precompute()

    # def _precompute(self):
    #    self._sigma4 = self._sigma2 * self._sigma2
    #    self._sigma2_factor = 1.0 / self._sigma2
    #    self._sigma2_factor2 = 0.5 / self._sigma2

    @partial(jax.jit, static_argnums=(0,))
    def _softplus(self, x):
        """Softplus activation function.

        Computes the element-wise function
                softplus(x) = log(1 + e^x)
        """
        return jnp.logaddexp(x, 0)

    def _log_ffnn(self, r, params):
        """Logarithmic FFNN
        r: input
        params: class containing the weights and biases of the network in the form
        params.weights = [w1, w2, ..., wn] where wi is the weight matrix of layer i
        params.biases = [b1, b2, ..., bn] where bi is the bias vector of layer i
        """

        return

    def wf(self, r, params):
        """Evaluate the wave function"""
        return self._factor * self._log_ffnn(r, params).sum()

    @abstractmethod
    def potential(self):
        """Potential energy function.

        To be overwritten by subclass.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, r, params):
        """Probability amplitude"""
        return jnp.exp(self.logprob(r, params))

    def logprob(self, r, params):
        """Log probability amplitude"""
        psi2 = self._ffnn_psi_repr * self._log_ffnn(r, params).sum()
        return psi2

    def _grad_wf(self, r, params):
        """Gradient of the wave function wrt position"""
        return

    def _laplace_wf(self, r, params):
        """Laplacian of the wave function"""

        return

    def _local_kinetic_energy(self, r, params):
        """Evaluate the local kinetic energy"""
        _laplace = self._laplace_wf(r, params).sum()
        _grad = self._grad_wf(r, params)
        _grad2 = np.sum(_grad * _grad)
        return -0.5 * (_laplace + _grad2)

    def local_energy(self, r, params):
        """Local energy of the system"""
        ke = self._local_kinetic_energy(r, params)
        pe = self.potential(r)
        return ke + pe

    def drift_force(self, r, params):
        """Drift force at each particle's location"""
        F = 2 * self._grad_wf(r, params)
        return F

    def grads(self, r, params):
        """Gradients of the wave function w.r.t. the parameters"""

        return

    def compute_sr_matrix(self, expval_grads, grads, shift=1e-4):
        """
        WIP: for now this does not involve the averages because r will be a single sample
        Compute the matrix for the stochastic reconfiguration algorithm
            for now we do it only for the kernel
            The expression here is for kernel element W_ij:
                S_ij,kl = < (d/dW_ij log(psi)) (d/dW_kl log(psi)) > - < d/dW_ij log(psi) > < d/dW_kl log(psi) >
          1. Compute the gradient ∂_W log(ψ) using the _grad_kernel function.
            2. Compute the outer product of the gradient with itself: ∂_W log(ψ) ⊗ ∂_W log(ψ)
            3. Compute the expectation value of the outer product over all the samples
            4. Compute the expectation value of the gradient ∂_W log(ψ) over all the samples
            5. Compute the outer product of the expectation value of the gradient with itself: <∂_W log(ψ)> ⊗ <∂_W log(ψ)>

            OBS: < d/dW_ij log(psi) > is already done inside train of the RBM class but we need still the < (d/dW_ij log(psi)) (d/dW_kl log(psi)) >
        """

        grads_outer = np.einsum("nij,nkl->nijkl", grads, grads)
        expval_outer_grad_kernel = np.mean(grads_outer, axis=0)

        outer_expval_grad = np.array(np.outer(expval_grads, expval_grads))
        sr_mat = (
            expval_outer_grad_kernel.reshape(outer_expval_grad.shape)
            - outer_expval_grad
        )
        sr_mat = sr_mat + shift * np.eye(sr_mat.shape[0])
        # print("sr_mat.shape", sr_mat.shape)
        return sr_mat

    @property
    def sigma2(self):
        return self._sigma2

    @sigma2.setter
    def sigma2(self, value):
        self._sigma2 = value
        self._precompute()
