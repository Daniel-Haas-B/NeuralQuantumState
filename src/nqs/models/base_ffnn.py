# from abc import abstractmethod
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


class FFNN:
    """
    WIP
    Base class for creating a quantum system where the wave function is
    represented by a FFNN.
    There will be no analytical expressions for the derivatives, so we
    should use numerical differentiation instead.
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

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        """Potential energy function
        This assumes non interacting particles!
        #TODO: add interaction
        """
        return 0.5 * jnp.sum(r * r)

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

    def laplacian_wf(self, r, params):
        """Laplacian of the wave function"""

        return

    def _local_kinetic_energy(self, r, params):
        """Evaluate the local kinetic energy"""
        _laplace = self.laplacian_wf(r, params).sum()
        _grad = self._grad_wf(r, params)
        _grad2 = np.sum(_grad * _grad)
        return -0.5 * (_laplace + _grad2)

    @partial(jax.jit, static_argnums=(0,))
    def local_energy(self, r, params):
        """Local energy of the system"""

        def ke_closure(r):
            return self._local_kinetic_energy(r, params)

        ke = jnp.sum(ke_closure(r))
        pe = self.potential(r)

        return ke + pe

    # def drift_force(self, r, params):
    #     """Drift force at each particle's location"""
    #     F = 2 * self._grad_wf(r, params)
    #     return F

    def grads(self, r, params):
        """Gradients of the wave function w.r.t. the parameters"""

        return
