from abc import abstractmethod

import numpy as np
from scipy.special import expit


class BaseRBM:
    """Base class for creating a quantum system where the wave function is
    represented by a gaussian-binary restricted Boltzmann machine.

    I think this guy should have all params
    The implementation assumes a logarithmic wave function.
    """

    def __init__(self, sigma2=1.0, factor=0.5):
        self._sigma2 = sigma2
        self._factor = factor
        self._rbm_psi_repr = 2 * self._factor
        self._precompute()
        self.params = None

    def _precompute(self):
        self._sigma4 = self._sigma2 * self._sigma2
        self._sigma2_factor = 1.0 / self._sigma2
        self._sigma2_factor2 = 0.5 / self._sigma2

    def _softplus(self, x):
        """Softplus activation function.

        Computes the element-wise function
                softplus(x) = log(1 + e^x)
        """
        return np.logaddexp(x, 0)

    def _log_rbm(self, r, v_bias, h_bias, kernel):
        """Logarithmic gaussian-binary RBM"""

        # visible layer
        x_v = np.linalg.norm(r - v_bias)
        x_v *= -x_v * self._sigma2_factor2

        # hidden layer
        x_h = self._softplus(h_bias + (r.T @ kernel) * self._sigma2_factor)
        x_h = np.sum(x_h, axis=-1)

        return x_v + x_h

    def wf(self, r, v_bias, h_bias, kernel):
        """Evaluate the wave function"""
        return self._factor * self._log_rbm(r, v_bias, h_bias, kernel).sum()

    @abstractmethod
    def potential(self):
        """Potential energy function.

        To be overwritten by subclass.
        """
        raise NotImplementedError

    def pdf(self, r, v_bias, h_bias, kernel):
        """Probability amplitude"""
        return np.exp(self.logprob(r, v_bias, h_bias, kernel))

    def logprob(self, r):
        """Log probability amplitude"""
        v_bias, h_bias, kernel = self.params.get(["v_bias", "h_bias", "kernel"])
        psi2 = self._rbm_psi_repr * self._log_rbm(r, v_bias, h_bias, kernel).sum()
        return psi2

    def grad_wf(self, r):
        """
        #TODO: maybe we dont need even to pass the params
        """
        v_bias, h_bias, kernel = self.params.get(["v_bias", "h_bias", "kernel"])
        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        gr = -(r - v_bias) + kernel @ _expit
        gr *= self._sigma2
        gr *= self._factor
        return gr

    def laplacian_wf(self, r):
        v_bias, h_bias, kernel = self.params.get(["v_bias", "h_bias", "kernel"])

        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        _expos = expit(-h_bias - (r @ kernel) * self._sigma2_factor)
        kernel2 = np.square(kernel)
        exp_prod = _expos * _expit
        gr = -self._sigma2 + self._sigma4 * kernel2 @ exp_prod
        gr *= self._factor
        return gr

    def grads(self, r):
        """Gradients of the wave function w.r.t. the parameters"""
        v_bias, h_bias, kernel = self.params.get(["v_bias", "h_bias", "kernel"])
        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        grad_h_bias = self._grad_h_bias(r, _expit)
        grad_kernel = self._grad_kernel(r, _expit)
        grad_v_bias = self._grad_v_bias(r, v_bias, h_bias, kernel)
        return grad_v_bias, grad_h_bias, grad_kernel

    def _grad_v_bias(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. visible bias"""
        gr = (r - v_bias) * self._sigma2
        gr *= self._factor
        return gr  # .sum()

    def _grad_h_bias(self, r, _expit):
        """Gradient of wave function w.r.t. hidden bias"""
        return self._factor * _expit  # .sum()

    def _grad_kernel(self, r, _expit):
        """Gradient of wave function w.r.t. weight matrix"""
        gr = self._sigma2 * r[:, np.newaxis] @ _expit[:, np.newaxis].T
        gr *= self._factor

        return gr

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
            if grad_value[0].ndim == 2:
                grads_outer = np.einsum("nij,nkl->nijkl", grad_value, grad_value)
            elif grad_value[0].ndim == 1:
                grads_outer = np.einsum("ni,nj->nij", grad_value, grad_value)

            expval_outer_grad = np.mean(grads_outer, axis=0)
            outer_expval_grad = np.outer(expval_grads[key], expval_grads[key])

            sr_mat = (
                expval_outer_grad.reshape(outer_expval_grad.shape) - outer_expval_grad
            )

            sr_matrices[key] = sr_mat + shift * np.eye(sr_mat.shape[0])

        return sr_matrices

    @property
    def sigma2(self):
        return self._sigma2

    @sigma2.setter
    def sigma2(self, value):
        self._sigma2 = value
        self._precompute()
