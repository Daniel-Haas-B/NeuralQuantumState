from abc import abstractmethod

import numpy as np
from scipy.special import expit


class BaseRBM:
    """Base class for creating a quantum system where the wave function is
    represented by a gaussian-binary restricted Boltzmann machine.

    The implementation assumes a logarithmic wave function.
    """

    def __init__(self, sigma2=1.0, factor=0.5, use_sr=False):
        self._sigma2 = sigma2
        self._factor = factor
        self._rbm_psi_repr = 2 * self._factor
        self._precompute()
        self.use_sr = use_sr

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

    def logprob(self, r, v_bias, h_bias, kernel):
        """Log probability amplitude"""
        psi2 = self._rbm_psi_repr * self._log_rbm(r, v_bias, h_bias, kernel).sum()
        return psi2

    def _grad_wf(self, r, v_bias, h_bias, kernel):
        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        gr = -(r - v_bias) + kernel @ _expit
        gr *= self._sigma2
        gr *= self._factor
        return gr

    def _laplace_wf(self, r, v_bias, h_bias, kernel):
        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        _expos = expit(-h_bias - (r @ kernel) * self._sigma2_factor)
        kernel2 = kernel * kernel
        exp_prod = _expos * _expit
        gr = -self._sigma2 + self._sigma4 * kernel2 @ exp_prod
        gr *= self._factor
        return gr

    def _local_kinetic_energy(self, r, v_bias, h_bias, kernel):
        """Evaluate the local kinetic energy"""
        _laplace = self._laplace_wf(r, v_bias, h_bias, kernel).sum()
        _grad = self._grad_wf(r, v_bias, h_bias, kernel)
        _grad2 = np.sum(_grad * _grad)
        return -0.5 * (_laplace + _grad2)

    def local_energy(self, r, v_bias, h_bias, kernel):
        """Local energy of the system"""
        ke = self._local_kinetic_energy(r, v_bias, h_bias, kernel)
        pe = self.potential(r)
        return ke + pe

    def drift_force(self, r, v_bias, h_bias, kernel):
        """Drift force at each particle's location"""
        F = 2 * self._grad_wf(r, v_bias, h_bias, kernel)
        return F

    def grads(self, r, v_bias, h_bias, kernel):
        """Gradients of the wave function w.r.t. the parameters"""
        grad_h_bias = self._grad_h_bias(r, v_bias, h_bias, kernel, self.use_sr)
        grad_v_bias = self._grad_v_bias(r, v_bias, h_bias, kernel, self.use_sr)
        grad_kernel = self._grad_kernel(r, v_bias, h_bias, kernel, self.use_sr)
        return grad_v_bias, grad_h_bias, grad_kernel

    def _grad_v_bias(self, r, v_bias, h_bias, kernel, use_sr=False):
        """Gradient of wave function w.r.t. visible bias"""
        gr = (r - v_bias) * self._sigma2
        gr *= self._factor
        return gr  # .sum()

    def _grad_h_bias(self, r, v_bias, h_bias, kernel, use_sr=False):
        """Gradient of wave function w.r.t. hidden bias"""
        gr = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        gr *= self._factor
        return gr  # .sum()

    def _grad_kernel(self, r, v_bias, h_bias, kernel, use_sr=False):
        """Gradient of wave function w.r.t. weight matrix"""
        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        gr = self._sigma2 * r[:, np.newaxis] @ _expit[:, np.newaxis].T
        gr *= self._factor
        # print("gr.shape", gr.shape)
        # print("gr type", type(gr))

        if use_sr:
            sr = self._compute_sr_matrix(r, v_bias, h_bias, kernel)
            # print("sr.shape", sr.shape)
            # print("sr type", type(sr))
            gr = gr @ np.linalg.pinv(sr + 1e-2 * np.eye(sr.shape[0]))
        return gr

    def _compute_sr_matrix(self, r, v_bias, h_bias, kernel):
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

        """

        # 1. Compute the gradient of log(psi) w.r.t. kernel for each sample r
        grad_log_psi = self._grad_kernel(r, v_bias, h_bias, kernel)
        # print("grad_log_psi.shape", grad_log_psi.shape)
        # 2. Compute the outer product of gradient with itself for each sample r
        outer_product = np.einsum("ni,nj->nij", grad_log_psi, grad_log_psi)
        # grad_log_psi.shape (2, 4)

        # 3. Compute the mean of the outer product over all samples r
        mean_outer_product = outer_product.mean(axis=0)

        # 4. Compute the mean gradient over all samples r
        mean_grad = grad_log_psi.mean(axis=0)

        # 5. Compute the outer product of the mean gradient with itself
        mean_grad_outer_product = np.outer(mean_grad, mean_grad)

        # 6. Subtract the two results to get the S matrix
        sr_mat = mean_outer_product - mean_grad_outer_product
        # print("sr_mat.shape", sr_mat.shape)
        # exit()
        return sr_mat

        return sr_mat

    @property
    def sigma2(self):
        return self._sigma2

    @sigma2.setter
    def sigma2(self, value):
        self._sigma2 = value
        self._precompute()
