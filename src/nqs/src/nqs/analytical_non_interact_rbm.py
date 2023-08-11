#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import expit


class AniRBM:
    """Base class for creating a quantum system where the wave function is
    represented by a gaussian-binary restricted Boltzmann machine.

    The implementation assumes a logarithmic wave function.
    """

    def __init__(self, sigma2=1.0):
        self._sigma2 = sigma2
        self._sigma4 = sigma2 * sigma2
        self._sigma2_factor = 1 / self._sigma2
        self._sigma2_factor2 = 0.5 * 1 / self._sigma2
        self._sigma4_factor = 0.5 * 1 / self._sigma4

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
        # x_v = r-v_bias
        x_v *= -x_v * self._sigma2_factor2
        # print("Shape x_v: ", x_v.shape)

        # hidden layer
        x_h = self._softplus(h_bias + (r.T @ kernel) * self._sigma2_factor)
        x_h = np.sum(x_h, axis=-1)
        # x_h_x_v = x_v + x_h
        # print("SHape x_h + x_v: ", x_h_x_v.shape)
        return x_v + x_h

    def wf(self, r, v_bias, h_bias, kernel):
        """Evaluate the wave function"""
        return 0.5 * self._log_rbm(r, v_bias, h_bias, kernel)  # .sum()

    def potential(self, r):
        """Potential energy function"""
        return 0.5 * np.sum(r * r)

    def pdf(self, r, v_bias, h_bias, kernel):
        """Probability amplitude"""
        return np.exp(self.logprob(r, v_bias, h_bias, kernel))

    def logprob(self, r, v_bias, h_bias, kernel):
        """Log probability amplitude"""
        return self._log_rbm(r, v_bias, h_bias, kernel).sum()

    def _grad_wf(self, r, v_bias, h_bias, kernel):
        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        gr = -(r - v_bias) + kernel @ _expit
        gr *= self._sigma2_factor2
        return gr

    def _laplace_wf(self, r, v_bias, h_bias, kernel):
        """
        Trial 1
        """
        _expit_pos = expit(-h_bias - (r @ kernel) * self._sigma2_factor)
        _exp = np.exp(h_bias + (r @ kernel) * self._sigma2_factor)
        kernel2 = kernel * kernel
        _expit_pos2 = _expit_pos * _expit_pos
        _exp_prod = _exp[:, np.newaxis] @ _expit_pos2[:, np.newaxis].T
        gr = -self._sigma2_factor2 + self._sigma4_factor * kernel2 @ _exp_prod
        return gr

    def _laplace_wf2(self, r, v_bias, h_bias, kernel):
        """
        Trial 2
        """
        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        _expos = expit(-h_bias - (r @ kernel) * self._sigma2_factor)
        kernel2 = kernel * kernel
        # exp_prod = _expos[:, np.newaxis] @ _expit[:, np.newaxis].T
        exp_prod = _expos * _expit
        gr = -self._sigma2_factor2 + self._sigma4_factor * kernel2 @ exp_prod
        # print(kernel2 @ exp_prod)
        return gr

    def _local_kinetic_energy(self, r, v_bias, h_bias, kernel):
        """Evaluate the local kinetic energy"""

        # where to sum?

        # _laplace = self._laplace_wf2(r, v_bias, h_bias, kernel)
        _laplace = self._laplace_wf2(r, v_bias, h_bias, kernel).sum()
        _grad = self._grad_wf(r, v_bias, h_bias, kernel)
        _grad2 = np.sum(_grad * _grad)
        # print("Laplace: ", _laplace)
        # print("grad2 : ", _grad2)
        # print("Laplace + grad2 : ", _laplace + _grad2)
        # _grad2 = _grad * _grad

        return -0.5 * (_laplace + _grad2)
        # return -0.5 * (_laplace + _grad2)

    def local_energy(self, r, v_bias, h_bias, kernel):
        """Local energy of the system"""
        ke = self._local_kinetic_energy(r, v_bias, h_bias, kernel)
        pe = self.potential(r)
        return ke + pe

    def drift_force(self, r, v_bias, h_bias, kernel):
        """Drift force at each particle's location"""
        F = 2 * self._grad_wf(r, v_bias, h_bias, kernel)
        return F

    def grad_v_bias(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. visible bias"""
        gr = (r - v_bias) * self._sigma2_factor2
        return gr  # .sum()

    def grad_h_bias(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. hidden bias"""
        gr = 0.5 * expit(h_bias + (r @ kernel) * self._sigma2_factor)
        return gr  # .sum()

    def grad_kernel(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. weight matrix"""
        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        gr = self._sigma2_factor2 * r[:, np.newaxis] @ _expit[:, np.newaxis].T
        return gr  # .sum()


if __name__ == "__main__":
    r = np.array(
        [
            0.30471707975443135,
            -1.0399841062404955,
            0.7504511958064572,
            0.9405647163912139,
        ]
    )
    v_bias = np.array([-0.00631753, 0.01129719, -0.001397, -0.01849913])
    h_bias = np.array([0.00869276, -0.00643394])
    kernel = np.array(
        [
            [-0.40775875, 0.08298116],
            [-0.36875534, 0.03443719],
            [0.40923255, -0.04661963],
            [-1.21311022, 0.80609878],
        ]
    )

    system = AniRBM()
    print("wf eval:", system.wf(r, v_bias, h_bias, kernel))
    print("logprob:", system.logprob(r, v_bias, h_bias, kernel))
    print("grad v_bias", system.grad_v_bias(r, v_bias, h_bias, kernel))
    print("grad h_bias", system.grad_h_bias(r, v_bias, h_bias, kernel))
    print("grad kernel", system.grad_kernel(r, v_bias, h_bias, kernel))
    print("drift force:", system.drift_force(r, v_bias, h_bias, kernel))
    print("local energy:", system.local_energy(r, v_bias, h_bias, kernel))

    """
    JAX:
    wf eval: 0.10676813669683652
    logprob: 0.21353627339367304
    grad v_bias 0.4853326778558034
    grad h_bias 0.5158698886876459
    grad kernel 0.4930420712853883
    drift force: [-0.4031509   0.94078729 -0.63485155 -0.85867653]
    local energy: 1.8890100437817727

    Analytical:
    wf eval: 0.10676813669683638
    logprob: 0.21353627339367276
    grad v_bias 0.4853326778558035
    grad h_bias 0.5158698886876458
    grad kernel 0.49304207128538846
    drift force: [-0.4031509   0.94078729 -0.63485155 -0.85867653]
    local energy: 1.8890100437817727

    energy is right
    """
