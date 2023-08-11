#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import abstractmethod

import numpy as np
from scipy.special import expit


class BaseRBM:
    """Base class for creating a quantum system where the wave function is
    represented by a gaussian-binary restricted Boltzmann machine.

    The implementation assumes a logarithmic wave function.
    """

    def __init__(self, sigma2=1.0, factor=0.5):
        self._sigma2 = sigma2
        self._factor = factor
        self._rbm_psi_repr = 2 * self._factor
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

    def grad_v_bias(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. visible bias"""
        gr = (r - v_bias) * self._sigma2
        gr *= self._factor
        return gr  # .sum()

    def grad_h_bias(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. hidden bias"""
        gr = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        gr *= self._factor
        return gr  # .sum()

    def grad_kernel(self, r, v_bias, h_bias, kernel):
        """Gradient of wave function w.r.t. weight matrix"""
        _expit = expit(h_bias + (r @ kernel) * self._sigma2_factor)
        gr = self._sigma2 * r[:, np.newaxis] @ _expit[:, np.newaxis].T
        gr *= self._factor
        return gr  # .sum()

    @property
    def sigma2(self):
        return self._sigma2

    @sigma2.setter
    def sigma2(self, value):
        self._sigma2 = value
        self._precompute()
