import numpy as np

from .optimizer import Optimizer


class Gd(Optimizer):
    """Gradient descent optimizer."""

    def __init__(self, params, eta, gamma=0.9):
        """Initialize the optimizer.

        Args:
            params (list): List of parameters to optimize.
            lr (float): Learning rate.
        """
        super().__init__(eta)
        self._param_keys = params.keys()
        self.v = {key: np.zeros_like(params.get(key)) for key in self._param_keys}
        self.gamma = gamma

    def step(self, params, grads, sr_matrices=None):
        """Update the parameters"""

        for key, grad in grads.items():
            self.v[key] = self.gamma * self.v[key] + grad
            params.set([key], [params.get(key) - self.eta * self.v[key]])
