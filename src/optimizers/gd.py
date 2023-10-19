import numpy as np

from .optimizer import Optimizer


class Gd(Optimizer):
    """Gradient descent optimizer."""

    def __init__(self, params, eta):
        """Initialize the optimizer.

        Args:
            params (list): List of parameters to optimize.
            lr (float): Learning rate.
        """
        super().__init__(eta)
        self._param_keys = params.keys()

    def step(self, params, grads, sr_matrices=None):
        """Update the parameters. Maybe performance bottleneck?"""

        if sr_matrices is not None:
            for key, sr_matrix in sr_matrices.items():
                # for the love of god change this later
                grads[key] = grads[key].reshape(sr_matrix.shape[0], -1)
                grads[key] = np.linalg.pinv(sr_matrix) @ grads[key]
                grads[key] = grads[key].reshape(params.get([key])[0].shape)

        for key, grad in grads.items():
            params.set([key], [params.get(key) - self.eta * grad])

        return params
