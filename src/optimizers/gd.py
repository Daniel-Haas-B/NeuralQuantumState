import numpy as np

from .optimizer import Optimizer


class Gd(Optimizer):
    """Gradient descent optimizer."""

    def __init__(self, eta):
        """Initialize the optimizer.

        Args:
            params (list): List of parameters to optimize.
            lr (float): Learning rate.
        """
        super().__init__(eta)

    def step(self, params, grads, sr_matrix=None):
        """Update the parameters. Maybe performance bottleneck?"""

        # for the love of god change this later
        param_keys = params.keys()
        if sr_matrix is not None:
            grads[-1] = grads[-1].reshape(sr_matrix.shape[0], -1)
            grads[-1] = np.linalg.pinv(sr_matrix) @ grads[-1]
            grads[-1] = grads[-1].reshape(params.get(["kernel"])[0].shape)

        for key, grad in zip(param_keys, grads):
            params.set([key], params.get([key]) - self.eta * grad)

        return params
