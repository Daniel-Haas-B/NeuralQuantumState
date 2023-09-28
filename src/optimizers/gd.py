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
        self.v_bias = None
        self.h_bias = None
        self.kernel = None

    def init(self, v_bias, h_bias, kernel):
        self.v_bias = v_bias
        self.h_bias = h_bias
        self.kernel = kernel

    def step(self, grads, sr_matrix=None):
        """Update the parameters. Maybe performance bottleneck?"""

        # change last grad

        # for the love of god change this later
        if sr_matrix is not None:
            grads[-1] = grads[-1].reshape(sr_matrix.shape[0], -1)
            grads[-1] = np.linalg.pinv(sr_matrix) @ grads[-1]
            grads[-1] = grads[-1].reshape(self.kernel.shape)

        params = [self.v_bias, self.h_bias, self.kernel]
        params = [param - self.eta * grad for param, grad in zip(params, grads)]
        self.v_bias, self.h_bias, self.kernel = params

        return self.v_bias, self.h_bias, self.kernel
