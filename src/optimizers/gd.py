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

    def step(self, grads):
        """Update the parameters. Maybe performance bottleneck?"""
        params = [self.v_bias, self.h_bias, self.kernel]
        params = [param - self.eta * grad for param, grad in zip(params, grads)]
        self.v_bias, self.h_bias, self.kernel = params

        return self.v_bias, self.h_bias, self.kernel
