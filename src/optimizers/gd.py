from .optimizer import Optimizer


class Gd(Optimizer):
    """Gradient descent optimizer."""

    def __init__(self, params, lr=1e-3):
        """Initialize the optimizer.

        Args:
            params (list): List of parameters to optimize.
            lr (float): Learning rate.
        """
        super().__init__(params, lr)

    def step(self):
        """Update the parameters."""
        for param in self.params:
            param.data -= self.lr * param.grad

    def train(self):
        return self.method.train(self)
