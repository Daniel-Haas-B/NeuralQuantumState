class Optimizer(object):
    """Base class for all optimizers."""

    def __init__(self, params, lr=1e-3):
        """Initialize the optimizer.

        Args:
            params (list): List of parameters to optimize.
            lr (float): Learning rate.
        """
        self.params = params
        self.method = None
        self.lr = lr

    def step(self):
        """Update the parameters."""
        raise NotImplementedError

    def zero_grad(self):
        """Zero out the gradients."""
        for param in self.params:
            param.zero_grad()

    def set_method(self, method):
        """Set the method."""
        self.method = method

    def train(self):
        """Train the model. To be overridden by subclasses."""
        raise NotImplementedError
