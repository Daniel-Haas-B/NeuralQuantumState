class Optimizer(object):
    """Base class for all optimizers."""

    def __init__(self, eta):
        """Initialize the optimizer.

        Args:
            params (list): List of parameters to optimize.
            lr (float): Learning rate.
        """
        self.params = None
        self.eta = eta

    def step(self):
        """Update the parameters."""
        raise NotImplementedError

    def train(self):
        """Train the model. To be overridden by subclasses."""
        raise NotImplementedError

    def set_method(self, method):
        """Set the optimization method."""
