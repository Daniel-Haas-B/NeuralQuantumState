import numpy as np

from .optimizer import Optimizer


class Adagrad(Optimizer):
    """Adagrad optimizer."""

    def __init__(self, params, eta, **kwargs):
        """Initialize the optimizer.

        Args:
            params (list): List of parameters to optimize.
            eta (float): Learning rate.
        """
        super().__init__(eta)
        self._param_keys = params.keys()

        self._v_params = {
            "v_" + key: np.zeros_like(params.get(key)) for key in self._param_keys
        }

        self.epsilon = kwargs["epsilon"] if "epsilon" in kwargs else 1e-8

    def step(self, params, grad_params_E, sr_matrices=None):
        """Update the parameters."""

        for key in self._param_keys:
            # Update m and v with the new gradients
            v_key = "v_" + key
            self._v_params[v_key] = self._v_params[v_key] + grad_params_E[key] ** 2

            current_value = params.get(key)
            updated_value = current_value - self.eta * grad_params_E[key] / (
                np.sqrt(self._v_params[v_key] + self.epsilon)
            )
            params.set([key], [updated_value])
