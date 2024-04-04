import numpy as np

from .optimizer import Optimizer


class RmsProp(Optimizer):
    """Adam optimizer."""

    def __init__(self, params, eta, **kwargs):
        """Initialize the optimizer.

        Args:
            params (list): List of parameters to optimize.
            lr (float): Learning rate.
        """
        super().__init__(eta)
        self._param_keys = params.keys()

        self._v_params = {
            "v_" + key: np.zeros_like(params.get(key)[0]) for key in self._param_keys
        }

        self.beta = kwargs["beta"]
        self.epsilon = kwargs["epsilon"]

    def step(self, params, grad_params_E, sr_matrices=None):
        """Update the parameters"""

        for key in self._param_keys:
            # Update m and v with the new gradients
            v_key = "v_" + key

            self._v_params[v_key] = (
                self.beta * self._v_params[v_key]
                + (1 - self.beta) * grad_params_E[key] ** 2
            )

            current_value = params.get(key)

            updated_value = current_value - self.eta * grad_params_E[key] / (
                np.sqrt(self._v_params[v_key] + self.epsilon)
            )
            params.set([key], [updated_value])
