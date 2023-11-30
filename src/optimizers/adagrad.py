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

    def step(self, params, grads, sr_matrices=None):
        """Update the parameters. Maybe performance bottleneck?"""

        if sr_matrices is not None:
            for key in sr_matrices.keys():
                sr_matrix = sr_matrices[key]
                # for the love of god change this later
                grads[key] = grads[key].reshape(sr_matrix.shape[0], -1)
                grads[key] = np.linalg.pinv(sr_matrix) @ grads[key]
                grads[key] = grads[key].reshape(params.get(key).shape)

        for key in self._param_keys:
            # Update m and v with the new gradients

            v_key = "v_" + key
            grads_val = grads[key]

            self._v_params[v_key] = self._v_params[v_key] + grads_val**2

            current_value = params.get(key)
            updated_value = current_value - self.eta * grads_val / (
                np.sqrt(self._v_params[v_key] + self.epsilon)
            )
            params.set([key], [updated_value])
