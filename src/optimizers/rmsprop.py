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

    def step(self, params, grads, sr_matrices=None):
        """Update the parameters. Maybe performance bottleneck?"""

        # grads_dict = {key: grad for key, grad in zip(self._param_keys, grads)}
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

            self._v_params[v_key] = (
                self.beta * self._v_params[v_key] + (1 - self.beta) * grads[key] ** 2
            )

            current_value = params.get(key)

            updated_value = current_value - self.eta * grads[key] / (
                np.sqrt(self._v_params[v_key] + self.epsilon)
            )
            params.set([key], [updated_value])
