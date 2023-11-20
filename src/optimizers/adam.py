import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(self, params, eta, **kwargs):
        """Initialize the optimizer.

        Args:
            params (list): List of parameters to optimize.
            lr (float): Learning rate.
        """
        super().__init__(eta)
        self._param_keys = params.keys()
        self.t = 0

        self._m_params = {
            "m_" + key: np.zeros_like(params.get(key)[0]) for key in self._param_keys
        }

        self._v_params = {
            "v_" + key: np.zeros_like(params.get(key)[0]) for key in self._param_keys
        }

        self.beta1 = kwargs["beta1"]
        self.beta2 = kwargs["beta2"]
        self.epsilon = kwargs["epsilon"]

    def step(self, params, grads, sr_matrices=None):
        """Update the parameters. Maybe performance bottleneck?"""
        self.t += 1  # increment time step
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

            m_key = "m_" + key
            v_key = "v_" + key
            self._m_params[m_key] = (
                self.beta1 * self._m_params[m_key] + (1 - self.beta1) * grads[key]
            )

            self._v_params[v_key] = (
                self.beta2 * self._v_params[v_key] + (1 - self.beta2) * grads[key] ** 2
            )

            # Calculate bias-corrected estimates
            m_hat = self._m_params[m_key] / (1 - self.beta1**self.t)
            v_hat = self._v_params[v_key] / (1 - self.beta2**self.t)

            # Update parameters using Adam optimization formula

            current_value = params.get(key)

            updated_value = current_value - self.eta * m_hat / (
                np.sqrt(v_hat) + self.epsilon
            )
            params.set([key], [updated_value])
