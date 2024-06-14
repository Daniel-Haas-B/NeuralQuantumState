import numpy as np

from .optimizer import Optimizer


class Sr(Optimizer):
    """Gradient descent optimizer."""

    def __init__(self, params, eta):
        """Initialize the optimizer."""
        super().__init__(eta)
        self._param_keys = params.keys()
        self.t = 0
        self.delta0 = eta * 0.1  # trust region upper bound. This is the
        self.delta_1 = eta * 0.01  # trust region lower bound
        self.trust_regions = {key: None for key in self._param_keys}
        self.v = {key: np.zeros_like(params.get(key)) for key in self._param_keys}
        self.gamma = 0.9

    def step(self, params, grad_params_E, sr_matrices=None):
        """Update the parameters."""

        self.t += 1  # increment time step

        for key, sr_matrix in sr_matrices.items():
            inv_sr_matrix = np.linalg.pinv(sr_matrix)

            grad_params_E[key] = grad_params_E[key].reshape(sr_matrix.shape[0], -1)
            condit_grad = inv_sr_matrix @ grad_params_E[key]

            dgd = np.linalg.norm(
                condit_grad.T @ grad_params_E[key]
            )  # this is dtheta^T * g dtheta # TODO: check this dim

            self.trust_regions[key] = np.min([self.delta0, np.sqrt(self.delta_1 / dgd)])
            grad_params_E[key] = condit_grad.reshape(
                params.get(key).shape
            )  # sheng cals this Gj

        # Fj is what sheng calls the original grad without the sr matrix
        for key, grad in grad_params_E.items():
            self.v[key] = self.gamma * self.v[key] + (self.trust_regions[key]) * grad
            params.set([key], [params.get(key) - self.v[key]])
            # params.set([key], [params.get(key) - (self.eta / np.sqrt(self.t)) * grad])
            # params.set([key], [params.get(key) - (self.eta) * grad])
