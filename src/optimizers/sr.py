import numpy as np

from .optimizer import Optimizer


class Sr(Optimizer):
    """Gradient descent optimizer."""

    def __init__(self, params, eta):
        """Initialize the optimizer.

        Args:
            params (list): List of parameters to optimize.
            lr (float): Learning rate.
        """
        super().__init__(eta)
        self._param_keys = params.keys()
        self.t = 0
        self.delta0 = 1e-1
        self.delta_1 = 1e-3
        self.trust_regions = {key: None for key in self._param_keys}

    def step(self, params, grads, sr_matrices=None):
        """Update the parameters. Maybe performance bottleneck?"""

        self.t += 1  # increment time step

        if sr_matrices is not None:
            for key, sr_matrix in sr_matrices.items():
                grads[key] = grads[key].reshape(sr_matrix.shape[0], -1)
                inv_sr_matrix = np.linalg.pinv(sr_matrix)
                condit_grad = inv_sr_matrix @ grads[key]

                dgd = np.linalg.norm(
                    condit_grad.T @ grads[key]
                )  # this is dtheta^T * g dtheta # TODO: check this dim

                self.trust_regions[key] = np.min(
                    [self.delta0, np.sqrt(self.delta_1 / dgd)]
                )
                grads[key] = condit_grad.reshape(
                    params.get(key).shape
                )  # sheng cals this Gj

        # Fj is what sheng calls the original grad without the sr matrix
        for key, grad in grads.items():
            params.set([key], [params.get(key) - (self.trust_regions[key]) * grad])
            # params.set([key], [params.get(key) - (self.eta/np.sqrt(self.t)) * grad])
            # params.set([key], [params.get(key) - (self.eta) * grad])
