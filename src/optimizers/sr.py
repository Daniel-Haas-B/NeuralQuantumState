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
        self.delta0 = 1e-2  # trust region upper bound. This is the
        self.delta_1 = 1e-3  # trust region lower bound

        self.trust_regions = {key: None for key in self._param_keys}

    def step(self, params, grads, sr_matrices=None):
        """Update the parameters.
        #TODO: we can make this better with einsum
        """

        self.t += 1  # increment time step

        for key, sr_matrix in sr_matrices.items():
            inv_sr_matrix = np.linalg.pinv(sr_matrix)

            # if grads[key].ndim == 1: we want np.einsum("ij,j->i", inv_sr_matrix, grads[key])
            # if grads[key].ndim == 2: we want np.einsum("ij,jk->ik", inv_sr_matrix, grads[key])
            # ein_string = "ij,j" +"k"*(grads[key].ndim-1) + "->i" + "k"*(grads[key].ndim-1)
            # print(f"ein_string: {ein_string}")
            # condit_grad = np.einsum(ein_string, inv_sr_matrix, grads[key]) # will be shape
            # (n, m) where n is the number of parameters and m is the number of elements in the parameter

            # or
            grads[key] = grads[key].reshape(sr_matrix.shape[0], -1)
            condit_grad = inv_sr_matrix @ grads[key]

            dgd = np.linalg.norm(
                condit_grad.T @ grads[key]
            )  # this is dtheta^T * g dtheta # TODO: check this dim

            self.trust_regions[key] = np.min([self.delta0, np.sqrt(self.delta_1 / dgd)])
            grads[key] = condit_grad.reshape(
                params.get(key).shape
            )  # sheng cals this Gj

        # Fj is what sheng calls the original grad without the sr matrix
        for key, grad in grads.items():
            params.set([key], [params.get(key) - (self.trust_regions[key]) * grad])
            # params.set([key], [params.get(key) - (self.eta/np.sqrt(self.t)) * grad])
            # params.set([key], [params.get(key) - (self.eta) * grad])
