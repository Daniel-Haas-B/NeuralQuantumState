import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    """Gradient descent optimizer."""

    def __init__(self, eta, **kwargs):
        """Initialize the optimizer.

        Args:
            params (list): List of parameters to optimize.
            lr (float): Learning rate.
        """
        super().__init__(eta)

        self.v_bias = None
        self.h_bias = None
        self.kernel = None

        self.m_v_bias = None
        self.v_v_bias = None

        self.m_h_bias = None
        self.v_h_bias = None

        self.m_kernel = None
        self.v_kernel = None
        self.beta1 = kwargs["beta1"]
        self.beta2 = kwargs["beta2"]
        self.epsilon = kwargs["epsilon"]

        self.t = None

    def init(self, v_bias, h_bias, kernel):
        self.t = 0

        self.v_bias = v_bias
        self.h_bias = h_bias
        self.kernel = kernel
        self.m_v_bias = np.zeros_like(v_bias)
        self.v_v_bias = np.zeros_like(v_bias)

        self.m_h_bias = np.zeros_like(h_bias)
        self.v_h_bias = np.zeros_like(h_bias)

        self.m_kernel = np.zeros_like(kernel)
        self.v_kernel = np.zeros_like(kernel)

    def step(self, grads, sr_matrix=None):
        """Update the parameters."""
        self.t += 1  # increment time step

        if sr_matrix is not None:
            grads[-1] = grads[-1].reshape(sr_matrix.shape[0], -1)
            grads[-1] = np.linalg.pinv(sr_matrix) @ grads[-1]
            grads[-1] = grads[-1].reshape(self.kernel.shape)

        param_keys = ["v_bias", "h_bias", "kernel"]
        params = {key: getattr(self, key) for key in param_keys}
        m_params = {key: getattr(self, "m_" + key) for key in param_keys}
        v_params = {key: getattr(self, "v_" + key) for key in param_keys}
        grads_dict = {key: grad for key, grad in zip(param_keys, grads)}

        for key in param_keys:
            # Update m and v with the new gradients
            m_params[key] = (
                self.beta1 * m_params[key] + (1 - self.beta1) * grads_dict[key]
            )
            v_params[key] = (
                self.beta2 * v_params[key] + (1 - self.beta2) * grads_dict[key] ** 2
            )

            # Calculate bias-corrected estimates
            m_hat = m_params[key] / (1 - self.beta1**self.t)
            v_hat = v_params[key] / (1 - self.beta2**self.t)

            # Update parameters using Adam optimization formula
            params[key] -= self.eta * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Set the updated values back to the instance variables
            setattr(self, key, params[key])
            setattr(self, "m_" + key, m_params[key])
            setattr(self, "v_" + key, v_params[key])

        return params["v_bias"], params["h_bias"], params["kernel"]
