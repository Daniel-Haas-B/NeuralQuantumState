# import jax
import jax.numpy as jnp
import numpy as np
from your_module import FFNN  # Import your FFNN class


def gaussian_wave_function(x, alpha):
    return jnp.exp(-alpha * x**2)


def compute_analytical_sr_matrix(alpha):
    # Placeholder function for analytical SR matrix calculation
    # Implement the analytical SR matrix calculation here
    pass


def compute_numerical_sr_matrix(ffnn, x, alpha):
    # Use your FFNN implementation to compute the SR matrix
    params = {"alpha": alpha}  # Assuming alpha is a parameter in your model
    # wf_value = ffnn.wf(x, params)
    grad_params = ffnn.grad_params(x, params)
    # Compute SR matrix using your method
    sr_matrix = ffnn.compute_sr_matrix(grad_params)
    return sr_matrix


def test_sr_matrix(alpha, x):
    # Compute SR matrices
    analytical_sr = compute_analytical_sr_matrix(alpha)
    ffnn = FFNN(...)  # Initialize your FFNN with appropriate parameters
    numerical_sr = compute_numerical_sr_matrix(ffnn, x, alpha)

    # Compare both matrices
    assert np.allclose(analytical_sr, numerical_sr), "SR matrix mismatch"


# Example test case
alpha_test = 0.5
x_test = jnp.linspace(-1, 1, 100)  # Test points
test_sr_matrix(alpha_test, x_test)

print("SR sanity test passed!")
