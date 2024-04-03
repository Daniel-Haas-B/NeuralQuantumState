import sys

sys.path.append("/Users/haas/Documents/Masters/GANQS/src/nqs")

import pytest
from models.ffnn import FFNN
from numpy.random import default_rng
import numpy as np
import copy

# Constants for tests
nparticles = 3
dim = 2
layer_sizes = [3, 3, 1]
activations = ["sigmoid", "sigmoid", "exp"]
factor = 1.0
sigma2 = 1.0
rng = default_rng()
r_test = rng.standard_normal(size=nparticles * dim)
ffnn = FFNN(nparticles, dim, layer_sizes, activations, factor, sigma2, rng=rng)


def finite_difference_gradient(func, x, epsilon=1e-4):
    grad_approx = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad_approx[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
    return grad_approx


def finite_difference_param_gradient(func, params, epsilon=1e-7):
    grad_approx = {}

    for key in params.keys():
        param_shape = params.get(key).shape
        grad_approx[key] = np.zeros(param_shape)

        for index in np.ndindex(param_shape):
            params_plus = copy.deepcopy(params)
            # convert to np array
            params_plus.set(key, np.array(params_plus.get(key)))
            params_minus = copy.deepcopy(params)
            # convert to np array
            params_minus.set(key, np.array(params_minus.get(key)))

            params_plus.get(key)[index] += epsilon
            params_minus.get(key)[index] -= epsilon

            grad_approx[key][index] = (func(params_plus) - func(params_minus)) / (
                2 * epsilon
            )

    return grad_approx


def finite_difference_laplacian(func, r, epsilon=1e-4):
    laplacian_approx = np.zeros_like(r)

    for i in range(len(r)):
        r_plus_epsilon = np.copy(r)
        r_minus_epsilon = np.copy(r)
        r_plus_2epsilon = np.copy(r)
        r_minus_2epsilon = np.copy(r)

        r_plus_epsilon[i] += epsilon
        r_minus_epsilon[i] -= epsilon
        r_plus_2epsilon[i] += 2 * epsilon
        r_minus_2epsilon[i] -= 2 * epsilon

        # Approximate second derivative for each dimension
        laplacian_approx[i] = (
            -func(r_plus_2epsilon)
            + 16 * func(r_plus_epsilon)
            - 30 * func(r)
            + 16 * func(r_minus_epsilon)
            - func(r_minus_2epsilon)
        ) / (12 * epsilon**2)

    # Sum over all dimensions to get the total Laplacian
    return laplacian_approx.sum()


def test_gradient_wrt_position():
    grad_wf = ffnn.grad_wf(r_test)
    grad_wf_approx = finite_difference_gradient(
        lambda r: ffnn.wf(r, ffnn.params), r_test
    )
    assert (
        np.linalg.norm(grad_wf - grad_wf_approx) < 1e-3
    ), "Gradient w.r.t position test failed"


def test_gradient_wrt_parameters():
    func_to_test = lambda p: ffnn.wf(r_test, p)  # noqa: E731
    grad_params_computed = ffnn.grad_params(r_test)
    grad_params_approx = finite_difference_param_gradient(func_to_test, ffnn.params)

    for key in grad_params_computed.keys():
        assert (
            np.linalg.norm(grad_params_computed[key] - grad_params_approx[key]) < 1e-3
        ), f"Gradient w.r.t parameters test failed for parameter {key}"


def test_laplacian():
    laplacian_computed = ffnn.laplacian(r_test)
    laplacian_approx = finite_difference_laplacian(
        lambda r: ffnn.wf(r, ffnn.params), r_test
    )
    assert np.abs(laplacian_computed - laplacian_approx) < 1e-3, "Laplacian test failed"


if __name__ == "__main__":
    pytest.main()
