#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class RBM(nn.Module):
    """Gaussian-Binary Restricted Boltzmann Machine

    RBM with one visible and one hidden layer – equivalent to a 2-layer FFNN
    with a nonlinear activation function in between.

    Initializers: https://jax.readthedocs.io/en/latest/jax.nn.initializers.html
    """

    # Number of hidden neurons
    nhidden: int = 2
    # Variance
    sigma2: float = 1.0
    # Precision
    param_dtype: Any = np.float64
    # Initializer for the Dense layer matrix
    kernel_init: Callable = nn.initializers.he_normal()
    # Initializer for the hidden bias
    hidden_bias_init: Callable = nn.initializers.normal(stddev=0.01)
    # Initializer for the visible bias
    visible_bias_init: Callable = nn.initializers.normal(stddev=0.01)

    @nn.compact
    def __call__(self, input):
        # Visible layer
        v_bias = self.param(
            "visible_bias",
            self.visible_bias_init,
            (input.shape[-1],),
            self.param_dtype,
        )

        x_v = jnp.linalg.norm(input - v_bias)
        x_v *= -x_v / (4 * self.sigma2)

        # Hidden layer
        layer = nn.Dense(
            name="hidden_layer",
            features=self.nhidden,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
            param_dtype=self.param_dtype,
        )

        x_h = layer(input / self.sigma2)
        x_h = nn.softplus(x_h)
        x_h = jnp.sum(x_h, axis=-1)
        x_h *= 0.5

        return x_v + x_h


if __name__ == "__main__":
    P = 2
    dim = 2
    nhidden = 2

    rng = np.random.default_rng(42)

    P = 2
    dim = 2
    M = P * dim
    r = rng.standard_normal(size=M)

    psi = RBM(nhidden=nhidden, sigma2=1.0)
    params = psi.init(jax.random.PRNGKey(0), r)
    output = psi.apply(params, r)

    print("init pos", list(r))
    print("init params", params)
    print("eval", output)

    # print(jax.tree_map(lambda x: x.shape, params))

    def gradient(model, params, arg):
        gr = jax.grad(lambda p, y: model.apply(p, y))(params, arg)["params"]
        return gr

    def gradient2(model, params, arg):
        gr = jax.grad(lambda p, y: model.apply(p, y), argnums=1)(params, arg)
        return gr

    gr = gradient(psi, params, r)
    print("Gradients")
    print(jax.tree_map(lambda x: x.sum(), gr))

    F = 2 * gradient2(psi, params, r)
    print("F =", F)

    """
    P = 2
    d = 2
    M = P * d  # visible
    N = 2  # hidden
    print(M, N)
    """
