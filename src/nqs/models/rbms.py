import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .base_analytical_rbm import BaseRBM
from .base_jax_rbm import BaseJAXRBM

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class NIRBM(BaseRBM):
    """Non-interacting with analytical"""

    def __init__(self, sigma2=1.0, factor=0.5):
        super().__init__(sigma2=sigma2, factor=factor)

    def potential(self, r):
        """Potential energy function"""
        return 0.5 * np.sum(r * r)


class IRBM(BaseRBM):
    """Interacting with analytical"""

    def __init__(self, nparticles, dim, sigma2=1.0, factor=0.5):
        super().__init__(sigma2=sigma2, factor=factor)
        self._N = nparticles
        self._dim = dim

    def potential(self, r):
        """Potential energy function"""
        # HO trap
        v_trap = 0.5 * np.sum(r * r)

        # Interaction
        r_cpy = copy.deepcopy(r).reshape(self._N, self._dim)
        r_dist = np.linalg.norm(r_cpy[None, ...] - r_cpy[:, None], axis=-1)
        v_int = np.sum(np.triu(1 / r_dist, k=1))
        return v_trap + v_int


class JAXNIRBM(BaseJAXRBM):
    """Non-interacting with JAX"""

    def __init__(self, sigma2=1.0, factor=0.5):
        super().__init__(sigma2=sigma2, factor=factor)

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        """Potential energy function"""
        return 0.5 * jnp.sum(r * r)


class JAXIRBM(BaseJAXRBM):
    """Interacting with JAX"""

    def __init__(self, nparticles, dim, sigma2=1.0, factor=0.5):
        super().__init__(sigma2=sigma2, factor=factor)
        self._N = nparticles
        self._dim = dim

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, r):
        """Potential energy function"""
        # HO trap
        v_trap = 0.5 * jnp.sum(r * r)

        # Interaction
        r_cpy = copy.deepcopy(r).reshape(self._N, self._dim)
        r_dist = jnp.linalg.norm(r_cpy[None, ...] - r_cpy[:, None], axis=-1)
        v_int = jnp.sum(jnp.triu(1 / r_dist, k=1))
        return v_trap + v_int


if __name__ == "__main__":
    """
    # [0.27273297 0.20895334]
    v_bias = jnp.array([-0.02033396, -0.01811685])
    h_bias = jnp.array([-0.0033904, -0.00575564])
    kernel = jnp.array([[1.555187, -1.9820777],
                        [-1.4803927,  0.00444124]])
    r = jnp.array([[0.2, 0.5], [0.3, 0.7]])
    """

    P = 2  # particles
    dim = 2  # dimensionality
    M = P * dim  # visible neurons
    N = 2  # hidden neurons

    rng = np.random.default_rng(42)
    r = rng.standard_normal(size=(M,))
    v_bias = rng.standard_normal(size=(M,))
    h_bias = rng.standard_normal(size=(N,))
    kernel = rng.standard_normal(size=(M, N))

    nirbm = NIRBM()
    jaxnirbm = JAXNIRBM()
    print("wf eval:", nirbm.wf(r, v_bias, h_bias, kernel))
    print("logprob:", nirbm.logprob(r, v_bias, h_bias, kernel))
    print("grad v_bias", nirbm.grad_v_bias(r, v_bias, h_bias, kernel).sum())
    print("grad h_bias", nirbm.grad_h_bias(r, v_bias, h_bias, kernel).sum())
    print("grad kernel", nirbm.grad_kernel(r, v_bias, h_bias, kernel).sum())
    print("drift force:", nirbm.drift_force(r, v_bias, h_bias, kernel))
    print("local energy:", nirbm.local_energy(r, v_bias, h_bias, kernel))
