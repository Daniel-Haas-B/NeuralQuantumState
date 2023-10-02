import copy

import numpy as np

# from functools import partial
# import jax
# import jax.numpy as jnp


class HarmonicOscillator:
    def __init__(
        self,
        nparticles,
        dim,
        int_type,
        backend,
    ):
        """
        note that nparticle and dim is part of the wavefunction,
        for example the RBM.
        """
        self._N = nparticles
        self._dim = dim
        self._int_type = int_type
        self._backend = backend

    def potential(self, r):
        """Potential energy function"""
        # HO trap
        v_trap = 0.5 * np.sum(r * r)
        v_int = 0.0

        # Interaction
        if self._int_type.lower() == "coulomb":
            r_cpy = copy.deepcopy(r).reshape(self._N, self._dim)

            if self._backend == "numpy":
                r_dist = np.linalg.norm(r_cpy[None, ...] - r_cpy[:, None], axis=-1)
                v_int = np.sum(np.triu(1 / r_dist, k=1))
            elif self._backend == "jax":
                raise NotImplementedError
                # r_dist = jnp.linalg.norm(r_cpy[None, ...] - r_cpy[:, None], axis=-1)
                # v_int = jnp.sum(jnp.triu(1 / r_dist, k=1))
            else:
                raise ValueError("Invalid interaction type")

        return v_trap + v_int

    def _local_kinetic_energy(self, wf, r, params):
        """Evaluate the local kinetic energy of the system"""
        _laplace = wf.laplacian_wf(r, params).sum()
        _grad = wf.grad_wf(r, params)
        _grad2 = np.sum(_grad * _grad)
        return -0.5 * (_laplace + _grad2)

    def local_energy(self, wf, r, params):
        """Local energy of the system"""
        ke = self._local_kinetic_energy(wf, r, params)
        pe = self.potential(r)
        return pe + ke

    def drift_force(self, r, v_bias, h_bias, kernel):
        """Drift force at each particle's location"""
        F = 2 * self._grad_wf(r, v_bias, h_bias, kernel)
        return F
