import copy

import jax
import jax.numpy as jnp
import numpy as np

# from .base_jax_rbm import BaseJAXRBM
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class Hamiltonian:
    def __init__(
        self,
        nparticles,
        dim,
        int_type,
        backend,
    ):
        self._N = nparticles
        self._dim = dim
        self._int_type = int_type

        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg
            self.potential = jax.jit(self.potential)
        else:
            raise ValueError("Invalid backend:", backend)

    # methods to be overwritten
    def potential(self, r):
        """Potential energy function"""
        raise NotImplementedError

    def _local_kinetic_energy(self, wf, r):
        """Evaluate the local kinetic energy of the system"""
        raise NotImplementedError

    def local_energy(self, wf, r):
        """Local energy of the system"""
        raise NotImplementedError

    def drift_force(self, wf, r):
        """Drift force at each particle's location"""
        raise NotImplementedError


class HarmonicOscillator(Hamiltonian):
    def __init__(
        self,
        nparticles,
        dim,
        int_type,
        backend,
        kwargs,
    ):
        """
        note that nparticle and dim is part of the wavefunction,
        for example the RBM.
        """
        super().__init__(nparticles, dim, int_type, backend)
        self.kwargs = kwargs

    def potential(self, r):
        """Potential energy function"""
        # HO trap
        v_trap = 0.5 * self.backend.sum(r * r)

        # Interaction
        v_int = 0.0
        if self._int_type == "Coulomb":
            r_cpy = copy.deepcopy(r).reshape(self._N, self._dim)
            r_dist = self.la.norm(r_cpy[None, ...] - r_cpy[:, None], axis=-1)
            v_int = self.backend.sum(self.backend.triu(1 / r_dist, k=1))
        elif self._int_type == "Calogero":
            r_cpy = copy.deepcopy(r).reshape(self._N, self._dim)
            r_dist = self.la.norm(r_cpy[None, ...] - r_cpy[:, None], axis=-1)
            v_int = self.backend.sum(self.backend.triu(1 / r_dist**2, k=1))
            v_int *= self.kwargs["coupling"] * (self.kwargs["coupling"] - 1)
        elif self._int_type is not None:
            raise ValueError("Invalid interaction type:", self._int_type)

        return v_trap + v_int

    def _local_kinetic_energy(self, wf, r):
        """Evaluate the local kinetic energy of the system"""
        _laplace = wf.laplacian(r).sum()
        _grad = wf.grad_wf(r)
        _grad2 = self.backend.sum(_grad * _grad)
        return -0.5 * (_laplace + _grad2)

    def local_energy(self, wf, r):
        """Local energy of the system"""
        ke = self._local_kinetic_energy(wf, r)
        pe = self.potential(r)
        return pe + ke

    def drift_force(self, wf, r):
        """Drift force at each particle's location"""
        F = 2 * wf.grad_wf(r)
        return F
