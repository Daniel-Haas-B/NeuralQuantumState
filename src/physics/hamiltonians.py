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
        """
        Note that this assumes that the wavefunction form is in the log domain
        """
        self._N = nparticles
        self._dim = dim
        self._int_type = int_type

        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg

            # self.potential = jax.jit(self.potential)
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
        # self.potential = jax.jit(self.potential) # if we regularize we cannot jit

        self.kwargs = kwargs

        self.reg_decay = (1e-10 / self.kwargs.get("r0_reg", 1.0)) ** (
            1 / self.kwargs.get("training_cycles", 1.0)
        )

    def regularized_potential(self, r):
        """Regularize the potential"""
        r0 = self.kwargs.get("r0_reg", 1.0)
        return self.backend.tanh(r / r0)

    def potential(self, r):
        """Potential energy function"""
        # HO trap
        v_trap = 0.5 * self.backend.sum(r * r) * self.kwargs["omega"]

        self.kwargs["r0_reg"] = self.kwargs["r0_reg"] * self.reg_decay
        # with open("decay.csv", "a") as f:
        #    f.write(str(self.kwargs["r0_reg"]) + "\n")
        # f.close()

        # Interaction
        v_int = 0.0
        if self._int_type == "Coulomb":
            r_cpy = copy.deepcopy(r).reshape(self._N, self._dim)
            r_dist = self.la.norm(r_cpy[None, ...] - r_cpy[:, None], axis=-1)

            # Apply tanh regularization
            f_r = 1  # self.regularized_potential(r_dist)
            v_int = self.backend.sum(
                self.backend.triu(f_r / r_dist, k=1)
            )  # k=1 to remove diagonal, since we don't want self-interaction

        elif self._int_type == "Calogero":
            r_cpy = copy.deepcopy(r).reshape(self._N, self._dim)
            r_dist = self.la.norm(r_cpy[None, ...] - r_cpy[:, None], axis=-1)
            v_int = self.backend.sum(
                self.backend.triu(1 / r_dist**2, k=1)
            )  # k=1 to remove diagonal, since we don't want self-interaction
            v_int *= self.kwargs["coupling"] * (self.kwargs["coupling"] - 1)
        elif self._int_type is not None:
            raise ValueError("Invalid interaction type:", self._int_type)

        return v_trap + v_int

    def _local_kinetic_energy(self, wf, r):
        """Evaluate the local kinetic energy of the system"""
        _laplace = wf.laplacian(r).sum()  # summing over all particles
        _grad = wf.grad_wf(r)
        _grad2 = self.backend.sum(_grad * _grad)  # summing over all particles
        return -0.5 * (_laplace + _grad2)

    def local_energy(self, wf, r):
        """Local energy of the system"""
        # print("r inside local energy", r)
        # time.sleep(1)
        # print("r0_reg", self.kwargs["r0_reg"])
        ke = self._local_kinetic_energy(wf, r)
        pe = self.potential(r)

        return pe + ke

    def turn_reg_off(self):
        # overwrite the regularized potential
        def regularized_potential(r):
            """Regularize the potential"""
            return 1

        self.regularized_potential = regularized_potential

    def drift_force(self, wf, r):
        """Drift force at each particle's location"""
        F = 2 * wf.grad_wf(r)
        return F
