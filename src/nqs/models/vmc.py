import jax.numpy as jnp
import numpy as np


class VMC:
    def __init__(
        self,
        nparticles,
        dim,
        sigma2=1.0,
        rng=None,
        log=False,
        logger=None,
        logger_level="INFO",
        backend="numpy",
    ):
        self.configure_backend(backend)

    def configure_backend(self, backend):
        if backend == "numpy":
            self.backend = np
            self.la = np.linalg
        elif backend == "jax":
            self.backend = jnp
            self.la = jnp.linalg
        else:
            raise ValueError("Invalid backend:", backend)

    def gaussian_wave_function(self, x, alpha):
        """
        Ψ(x)=exp(-α ∑_{i=1}^{N} x_i.T * x_i)
        """
        return self.backend.exp(-alpha * x**2)
