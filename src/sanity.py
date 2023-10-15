import jax
import jax.numpy as jnp
from pathos.multiprocessing import ProcessPool


class simple_class:
    def __init__(self):
        pass

    def simple_function(self, x):
        return x * x


simple_class_instance = simple_class()


def simple_function_wrapper(x):
    jit_func = jax.jit(simple_class_instance.simple_function)
    return jit_func(x)


with ProcessPool(4) as pool:
    results = pool.map(
        simple_function_wrapper, [jax.device_get(jnp.array([i])) for i in range(4)]
    )
