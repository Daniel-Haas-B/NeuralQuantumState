# import jax
import time

import jax
import numpy as np

# import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

print(jax.devices())


a = np.array(np.random.rand(10000, 10000))
b = np.array(np.random.rand(10000, 10000))

start = time.time()
np.dot(a, b)
end = time.time()

print((end - start), "with", jax.devices())
