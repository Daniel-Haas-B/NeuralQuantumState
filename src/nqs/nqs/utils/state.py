# DISCLAIMER: Idea and code structure from blackjax
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import NamedTuple
from typing import Union

import jax.numpy as jnp
import numpy as onp

# from typing import Callable

Array = Union[onp.ndarray, jnp.ndarray]
PyTree = Union[Array, Iterable[Array], Mapping[Any, Array]]


class State(NamedTuple):
    positions: PyTree
    logp: Union[float, PyTree]
    n_accepted: int
    delta: int
