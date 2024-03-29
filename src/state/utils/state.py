# DISCLAIMER: Idea and code structure from blackjax
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Union

import jax.numpy as jnp
import numpy as np

# from typing import Callable

Array = Union[np.ndarray, jnp.ndarray]  # either numpy or jax numpy array
PyTree = Union[
    Array, Iterable[Array], Mapping[Any, Array]
]  # either array, iterable of arrays or mapping of arrays

from dataclasses import dataclass


@dataclass(frozen=False)
class State:
    positions: PyTree
    logp: Union[float, PyTree]
    n_accepted: int
    delta: int

    def __init__(self, positions, logp, n_accepted=0, delta=0):
        self.positions = positions
        self.logp = logp
        self.n_accepted = n_accepted
        self.delta = delta

    def create_batch_of_states(self, batch_size):
        """ """
        # Replicate each property of the state

        batch_positions = np.array([self.positions] * batch_size)
        batch_logp = np.array([self.logp] * batch_size)
        batch_n_accepted = np.array([self.n_accepted] * batch_size)
        batch_delta = np.array([self.delta] * batch_size)

        # Create a new State object with these batched properties
        batch_state = State(batch_positions, batch_logp, batch_n_accepted, batch_delta)
        return batch_state

    def __repr__(self):
        return f"State(positions={self.positions}, logp={self.logp}, n_accepted={self.n_accepted}, delta={self.delta})"

    def __getitem__(self, key):
        return State(
            self.positions[key], self.logp[key], self.n_accepted[key], self.delta[key]
        )

    def __setitem__(self, key, value):
        self.positions[key] = value.positions
        self.logp[key] = value.logp
        self.n_accepted[key] = value.n_accepted
        self.delta[key] = value.delta
