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


@dataclass(frozen=False)  # frozen=False allows for mutable dataclass
class State:
    positions: PyTree
    logp: Union[float, PyTree]
    n_accepted: int
    delta: int
    sign: int

    def __init__(self, positions, logp, n_accepted=0, delta=0, sign=0):
        self.positions = positions
        self.logp = logp
        self.n_accepted = n_accepted
        self.delta = delta
        self.sign = sign

    def create_batch_of_states(self, batch_size):
        batch_states = [
            State(
                self.positions.copy(),
                self.logp,
                self.n_accepted,
                self.delta + i,
                self.sign,
            )
            for i in range(batch_size)
        ]
        return BatchedStates(batch_states)

    def __repr__(self):
        return f"State(positions={self.positions}, logp={self.logp}, n_accepted={self.n_accepted}, delta={self.delta}, sign={self.sign})"

    def __getitem__(self, key):
        return State(
            self.positions[key],
            self.logp[key],
            self.n_accepted[key],
            self.delta[key],
            self.sign[key],
        )

    def __setitem__(self, key, value):
        self.positions[key] = value.positions
        self.logp[key] = value.logp
        self.n_accepted[key] = value.n_accepted
        self.delta[key] = value.delta
        self.sign[key] = value.sign


class BatchedStates:
    def __init__(self, states):
        self.states = states

    @property
    def positions(self):
        return np.array([state.positions for state in self.states])

    @positions.setter
    def positions(self, new_positions):
        for i, state in enumerate(self.states):
            state.positions = new_positions[i]

    @property
    def signs(self):
        return np.array([state.sign for state in self.states])

    @signs.setter
    def signs(self, new_signs):
        for i, state in enumerate(self.states):
            state.sign = new_signs[i]

    @property
    def logp(self):
        return np.array([state.logp for state in self.states])

    @logp.setter
    def logp(self, new_logp):
        for i, state in enumerate(self.states):
            state.logp = new_logp[i]

    @property
    def n_accepted(self):
        return np.array([state.n_accepted for state in self.states])

    @n_accepted.setter
    def n_accepted(self, new_n_accepted):
        for i, state in enumerate(self.states):
            state.n_accepted = new_n_accepted[i]

    @property
    def delta(self):
        return np.array([state.delta for state in self.states])

    @delta.setter
    def delta(self, new_delta):
        for i, state in enumerate(self.states):
            state.delta = new_delta[i]

    def __getitem__(self, key):
        return self.states[key]

    def __setitem__(self, key, value):
        if not isinstance(value, State):
            raise ValueError("Value must be an instance of State.")
        self.states[key] = value

    def __len__(self):
        return len(self.states)
