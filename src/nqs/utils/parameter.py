from typing import Dict
from typing import List
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Updating ParameterDataType to account for list, np arrays, or jnp arrays
ParameterDataType = Union[np.ndarray, jnp.ndarray]


class Parameter:
    def __init__(self, data: Dict[str, ParameterDataType] = None) -> None:
        self.data = data if data is not None else {}

    @property
    def type(self) -> str:
        return "Custom Parameter Class"

    def set(
        self,
        names_or_parameter: Union[
            str, List[str], "Parameter", Dict[str, ParameterDataType]
        ],
        values: Union[ParameterDataType, List[ParameterDataType]] = None,
    ) -> None:
        if isinstance(names_or_parameter, Parameter):
            # If names_or_parameter is a Parameter instance, replace the current data
            self.data = names_or_parameter.data
        elif isinstance(names_or_parameter, list) and isinstance(values, list):
            # If names_or_parameter is a list and values is a list, set each key-value pair
            for key, value in zip(names_or_parameter, values):
                self.data[key] = value
        elif isinstance(names_or_parameter, dict):
            # If names_or_parameter is a dictionary, update the current data
            self.data.update(names_or_parameter)
        elif isinstance(names_or_parameter, str) and values is not None:
            # If names_or_parameter is a single key and values is provided, set the key-value pair
            self.data[names_or_parameter] = values
        else:
            raise ValueError("Invalid arguments")

    def get(self, name: str) -> ParameterDataType:
        return self.data[name]

    def keys(self) -> List[str]:
        return list(self.data.keys())

    def to_jax(self) -> "Parameter":
        new_data = {}
        for key, value in self.data.items():
            new_data[key] = jnp.array(value) if isinstance(value, np.ndarray) else value
        return Parameter(new_data)

    def tree_flatten(self):
        # Return the flatten representation (leaves) and auxiliary data (here, just the keys)
        leaves = list(self.data.values())
        aux_data = list(self.data.keys())
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        # Reconstruct an instance of Parameter from leaves and aux_data
        return cls(dict(zip(aux_data, leaves)))

    def __repr__(self) -> str:
        return f"Parameter(data={self.data})"

    def rescale(self, scale: float) -> None:
        for key in self.data:
            self.data[key] *= scale

    def __getitem__(self, key):
        return self.data[key]

    # make it such that a constant number can be multiplied by a parameter type for example 2*Parameter
    def __mul__(self, other):
        new_data = {}
        for key in self.data:
            new_data[key] = self.data[key] * other
        return Parameter(new_data)

    # now allow the other way around
    def __rmul__(self, other):
        return self.__mul__(other)


# Registering the Parameter class with JAX
register_pytree_node(Parameter, Parameter.tree_flatten, Parameter.tree_unflatten)
