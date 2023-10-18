from typing import Dict
from typing import List
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Updating ParameterDataType to account for list, np arrays, or jnp arrays
ParameterDataType = Union[List, np.ndarray, jnp.ndarray]


class Parameter:
    def __init__(self) -> None:
        self.data: Dict[str, ParameterDataType] = {}

    def set(
        self,
        names_or_parameter: Union[List[str], "Parameter", Dict[str, ParameterDataType]],
        values: List[ParameterDataType] = None,
    ) -> None:
        if isinstance(names_or_parameter, Parameter):
            self.data = names_or_parameter.data
        elif values is not None:
            for key, value in zip(names_or_parameter, values):
                self.data[key] = value
        elif isinstance(names_or_parameter, dict):
            # Case 3: A dictionary is provided.
            self.data.update(names_or_parameter)  # Merge dictionary with existing data.

        else:
            raise ValueError("Invalid arguments")

    def get(self, names: List[str]) -> List[ParameterDataType]:
        return [self.data[name] for name in names]

    def keys(self) -> List[str]:
        return list(self.data.keys())

    def to_jax(self) -> "Parameter":
        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                self.data[key] = jnp.array(value)
            else:
                self.data[key] = value
        return self.data

    def __repr__(self) -> str:
        return f"Parameter(data={self.data})"
