from dataclasses import dataclass
from typing import Dict, Union, Optional

import numpy as np
from numpy.typing import NDArray

from multiphasessrve.types.json_serializable_mixin import JSONSerializableMixin


@dataclass
class DictResult(JSONSerializableMixin):
    """Container for dictionary results."""

    values: Dict[str, Union[float, NDArray[np.float64]]]
    name: Optional[str] = None
    units: Dict[str, str] = None  # Optional units for each value

    def __post_init__(self):
        if self.units is None:
            self.units = {key: "" for key in self.values.keys()}

    def __str__(self):
        result = []

        # Add name if available
        if self.name:
            result.append(f"DictResult: {self.name}")
        else:
            result.append("DictResult:")

        # Format each value with its unit
        for key, value in self.values.items():
            unit = self.units.get(key, "")
            unit_str = f" {unit}" if unit else ""

            # Handle different types of values
            if isinstance(value, np.ndarray):
                if value.size == 1:  # Single value array
                    value_str = f"{float(value.item()):.6g}{unit_str}"
                else:  # Multi-value array
                    value_str = f"array(shape={value.shape}){unit_str}"
            elif isinstance(value, list):
                # For lists, format each element properly without showing type info
                formatted_items = []
                for item in value:
                    if isinstance(item, (float, np.float64, np.float32)):
                        formatted_items.append(f"{float(item):.6g}")
                    else:
                        formatted_items.append(str(item))
                value_str = f"[{', '.join(formatted_items)}]{unit_str}"
            elif isinstance(value, (float, np.float64, np.float32)):
                # Handle float types with precision formatting
                value_str = f"{float(value):.6g}{unit_str}"
            else:
                # For other types, use string conversion
                value_str = f"{str(value)}{unit_str}"

            result.append(f"  {key}: {value_str}")

        return "\n".join(result)

    def __repr__(self):
        # For debugging and development, include more details
        values_repr = ", ".join(f"{k}={repr(v)}" for k, v in self.values.items())
        name_repr = f", name={repr(self.name)}" if self.name is not None else ""
        units_repr = f", units={repr(self.units)}" if any(self.units.values()) else ""

        return f"DictResult(values={{{values_repr}}}{name_repr}{units_repr})"
