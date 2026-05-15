from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from multiphasessrve.types.json_serializable_mixin import JSONSerializableMixin


@dataclass
class ArrayResult(JSONSerializableMixin):
    """Container for array results."""

    values: NDArray[np.float64]
    name: Optional[str] = None
    unit: str = ""

    def __str__(self):
        return f"ArrayResult(values={self.values})"
