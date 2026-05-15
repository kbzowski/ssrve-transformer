from dataclasses import dataclass
from typing import Optional

from multiphasessrve.types.json_serializable_mixin import JSONSerializableMixin


@dataclass
class ScalarResult(JSONSerializableMixin):
    """Container for single numeric results."""

    value: float
    name: Optional[str] = None
    unit: str = ""  # Optional unit of measurement

    def __str__(self):
        return f"ScalarResult(value={self.value})"
