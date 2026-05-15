from typing import Any
import json
from dataclasses import asdict
from typing import Any

import numpy as np


class JSONSerializableMixin:
    """Mixin class to add JSON serialization capabilities."""

    def __json__(self):
        """Make the object JSON serializable."""
        return self.to_dict()

    def to_json(self) -> str:
        """Convert the object to a JSON string."""
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict:
        """Convert the object to a dictionary."""

        def convert_value(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, "to_dict"):
                return obj.to_dict()
            return obj

        # Convert to dict using dataclasses.asdict
        d = asdict(self)

        # Handle numpy arrays and nested objects
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = {k: convert_value(v) for k, v in value.items()}
            else:
                d[key] = convert_value(value)

        return d

    @classmethod
    def from_json(cls, json_str: str) -> "JSONSerializableMixin":
        """Create an instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "JSONSerializableMixin":
        """Create an instance from a dictionary."""

        def convert_value(value: Any, field_type: type) -> Any:
            if field_type == np.ndarray:
                return np.array(value)
            return value

        # Get type hints for the class
        hints = cls.__annotations__

        # Convert values based on type hints
        for key, field_type in hints.items():
            if key in data:
                data[key] = convert_value(data[key], field_type)

        return cls(**data)
