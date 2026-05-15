from functools import cached_property
from typing import Tuple, Dict, List, Sequence
from pydantic import RootModel, model_validator, BaseModel


class PhaseModel(BaseModel):
    id: int
    color: Tuple[int, int, int]
    name: str

    @property
    def normalized_color(self) -> Tuple[float, float, float]:
        """Get normalized color of the phase."""
        return tuple(c / 255.0 for c in self.color)


class PhasesModel(RootModel):
    root: Sequence[PhaseModel]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = tuple(self.root)  # Convert sequence to tuple to make it immutable

    @model_validator(mode="after")
    def validate_phases(self) -> "PhasesModel":
        """Validate phase data integrity."""
        if not self.root:
            raise ValueError("At least one phase must be defined")

        # Check for unique IDs
        ids = [phase.id for phase in self.root]
        if len(ids) != len(set(ids)):
            duplicate_ids = [id for id in ids if ids.count(id) > 1]
            raise ValueError(f"Duplicate phase IDs found: {duplicate_ids}")

        # Check for unique names
        names = [phase.name for phase in self.root]
        if len(names) != len(set(names)):
            duplicate_names = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate phase names found: {duplicate_names}")

        # Verify ID sequence is continuous and starts from 1
        sorted_ids = sorted(ids)
        expected_ids = list(range(1, len(sorted_ids) + 1))
        if sorted_ids != expected_ids:
            raise ValueError(
                f"Phase IDs must form a continuous sequence starting from 1. "
                f"Got {sorted_ids}, expected {expected_ids}"
            )

        return self

    @cached_property
    def count(self) -> int:
        """Get number of phases."""
        return len(self.root)

    @cached_property
    def colors_list(self) -> List[Tuple[int, int, int]]:
        """Get list of colors ordered by phase ID."""
        return [phase.color for phase in self.root]

    @cached_property
    def colors_list_normalized(self) -> List[Tuple[float, float, float]]:
        """Get list of normalized colors ordered by phase ID."""
        return [phase.normalized_color for phase in self.root]

    @cached_property
    def phase_names(self) -> Dict[int, str]:
        """Get mapping of phase IDs to their names."""
        return {phase.id: phase.name for phase in self.root}

    @cached_property
    def _name_to_id(self) -> Dict[str, int]:
        """Internal cached mapping of names to IDs."""
        return {phase.name: phase.id for phase in self.root}

    @cached_property
    def _id_to_name(self) -> Dict[int, str]:
        """Internal cached mapping of IDs to names."""
        return {phase.id: phase.name for phase in self.root}

    def get_phase_id(self, phase_name: str) -> int:
        """Get phase ID by name.

        Args:
            phase_name: Name of the phase

        Returns:
            Phase ID

        Raises:
            ValueError: If phase name not found
        """
        if phase_name not in self._name_to_id:
            raise ValueError(f"Phase {phase_name} not found in {self.root}")
        return self._name_to_id[phase_name]

    def get_phase_name(self, phase_id: int) -> str:
        """Get phase name by ID.

        Args:
            phase_id: Phase ID

        Returns:
            Phase name

        Raises:
            ValueError: If phase ID not found
        """
        if phase_id not in self._id_to_name:
            raise ValueError(f"Phase ID {phase_id} not found in {self.root}")
        return self._id_to_name[phase_id]

    def __getitem__(self, index: int) -> PhaseModel:
        """Allow direct indexing into phases."""
        return self.root[index]

    def __iter__(self):
        """Enable iteration over phases."""
        return iter(self.root)
