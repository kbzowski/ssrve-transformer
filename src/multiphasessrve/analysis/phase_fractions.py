import numpy as np
from collections import Counter

from multiphasessrve.analysis.registry import registry
from multiphasessrve.ssrve import Ssrve
from multiphasessrve.types.dict_result import DictResult


@registry.register("phase_fractions")
class VolumePhaseAnalyzer:
    """Phase fraction analysis for SSRVE."""

    @staticmethod
    def analyze(ssrve: Ssrve, **kwargs) -> DictResult:
        """Calculate volume fraction of each phase in the SSRVE.

        Args:
            ssrve: SSRVE instance
            **kwargs: Additional parameters (unused in this analysis)

        Returns:
            DictResult containing phase fractions mapping phase names to their volume fractions
        """
        if any(props.phase_id < 1 for props in ssrve.grain_properties.values()):
            raise ValueError(
                "Invalid phase ID detected: phase IDs must be >= 1. This can happen if growth steps were skipped."
            )

        rows, cols = ssrve.get_rows(), ssrve.get_cols()
        total_cells = rows * cols
        filled_cells = 0

        # Use Counter for efficient phase counting
        phase_counter = Counter()

        # Count phases using grain properties
        for x in range(rows):
            for y in range(cols):
                grain_id = ssrve.arr[x, y]
                if grain_id != 0:  # Skip empty cells
                    filled_cells += 1
                    phase_id = ssrve.grain_properties[grain_id].phase_id

                    if phase_id <= 0:
                        raise ValueError("Phase ID is not valid: {}".format(phase_id))

                    phase_counter[phase_id] += 1

        # Calculate fractions using only filled cells as total
        if filled_cells == 0:
            raise ValueError("No filled cells in SSRVE - cannot calculate phase fractions")

        phases_fractions = {}
        for phase in ssrve.phases:  # Iterate through PhaseModel objects
            count = phase_counter.get(phase.id, 0)  # Use .get() to handle missing keys
            phases_fractions[phase.name] = float(count / filled_cells)

        return DictResult(values=phases_fractions)
