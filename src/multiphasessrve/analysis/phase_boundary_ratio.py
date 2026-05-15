from typing import Tuple

import numpy as np
from numba import jit
from numpy.typing import NDArray

from multiphasessrve.analysis.registry import registry
from multiphasessrve.ssrve import Ssrve
from multiphasessrve.types.scalar_result import ScalarResult


@jit(nopython=True)
def calculate_phase_boundary_ratio(
    phase_array: NDArray[np.int32], rows: int, cols: int, phase_id1: int, phase_id2: int
) -> Tuple[int, int]:
    """Calculate the ratio of boundaries between two phases to all boundaries.

    Args:
        phase_array: Array where each cell contains its phase ID
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        phase_id1: First phase ID
        phase_id2: Second phase ID

    Returns:
        Tuple of (target_phase_boundaries, total_boundaries)
    """
    total_boundaries = 0
    target_phase_boundaries = 0

    # Define directions for checking neighbors (right, down)
    # Using just right and down avoids double-counting boundaries
    for x in range(rows):
        for y in range(cols):
            current_phase = phase_array[x, y]

            # Skip empty cells
            if current_phase == 0:
                continue

            # Check right neighbor
            if y + 1 < cols:
                neighbor_phase = phase_array[x, y + 1]

                # Skip neighbors that are empty
                if neighbor_phase != 0:
                    # If phases are different, we have a boundary
                    if current_phase != neighbor_phase:
                        total_boundaries += 1

                        # Check if this boundary is between our target phases
                        if (current_phase == phase_id1 and neighbor_phase == phase_id2) or (
                            current_phase == phase_id2 and neighbor_phase == phase_id1
                        ):
                            target_phase_boundaries += 1

            # Check down neighbor
            if x + 1 < rows:
                neighbor_phase = phase_array[x + 1, y]

                # Skip neighbors that are empty
                if neighbor_phase != 0:
                    # If phases are different, we have a boundary
                    if current_phase != neighbor_phase:
                        total_boundaries += 1

                        # Check if this boundary is between our target phases
                        if (current_phase == phase_id1 and neighbor_phase == phase_id2) or (
                            current_phase == phase_id2 and neighbor_phase == phase_id1
                        ):
                            target_phase_boundaries += 1

    return target_phase_boundaries, total_boundaries


@registry.register("phase_boundary_ratio")
class PhaseBoundaryRatioAnalyzer:
    """Analyzer for calculating the ratio of grain boundaries between specified phases to all boundaries."""

    @staticmethod
    def analyze(ssrve: Ssrve, **kwargs) -> ScalarResult:
        """Calculate the ratio of boundaries between two specific phases to all boundaries.

        Boundaries are defined as cells that have a neighbor with a different phase.

        Args:
            ssrve: SSRVE instance
            **kwargs: Additional parameters, must include:
                phase_id1: First phase ID (must be >= 1)
                phase_id2: Second phase ID (must be >= 1)

        Returns:
            ScalarResult containing the ratio of boundaries between specified phases to all boundaries
        """
        # Extract phase IDs from kwargs
        if "phase_id1" not in kwargs or "phase_id2" not in kwargs:
            raise ValueError("Both 'phase_id1' and 'phase_id2' parameters are required")

        phase_id1 = kwargs["phase_id1"]
        phase_id2 = kwargs["phase_id2"]

        # Validate phase IDs
        if phase_id1 < 1 or phase_id2 < 1:
            raise ValueError(f"Invalid phase IDs: {phase_id1}, {phase_id2}. Phase IDs must be >= 1.")

        if phase_id1 not in [phase.id for phase in ssrve.phases] or phase_id2 not in [
            phase.id for phase in ssrve.phases
        ]:
            raise ValueError(f"Phase IDs {phase_id1} or {phase_id2} not found in the SSRVE's phases.")

        # Get the phase array where each cell contains its phase ID
        phase_array = ssrve.get_phase_array()
        rows, cols = ssrve.get_rows(), ssrve.get_cols()

        # Use the Numba-optimized function to calculate the ratio
        target_phase_boundaries, total_boundaries = calculate_phase_boundary_ratio(
            phase_array, rows, cols, phase_id1, phase_id2
        )

        # Calculate the ratio
        ratio = 0.0
        if total_boundaries > 0:
            ratio = float(target_phase_boundaries) / total_boundaries

        return ScalarResult(value=ratio, name=f"boundary_ratio_{phase_id1}_{phase_id2}")
