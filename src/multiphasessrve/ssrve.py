from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

from multiphasessrve.grain_properties import GrainProperties
from multiphasessrve.ssrve_algs import (
    init_arrays,
    detect_boundaries,
    simple_grow_step,
    update_grain_id,
    get_neighbor_ids,
    count_neighbor_frequencies,
    smooth_step,
)
from multiphasessrve.task import PhasesModel

# Type aliases for clarity
StartingPoint = Tuple[
    int, int, int, float, float, float
]  # x, y, phase, compatibility, compatibility_threshold, nucleation_ratio


class Ssrve:
    """Main SSRVE (Statistically Similar Representative Volume Element) class.

    Phase numbering convention:
    - Empty cells (grain_id = 0): No phase
    - Valid phases: Numbered from 1 to N (where N is the number of phases)
    - There is no "phase 0" - all valid phases start from 1
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        starting_points: List[Tuple[StartingPoint]],
        phases: PhasesModel,
    ):
        """Initialize the SSRVE grid.

        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            starting_points: List of (x, y, phase, orientation) tuples where:
                If not normalized:
                    - x, y: coordinates (will be rounded to integers)
                    - phase: continuous value in [0, max_phases] range from optimization
                    - compatibility: value in [0,1] range
                    - merge threshold: value in [0,1] range
                    - nucleation time: value in [0, max_iterations] range, rounded to integer
            phases: Phase definitions
        """
        self.rows = rows
        self.cols = cols
        self.phases = phases
        self.evolution = 0

        # Convert starting points to numpy array for Numba functions
        self.starting_points_array = np.array(starting_points, dtype=np.float64)

        # Track which grains have nucleated
        self.nucleated = np.zeros(len(starting_points), dtype=np.bool_)

        # Initialize grain ID array using Numba-optimized function
        self.arr = init_arrays(rows, cols, self.starting_points_array)

        # Mark grains with nucleation_ratio=0 as already nucleated
        for i, (x, y, _, _, _, nucleation_ratio) in enumerate(starting_points):
            if nucleation_ratio == 0.0 and 0 <= x < rows and 0 <= y < cols:
                self.nucleated[i] = True

        # Store grain properties in dictionary
        self.grain_properties: Dict[int, GrainProperties] = {}
        for i, (x, y, phase_id, compatibility, threshold, nucleation_ratio) in enumerate(starting_points, 1):
            if x < 0 or x >= rows or y < 0 or y >= cols:
                continue

            if phase_id < 1:
                raise ValueError(f"Invalid phase ID {phase_id} detected. Phase IDs must be >= 1.")

            if not 0.0 <= nucleation_ratio <= 1.0:
                raise ValueError(f"Invalid nucleation ratio {nucleation_ratio} detected. Must be in [0, 1].")

            self.grain_properties[i] = GrainProperties(
                phase_id=phase_id,
                compatibility=float(compatibility),
                threshold=float(threshold),
                nucleation_ratio=float(nucleation_ratio),
            )

    def can_grains_merge(self, grain_id1: int, grain_id2: int) -> bool:
        """Check if two grains can merge based on their properties."""
        if grain_id1 not in self.grain_properties or grain_id2 not in self.grain_properties:
            return False

        grain1 = self.grain_properties[grain_id1]
        grain2 = self.grain_properties[grain_id2]

        # Must be same phase
        if grain1.phase_id != grain2.phase_id:
            return False

        # Check compatibility difference against minimum threshold
        compatibility_diff = abs(grain1.compatibility - grain2.compatibility)
        effective_threshold = min(grain1.threshold, grain2.threshold)

        return compatibility_diff <= effective_threshold

    def merge_grains(self, grain_id1: int, grain_id2: int) -> None:
        """Merge two grains, updating both array and properties."""
        grain1 = self.grain_properties[grain_id1]
        grain2 = self.grain_properties[grain_id2]

        # Calculate new properties
        new_compatibility = (grain1.compatibility + grain2.compatibility) / 2.0
        new_threshold = (grain1.threshold + grain2.threshold) / 2.0  # Use average threshold
        new_nucleation_ratio = min(grain1.nucleation_ratio, grain2.nucleation_ratio)  # Earlier nucleation wins

        # Update grain properties
        self.grain_properties[grain_id1] = GrainProperties(
            phase_id=grain1.phase_id,
            compatibility=new_compatibility,
            threshold=new_threshold,
            nucleation_ratio=new_nucleation_ratio,
        )

        # Update grain IDs in array (using Numba-optimized function)
        update_grain_id(self.arr, grain_id2, grain_id1)

        # Remove absorbed grain
        del self.grain_properties[grain_id2]

    def to_img(self, show_boundaries: bool = True) -> Image.Image:
        """Convert current state to PIL Image.

        Shows grain boundaries between different grain IDs with single-pixel boundaries.

        Args:
            show_boundaries: Whether to show grain boundaries in black

        Returns:
            PIL Image representation of the current state
        """
        img = Image.new("RGB", (self.rows, self.cols), color="white")

        # Fill all pixels with their phase colors
        for x in range(self.rows):
            for y in range(self.cols):
                grain_id = self.arr[x, y]
                if grain_id == 0:
                    color = (0, 0, 0)  # Unfilled cells
                else:
                    props = self.grain_properties[grain_id]
                    color = self.phases.colors_list[props.phase_id - 1]
                img.putpixel((x, y), color)

        if show_boundaries:
            # Add boundaries between different grains
            for x in range(self.rows):
                for y in range(self.cols):
                    current_id = self.arr[x, y]
                    if current_id == 0:
                        continue

                    # Check right and bottom neighbors
                    if x < self.rows - 1 and self.arr[x + 1, y] != current_id:
                        img.putpixel((x, y), (0, 0, 0))
                    if y < self.cols - 1 and self.arr[x, y + 1] != current_id:
                        img.putpixel((x, y), (0, 0, 0))

        return img

    def try_merge_all(self) -> None:
        """Try to merge all neighboring grains that meet criteria."""
        while True:
            merges_this_round = 0

            # Check each cell for potential merges
            for x in range(self.rows):
                for y in range(self.cols):
                    current_id = self.arr[x, y]
                    if current_id == 0:
                        continue

                    # Get neighbor IDs using Numba-optimized function
                    neighbor_ids = get_neighbor_ids(self.arr, x, y, self.rows, self.cols)

                    # Try to merge with each neighbor
                    for neighbor_id in neighbor_ids:
                        if neighbor_id != 0 and neighbor_id != current_id and neighbor_id in self.grain_properties:

                            if self.can_grains_merge(current_id, neighbor_id):
                                self.merge_grains(current_id, neighbor_id)
                                merges_this_round += 1
                                break  # Only one merge per grain per round

            if merges_this_round == 0:
                break

    def save_image(self, filename: str, show_boundaries: bool = True) -> None:
        """Save current state as image.

        Args:
            filename: Path where to save the image
            show_boundaries: Whether to show grain boundaries in black
        """
        img = self.to_img(show_boundaries=show_boundaries)
        img.save(filename)

    def create_ssrve(self, max_iterations: float = float("inf"), enable_merging: bool = True) -> None:
        """Create the complete SSRVE structure.

        Args:
            max_iterations: Maximum number of growth steps to perform
            enable_merging: Whether to merge grains based on compatibility after growth
        """
        # Phase 1: Grow all grains without merging
        stagnation_counter = 0
        max_stagnation = 10  # Stop if no change for 10 iterations

        while self.can_grow() and self.evolution < max_iterations:
            # Check if growth actually happened
            filled_before = np.count_nonzero(self.arr)
            self.single_grow()
            filled_after = np.count_nonzero(self.arr)

            # Detect stagnation (no growth)
            if filled_after == filled_before:
                stagnation_counter += 1
                if stagnation_counter >= max_stagnation:
                    # No growth for max_stagnation iterations - stop
                    break
            else:
                stagnation_counter = 0

        # Phase 2: Merge grains based on compatibility (if enabled)
        if enable_merging:
            self.try_merge_all()

    def get_grain_phase_map(self) -> Dict[int, int]:
        """Get mapping from grain IDs to phase numbers.

        Returns:
            Dict[int, int]: Dictionary mapping grain IDs to their phases
        """
        return {grain_id: props.phase_id for grain_id, props in self.grain_properties.items()}

    def apply_smoothing(self, iterations: int = 3) -> None:
        """Smooth grain boundaries by removing small irregularities."""
        if iterations < 1:
            return

        # Create grain to phase mapping using the maximum of both array and properties
        max_id = (
            max(self.arr.max(), max(self.grain_properties.keys(), default=0)) + 1
        )  # +1 because we need space for index max_id itself

        phases = np.zeros(max_id, dtype=np.int32)
        for grain_id, props in self.grain_properties.items():
            phases[grain_id] = props.phase_id

        # Perform smoothing iterations
        for iteration in range(iterations):
            new_arr, changes_made = smooth_step(self.arr, self.rows, self.cols)
            if changes_made:
                self.arr = new_arr
            else:
                break

        # Update remaining grain properties
        existing_grains = set(np.unique(self.arr))
        existing_grains.discard(0)
        self.grain_properties = {
            grain_id: props for grain_id, props in self.grain_properties.items() if grain_id in existing_grains
        }

    def can_grow(self) -> bool:
        """Check if growth can continue (if there are any unfilled cells)."""
        return np.any(self.arr == 0)

    def single_grow(self) -> None:
        """Perform a single growth step using Numba-optimized function."""
        self.arr = simple_grow_step(self.arr, self.rows, self.cols, self.evolution, self.starting_points_array, self.nucleated)
        self.evolution += 1

    def get_boundaries(self) -> np.ndarray:
        """Get grain boundaries from current state."""
        return detect_boundaries(self.arr, self.rows, self.cols)

    def get_rows(self) -> int:
        """Get number of rows in the grid."""
        return self.rows

    def get_cols(self) -> int:
        """Get number of columns in the grid."""
        return self.cols

    def get_grains_array(self) -> np.ndarray:
        """Get grains array."""
        return self.arr

    def get_phase_array(self) -> np.ndarray:
        """Get phases array."""
        phase_arr = np.zeros_like(self.arr)
        for grain_id, props in self.grain_properties.items():
            phase_arr[self.arr == grain_id] = props.phase_id

        return phase_arr

    def clear(self) -> "Ssrve":
        """Reinitialize the SSRVE grid. Removes all growth steps."""
        self.evolution = 0

        # Reset nucleation tracking
        self.nucleated = np.zeros(len(self.starting_points_array), dtype=np.bool_)

        # Reinitialize array with starting points that have nucleation_ratio = 0
        self.arr = init_arrays(self.rows, self.cols, self.starting_points_array)

        # Mark grains with nucleation_ratio=0 as already nucleated
        for i, (x, y, _, _, _, nucleation_ratio) in enumerate(self.starting_points_array):
            if nucleation_ratio == 0.0 and 0 <= x < self.rows and 0 <= y < self.cols:
                self.nucleated[i] = True

        # Reinitialize grain properties - for all points
        self.grain_properties = {}
        for i, (_, _, phase, compatibility, threshold, nucleation_ratio) in enumerate(self.starting_points_array, 1):
            self.grain_properties[i] = GrainProperties(
                phase_id=int(phase),
                compatibility=float(compatibility),
                threshold=float(threshold),
                nucleation_ratio=float(nucleation_ratio),
            )

        return self

    def renumber_grains(self) -> "Ssrve":
        """Renumber the grain IDs into consecutive integers starting from 1."""
        # Get unique grain IDs (excluding 0 which represents unfilled cells)
        unique_grains = sorted(list(set(self.arr.flat)))  # Get all unique IDs
        if 0 in unique_grains:
            unique_grains.remove(0)  # Remove unfilled cells

        # Create new properties dictionary
        new_properties = {}
        new_arr = np.zeros_like(self.arr)

        # Renumber grains
        for new_id, old_id in enumerate(unique_grains, 1):
            # Update array
            mask = self.arr == old_id
            new_arr[mask] = new_id

            # Transfer properties
            if old_id in self.grain_properties:
                new_properties[new_id] = self.grain_properties[old_id]

        # Update instance variables
        self.arr = new_arr
        self.grain_properties = new_properties

        return self

    def copy_phase_from(self, other: "Ssrve", phase_id: int) -> "Ssrve":
        """Copy grains of specified phase from another SSRVE object.

        Args:
            other: Source SSRVE object to copy from
            phase_id: Phase ID to copy

        Returns:
            Self for method chaining

        Raises:
            ValueError: If SSRVEs have different dimensions or phase_id is invalid
        """
        # Validate inputs
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                f"SSRVEs must have same dimensions. Got {self.rows}x{self.cols} and {other.rows}x{other.cols}"
            )

        if phase_id < 1:
            raise ValueError(f"Invalid phase_id {phase_id}. Phase IDs must be >= 1")

        # Create masks for the phase in both SSRVEs
        self_phase_mask = np.zeros_like(self.arr, dtype=bool)
        other_phase_mask = np.zeros_like(other.arr, dtype=bool)

        # Find grains belonging to the specified phase
        self_phase_grains = []
        for grain_id, props in self.grain_properties.items():
            if props.phase_id == phase_id:
                self_phase_mask |= self.arr == grain_id
                self_phase_grains.append(grain_id)

        other_phase_grains = []
        for grain_id, props in other.grain_properties.items():
            if props.phase_id == phase_id:
                other_phase_mask |= other.arr == grain_id
                other_phase_grains.append(grain_id)

        # Remove properties of grains with specified phase
        for grain_id in self_phase_grains:
            del self.grain_properties[grain_id]

        # Clear cells of specified phase
        self.arr[self_phase_mask] = 0

        # Renumber remaining grains to be consecutive
        self.renumber_grains()

        # Get the next available grain ID
        next_grain_id = max(self.grain_properties.keys(), default=0) + 1

        # Create mapping for new grain IDs
        grain_id_map = {}
        for old_id in other_phase_grains:
            grain_id_map[old_id] = next_grain_id
            next_grain_id += 1

        # Copy grains and their properties
        for old_id, new_id in grain_id_map.items():
            # Copy cells
            mask = other.arr == old_id
            self.arr[mask & self_phase_mask] = new_id

            # Copy properties
            self.grain_properties[new_id] = other.grain_properties[old_id]

        return self

    def __str__(self) -> str:
        """Return string representation of SSRVE with growth status and all starting points.

        Returns:
            str: Formatted string containing evolution status and starting points information
        """
        output = []
        output.append(f"SSRVE Grid ({self.rows}x{self.cols})")

        # Add evolution status
        filled_cells = np.count_nonzero(self.arr)
        total_cells = self.rows * self.cols
        fill_percentage = (filled_cells / total_cells) * 100

        output.append("\nGrowth Status:")
        output.append(f"  Evolution Steps: {self.evolution}")
        output.append(f"  Filled Cells: {filled_cells}/{total_cells} ({fill_percentage:.1f}%)")
        output.append(f"  Active Grains: {len(self.grain_properties)}")

        output.append("\nStarting Points:")

        for i, (x, y, phase_id, compatibility, threshold, nucleation_ratio) in enumerate(self.starting_points_array, 1):
            # Check if this grain actually nucleated by looking for its ID in the array
            did_nucleate = (self.arr == i).any()
            nucleation_status = "[OK]" if did_nucleate else "[--]"

            point_info = [
                f"Point {i}: {nucleation_status}",
                f"  Position: ({x:.1f}, {y:.1f})",
                f"  Phase: {phase_id:.0f}",
                f"  Compatibility: {compatibility:.3f}",
                f"  Threshold: {threshold:.3f}",
                f"  Nucleation Ratio: {nucleation_ratio:.3f}",
            ]
            output.append("\n".join(point_info))

        return "\n".join(output)
