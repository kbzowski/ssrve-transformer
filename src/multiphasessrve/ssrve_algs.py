import math
from typing import Tuple, Dict, List

import numpy as np

from numba import jit, float64  # type: ignore
from numpy.typing import NDArray


@jit(nopython=True)
def calculate_compatibility_difference(value1: float64, value2: float64) -> float64:
    """Calculate the difference between two compatibility values.

    Args:
        value1: First compatibility value (0-1 range)
        value2: Second compatibility value (0-1 range)

    Returns:
        Absolute difference between compatibility values (0-1 range)
    """
    return abs(float64(value1) - float64(value2))


@jit(nopython=True)
def init_arrays(rows: int, cols: int, starting_points: NDArray) -> NDArray[np.int32]:
    """Initialize the grid array with starting points.
    Only initializes grain IDs since properties will be stored in dictionary.

    Raises:
        ValueError: If no valid points remain after filtering out-of-bounds points
    """
    arr = np.zeros((rows, cols), dtype=np.int32)  # Only grain IDs
    valid_points = 0

    for i in range(len(starting_points)):
        x = int(starting_points[i, 0])
        y = int(starting_points[i, 1])
        nucleation_ratio = starting_points[i, 5]

        # Skip points outside grid bounds
        if x < 0 or x >= rows or y < 0 or y >= cols:
            continue

        # Only initialize points with nucleation_ratio = 0 (nucleate immediately)
        if nucleation_ratio == 0.0 and arr[x, y] == 0:
            arr[x, y] = i + 1
            valid_points += 1

    return arr


@jit(nopython=True)
def handle_periodic_boundaries(
    arr: NDArray[np.int32],
    rows: int,
    cols: int,
) -> NDArray[np.int32]:
    """Handle periodic boundaries for grain ID array.

    Copies grain IDs across boundaries only when one side is empty.
    Does NOT overwrite existing grains - preserves distinct grains from different seeds.
    """
    result_arr = np.copy(arr)

    # Handle vertical boundaries (top-bottom)
    for y in range(cols):
        top_val = result_arr[0, y]
        bottom_val = result_arr[rows - 1, y]
        if top_val and not bottom_val:
            result_arr[rows - 1, y] = top_val
        elif bottom_val and not top_val:
            result_arr[0, y] = bottom_val
        # else: both have values or both empty to no copy (distinct grains)

    # Handle horizontal boundaries (left-right)
    for x in range(rows):
        left_val = result_arr[x, 0]
        right_val = result_arr[x, cols - 1]
        if left_val and not right_val:
            result_arr[x, cols - 1] = left_val
        elif right_val and not left_val:
            result_arr[x, 0] = right_val
        # else: both have values or both empty to no copy (distinct grains)

    return result_arr


@jit(nopython=True)
def apply_nucleation(
    arr: NDArray[np.int32],
    rows: int,
    cols: int,
    fill_ratio: float,
    starting_points: NDArray,
    nucleated: NDArray,
) -> NDArray[np.int32]:
    """Apply nucleation based on fill ratio.

    Args:
        arr: Current state array
        rows: Number of rows
        cols: Number of columns
        fill_ratio: Current fill ratio (0-1) of the grid
        starting_points: All nucleation points and their properties
        nucleated: Boolean array tracking which grains have already nucleated

    Returns:
        Updated array with new nucleation points
    """
    next_arr = np.copy(arr)

    for i in range(len(starting_points)):
        # Skip if already nucleated
        if nucleated[i]:
            continue

        x = int(starting_points[i, 0])
        y = int(starting_points[i, 1])
        nucleation_ratio = starting_points[i, 5]

        # Skip points outside grid bounds
        if x < 0 or x >= rows or y < 0 or y >= cols:
            continue

        # Check if this point should nucleate based on fill ratio
        if fill_ratio >= nucleation_ratio and next_arr[x, y] == 0:
            next_arr[x, y] = i + 1
            nucleated[i] = True

    return next_arr


@jit(nopython=True)
def simple_grow_step(
    arr: NDArray[np.int32],
    rows: int,
    cols: int,
    evolution: int,
    starting_points: NDArray,
    nucleated: NDArray,
) -> NDArray[np.int32]:
    """Perform one growth step with fill-ratio-based nucleation.

    Args:
        arr: Current state array
        rows: Number of rows
        cols: Number of columns
        evolution: Current iteration number (used for diagonal alternation)
        starting_points: All nucleation points and their properties
        nucleated: Boolean array tracking which grains have nucleated

    Returns:
        Updated array after one growth step
    """
    # Calculate current fill ratio
    total_pixels = rows * cols
    filled_pixels = np.count_nonzero(arr)
    fill_ratio = float(filled_pixels) / float(total_pixels)

    # First, apply nucleation based on fill ratio
    next_arr = apply_nucleation(arr, rows, cols, fill_ratio, starting_points, nucleated)

    # Then perform growth for existing grains
    for x in range(rows):
        xm1 = x - 1
        xp1 = x + 1
        for y in range(cols):
            if arr[x, y] != 0:
                val = arr[x, y]
                ym1 = y - 1
                yp1 = y + 1

                # Simple growth without merging
                if ym1 >= 0 and next_arr[x, ym1] == 0:
                    next_arr[x, ym1] = val

                if yp1 < cols and next_arr[x, yp1] == 0:
                    next_arr[x, yp1] = val

                if xm1 >= 0 and next_arr[xm1, y] == 0:
                    next_arr[xm1, y] = val

                if xp1 < rows and next_arr[xp1, y] == 0:
                    next_arr[xp1, y] = val

                # Diagonal neighbors
                if evolution & 1 == 0:
                    if xp1 < rows and ym1 >= 0 and next_arr[xp1, ym1] == 0:
                        next_arr[xp1, ym1] = val
                    if xm1 >= 0 and yp1 < cols and next_arr[xm1, yp1] == 0:
                        next_arr[xm1, yp1] = val
                else:
                    if xm1 >= 0 and ym1 >= 0 and next_arr[xm1, ym1] == 0:
                        next_arr[xm1, ym1] = val
                    if xp1 < rows and yp1 < cols and next_arr[xp1, yp1] == 0:
                        next_arr[xp1, yp1] = val

    return handle_periodic_boundaries(next_arr, rows, cols)


@jit(nopython=True)
def detect_boundaries(arr: NDArray[np.int32], rows: int, cols: int) -> NDArray[np.bool_]:
    """Detect grain boundaries in the current state.

    A boundary is defined as any cell that has at least one neighbor
    (including diagonals) belonging to a different grain.

    Uses periodic boundary conditions when checking neighbors at grid edges.

    Args:
        arr: Current state array where non-zero values indicate grain IDs
        rows: Number of rows in the grid
        cols: Number of columns in the grid

    Returns:
        Boolean array where True indicates boundary cells
    """
    boundaries = np.zeros((rows, cols), dtype=np.bool_)

    for x in range(rows):
        xm1 = (x - 1) % rows
        xp1 = (x + 1) % rows
        for y in range(cols):
            if arr[x, y]:
                current = arr[x, y]
                ym1 = (y - 1) % cols
                yp1 = (y + 1) % cols

                # Check all neighbors using optimized short-circuit evaluation
                if (
                    (arr[x, ym1] and arr[x, ym1] != current)
                    or (arr[xp1, y] and arr[xp1, y] != current)
                    or (arr[xm1, y] and arr[xm1, y] != current)
                    or (arr[x, yp1] and arr[x, yp1] != current)
                    or (arr[xm1, ym1] and arr[xm1, ym1] != current)
                    or (arr[xm1, yp1] and arr[xm1, yp1] != current)
                    or (arr[xp1, ym1] and arr[xp1, ym1] != current)
                    or (arr[xp1, yp1] and arr[xp1, yp1] != current)
                ):
                    boundaries[x, y] = True

    return boundaries


@jit(nopython=True)
def count_grain_cells(arr: NDArray[np.int32], grain_id: int) -> int:
    """Count number of cells belonging to a grain (optimized with Numba)."""
    return np.sum(arr == grain_id)


@jit(nopython=True)
def update_grain_id(arr: NDArray[np.int32], old_id: int, new_id: int) -> None:
    """Update all cells of one grain to new ID (optimized with Numba)."""
    rows, cols = arr.shape
    for x in range(rows):
        for y in range(cols):
            if arr[x, y] == old_id:
                arr[x, y] = new_id


@jit(nopython=True)
def get_neighbor_ids(
    arr: NDArray[np.int32],
    x: int,
    y: int,
    rows: int,
    cols: int,
) -> Tuple[int, int, int, int]:
    """Get IDs of direct neighbors (optimized with Numba)."""
    return (
        arr[(x - 1) % rows, y],  # left
        arr[(x + 1) % rows, y],  # right
        arr[x, (y - 1) % cols],  # top
        arr[x, (y + 1) % cols],  # bottom
    )


@jit(nopython=True)
def is_grain_infinite(
    arr: NDArray[np.int32],
    grain_id: int,
    rows: int,
    cols: int,
) -> bool:
    """Check if grain is infinite (has continuous path through opposite boundaries).

    Uses BFS with 8-connectivity to check if grain connects:
    - Top to Bottom, OR
    - Left to Right

    Args:
        arr: Grain ID array
        grain_id: ID of grain to check
        rows: Number of rows
        cols: Number of columns

    Returns:
        True if grain is infinite, False otherwise
    """
    # Check vertical connectivity (top to bottom)
    if _has_vertical_path(arr, grain_id, rows, cols):
        return True

    # Check horizontal connectivity (left to right)
    if _has_horizontal_path(arr, grain_id, rows, cols):
        return True

    return False


@jit(nopython=True)
def _has_vertical_path(
    arr: NDArray[np.int32],
    grain_id: int,
    rows: int,
    cols: int,
) -> bool:
    """Check if grain has continuous path from top to bottom edge."""
    # Find starting points on top edge
    top_pixels = np.zeros(cols, dtype=np.int32)
    num_top = 0
    for y in range(cols):
        if arr[0, y] == grain_id:
            top_pixels[num_top] = y
            num_top += 1

    if num_top == 0:
        return False

    # BFS from top edge to bottom edge
    visited = np.zeros((rows, cols), dtype=np.bool_)
    queue_x = np.zeros(rows * cols, dtype=np.int32)
    queue_y = np.zeros(rows * cols, dtype=np.int32)
    queue_start = 0
    queue_end = 0

    # Initialize queue with top edge pixels
    for i in range(num_top):
        y = top_pixels[i]
        queue_x[queue_end] = 0
        queue_y[queue_end] = y
        queue_end += 1
        visited[0, y] = True

    # 8-connectivity offsets
    dx = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
    dy = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)

    # BFS
    while queue_start < queue_end:
        x = queue_x[queue_start]
        y = queue_y[queue_start]
        queue_start += 1

        # Check if we reached bottom edge
        if x == rows - 1:
            return True

        # Explore neighbors (8-connectivity, no periodic wrap for internal path)
        for i in range(8):
            nx = x + dx[i]
            ny = y + dy[i]

            # Bounds check (no periodic wrap - checking internal connectivity)
            if 0 <= nx < rows and 0 <= ny < cols:
                if not visited[nx, ny] and arr[nx, ny] == grain_id:
                    visited[nx, ny] = True
                    queue_x[queue_end] = nx
                    queue_y[queue_end] = ny
                    queue_end += 1

    return False


@jit(nopython=True)
def _has_horizontal_path(
    arr: NDArray[np.int32],
    grain_id: int,
    rows: int,
    cols: int,
) -> bool:
    """Check if grain has continuous path from left to right edge."""
    # Find starting points on left edge
    left_pixels = np.zeros(rows, dtype=np.int32)
    num_left = 0
    for x in range(rows):
        if arr[x, 0] == grain_id:
            left_pixels[num_left] = x
            num_left += 1

    if num_left == 0:
        return False

    # BFS from left edge to right edge
    visited = np.zeros((rows, cols), dtype=np.bool_)
    queue_x = np.zeros(rows * cols, dtype=np.int32)
    queue_y = np.zeros(rows * cols, dtype=np.int32)
    queue_start = 0
    queue_end = 0

    # Initialize queue with left edge pixels
    for i in range(num_left):
        x = left_pixels[i]
        queue_x[queue_end] = x
        queue_y[queue_end] = 0
        queue_end += 1
        visited[x, 0] = True

    # 8-connectivity offsets
    dx = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
    dy = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)

    # BFS
    while queue_start < queue_end:
        x = queue_x[queue_start]
        y = queue_y[queue_start]
        queue_start += 1

        # Check if we reached right edge
        if y == cols - 1:
            return True

        # Explore neighbors (8-connectivity, no periodic wrap for internal path)
        for i in range(8):
            nx = x + dx[i]
            ny = y + dy[i]

            # Bounds check (no periodic wrap - checking internal connectivity)
            if 0 <= nx < rows and 0 <= ny < cols:
                if not visited[nx, ny] and arr[nx, ny] == grain_id:
                    visited[nx, ny] = True
                    queue_x[queue_end] = nx
                    queue_y[queue_end] = ny
                    queue_end += 1

    return False


@jit(nopython=True)
def smooth_step(
    arr: NDArray[np.int32],
    rows: int,
    cols: int,
) -> Tuple[NDArray[np.int32], bool]:
    """Smooth all grain boundaries by removing small irregularities.

    Smooths boundaries between all grains regardless of their phase.
    A cell changes its grain ID if it has 5 or more neighbors of a different grain ID.
    Uses periodic boundary conditions when checking neighbors at grid edges.

    Args:
        arr: Current grain ID array
        rows: Number of rows in the grid
        cols: Number of columns in the grid

    Returns:
        Tuple of (smoothed array, whether changes were made)
    """
    result_arr = np.copy(arr)
    changes_made = False
    max_id = arr.max()

    # Pre-allocate neighbor arrays for efficiency
    neighbor_counts = np.zeros(max_id + 1, dtype=np.int32)

    # Perform smoothing
    for x in range(rows):
        for y in range(cols):
            if arr[x, y] == 0:
                continue

            # Reset neighbor counts for this cell
            neighbor_counts.fill(0)

            # Count neighbors efficiently
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx) % rows
                    ny = (y + dy) % cols
                    neighbor_grain = arr[nx, ny]
                    if neighbor_grain > 0 and neighbor_grain != arr[x, y]:
                        neighbor_counts[neighbor_grain] += 1

            # Find maximum neighbor count
            max_count = 0
            max_grain = 0
            for grain_id in range(1, len(neighbor_counts)):
                if neighbor_counts[grain_id] > max_count:
                    max_count = neighbor_counts[grain_id]
                    max_grain = grain_id

            # Change if there's a dominant different grain - no phase check!
            if max_count >= 5 and max_grain != arr[x, y]:
                result_arr[x, y] = max_grain
                changes_made = True

    return result_arr, changes_made


@jit(nopython=True)
def count_neighbor_frequencies(
    arr: NDArray[np.int32],
    x: int,
    y: int,
    rows: int,
    cols: int,
) -> Tuple[int, int]:
    """Count frequencies of neighboring grain IDs and return the most frequent.

    Returns:
        Tuple[int, int]: (most_frequent_id, frequency)
    """
    # Pre-allocate frequency dictionary (as array since Numba doesn't support dict)
    # Use a large enough array to store possible grain IDs
    max_id = arr.max() + 1
    frequencies = np.zeros(max_id, dtype=np.int32)
    current_id = arr[x, y]

    # Count neighbors
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = (x + dx) % rows
            ny = (y + dy) % cols
            neighbor_id = arr[nx, ny]
            if neighbor_id != 0 and neighbor_id != current_id:
                frequencies[neighbor_id] += 1

    # Find most frequent
    max_freq = 0
    max_id = 0
    for i in range(len(frequencies)):
        if frequencies[i] > max_freq:
            max_freq = frequencies[i]
            max_id = i

    return max_id, max_freq
