from typing import Dict, Optional, Union, List
import numpy as np
from multiphasessrve.ssrve import Ssrve
from multiphasessrve.analysis.registry import registry
from multiphasessrve.types.histogram_result import HistogramResult
from multiphasessrve.units import MICRO_METER
from multiphasessrve.ssrve_algs import is_grain_infinite


@registry.register("grain_size_histogram")
class GrainSizeHistogramAnalyzer:
    """Grain size histogram analyzer for SSRVE."""

    @staticmethod
    def _calculate_equivalent_diameter(area: float) -> float:
        """Calculate equivalent circle diameter from area."""
        return 2 * np.sqrt(area / np.pi)

    @staticmethod
    def analyze(ssrve: Ssrve, **kwargs) -> HistogramResult:
        """Calculate grain diameter histogram using equivalent circle method.

        Args:
            ssrve: SSRVE instance
            **kwargs:
                phase_id (int): Calculate only for specific phase
                real_size (Optional[float]): Size of the SSRVE in micrometers (μm)
                bins (Union[int, List[float]]): Number of bins or bin edges for histogram (default: 10)
                min_grain_area (int): Minimum grain area in pixels to be included (default: 10)
                max_value (Optional[float]): Maximum value to include in histogram

        Returns:
            HistogramResult with grain diameter distribution
        """
        phase_id = kwargs.get("phase_id")
        real_size = kwargs.get("real_size")
        bins = kwargs.get("bins", 10)
        min_grain_area = kwargs.get("min_grain_area", 10)
        max_value = kwargs.get("max_value")

        if phase_id is None:
            raise ValueError("phase_id parameter is required for grain size histogram analysis")

        # Get basic arrays
        grain_arr = ssrve.arr
        rows, cols = ssrve.get_rows(), ssrve.get_cols()

        # Calculate pixel to μm conversion factor
        pixel_size = real_size / rows if real_size is not None else 1.0

        # Get grain phase mapping if needed
        grain_phase_map = ssrve.get_grain_phase_map() if phase_id is not None else None

        # Calculate grain diameters
        diameters = []

        for grain_id in np.unique(grain_arr):
            if grain_id > 0:  # Skip unfilled cells
                if phase_id is not None and grain_phase_map.get(grain_id) != phase_id:
                    continue

                area_pixels = np.sum(grain_arr == grain_id)

                # Skip grains with small area
                if area_pixels < min_grain_area:
                    continue

                # Skip infinite grains (they have infinite physical size)
                if is_grain_infinite(grain_arr, grain_id, rows, cols):
                    continue

                area = area_pixels * (pixel_size**2)
                diameter = GrainSizeHistogramAnalyzer._calculate_equivalent_diameter(area)

                # If max_value is specified and diameter exceeds it, cap the diameter at max_value
                if max_value is not None and diameter > max_value:
                    diameter = max_value

                diameters.append(diameter)

        # Unit label for the histogram
        unit = MICRO_METER if real_size is not None else "pixels"

        # Handle edge case with no diameters
        if not diameters:
            # Create an empty histogram with a single bin
            return HistogramResult(
                bin_edges=np.array([0.0, 1.0]),
                counts=np.array([0.0]),
                name=f"grain_size_hist_phase_{phase_id}",
                unit=unit,
            )

        histogram = HistogramResult.from_data(
            data=np.array(diameters), bins=bins, name=f"grain_size_hist_phase_{phase_id}", unit=unit, density=True
        )

        debug_stats = GrainSizeHistogramAnalyzer.get_statistics(histogram)

        return histogram

    @staticmethod
    def get_statistics(histogram: HistogramResult) -> Dict[str, float]:
        """Extract common grain size statistics from histogram.

        Args:
            histogram: HistogramResult from grain size analysis

        Returns:
            Dictionary with grain size statistics (min, max, mean, median diameter)
        """
        if np.sum(histogram.counts) == 0:
            return {"min_diameter": 0.0, "max_diameter": 0.0, "mean_diameter": 0.0, "median_diameter": 0.0, "count": 0}

        # Calculate mean (expected value)
        mean_diameter = histogram.expected_value()

        # Find min/max diameters (first/last bins with non-zero counts)
        min_idx = next((i for i, count in enumerate(histogram.counts) if count > 0), 0)
        max_idx = next((i for i in range(len(histogram.counts) - 1, -1, -1) if histogram.counts[i] > 0), 0)

        min_diameter = histogram.bin_edges[min_idx]
        max_diameter = histogram.bin_edges[max_idx + 1]

        # Calculate median diameter
        total_count = np.sum(histogram.counts)
        cum_counts = np.cumsum(histogram.counts)
        median_idx = np.searchsorted(cum_counts, total_count / 2)

        if median_idx >= len(histogram.bin_centers):
            median_diameter = histogram.bin_centers[-1]
        else:
            median_diameter = histogram.bin_centers[median_idx]

        return {
            "min_diameter": float(min_diameter),
            "max_diameter": float(max_diameter),
            "mean_diameter": float(mean_diameter),
            "median_diameter": float(median_diameter),
            "count": int(total_count),
        }
