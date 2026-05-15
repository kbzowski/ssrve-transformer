from typing import Dict, Optional
import numpy as np
from multiphasessrve.ssrve import Ssrve
from multiphasessrve.analysis.registry import registry
from multiphasessrve.types.dict_result import DictResult
from multiphasessrve.ssrve_algs import is_grain_infinite


@registry.register("grain_size")
class GrainSizeAnalyzer:
    """Grain size analysis for SSRVE."""

    @staticmethod
    def _calculate_equivalent_diameter(area: float) -> float:
        """Calculate equivalent circle diameter from area."""
        return 2 * np.sqrt(area / np.pi)

    @staticmethod
    def analyze(ssrve: Ssrve, **kwargs) -> DictResult:
        """Calculate mean grain diameter using equivalent circle method.

        Args:
            ssrve: SSRVE instance
            **kwargs:
                phase_id (int): Calculate only for specific phase
                real_size (Optional[float]): Size of the SSRVE in micrometers (μm)

        Returns:
            ScalarResult with mean grain diameter by default
            If detailed=True, returns DictResult with full statistics
        """
        phase_id = kwargs.get("phase_id")
        real_size = kwargs.get("real_size")

        if phase_id is None:
            raise ValueError("phase_id parameter is required for grain size analysis")

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
                if area_pixels < 10:
                    continue

                # Skip infinite grains (they have infinite physical size)
                if is_grain_infinite(grain_arr, grain_id, rows, cols):
                    continue

                area = area_pixels * (pixel_size**2)
                diameter = GrainSizeAnalyzer._calculate_equivalent_diameter(area)
                diameters.append(diameter)

        units = {
            "min_diameter": "μm" if real_size is not None else "pixels",
            "max_diameter": "μm" if real_size is not None else "pixels",
            "mean_diameter": "μm" if real_size is not None else "pixels",
            "median_diameter": "μm" if real_size is not None else "pixels",
            "diameters": "μm" if real_size is not None else "pixels",
        }

        if not diameters:
            return DictResult(values={"min_diameter": 0, "max_diameter": 0}, units=units)

        # Store detailed statistics for debugging. Do not remove!
        stats = {
            "min_diameter": float(np.min(diameters)),
            "max_diameter": float(np.max(diameters)),
            "mean_diameter": float(np.mean(diameters)),
            "median_diameter": float(np.median(diameters)),
            "count": len(diameters),
            "diameters": np.sort(diameters).tolist(),
        }

        return DictResult(values=stats, units=units)
