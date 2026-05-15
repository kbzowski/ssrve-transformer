from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union

from multiphasessrve.types.json_serializable_mixin import JSONSerializableMixin


@dataclass
class HistogramResult(JSONSerializableMixin):
    """Container for histogram results."""

    bin_edges: NDArray[np.float64]  # Array of bin edges (length n+1 for n bins)
    counts: NDArray[np.float64]  # Array of counts/frequencies for each bin
    name: Optional[str] = None
    bin_centers: Optional[NDArray[np.float64]] = None  # Precalculated bin centers for convenience
    density: bool = True  # Whether counts represent density (normalized) or absolute counts
    unit: str = ""  # Optional unit for the bin values (x-axis)

    def __post_init__(self):
        """Calculate bin centers if not provided."""
        if self.bin_centers is None:
            # Calculate centers as midpoint between edges
            self.bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2

        # Verify shapes
        if len(self.bin_edges) != len(self.counts) + 1:
            raise ValueError(
                f"bin_edges must have length n+1 where n is length of counts. "
                f"Got {len(self.bin_edges)} edges and {len(self.counts)} counts"
            )

    def expected_value(self) -> float:
        """
        Calculate the expected value (mean) of the histogram.

        The expected value is computed as the weighted average of bin centers,
        where weights are the normalized bin counts.

        Returns:
            float: The expected value of the histogram.

        Note:
            Returns NaN if the histogram has no counts (sum is zero).
        """
        total_counts = np.sum(self.counts)
        if total_counts <= 0:
            return np.nan

        # Calculate weighted average of bin centers
        return np.sum(self.bin_centers * self.counts) / total_counts

    def __str__(self):
        density_str = "density" if self.density else "counts"
        unit_str = f" {self.unit}" if self.unit else ""
        name_str = f": {self.name}" if self.name else ""
        return (
            f"HistogramResult{name_str}\n"
            f"  {len(self.counts)} bins ranging from {self.bin_edges[0]:.6g} to {self.bin_edges[-1]:.6g}{unit_str}\n"
            f"  Total {density_str}: {np.sum(self.counts):.6g}"
        )

    def __repr__(self):
        return (
            f"HistogramResult(bin_edges={repr(self.bin_edges)}, counts={repr(self.counts)}, "
            f"name={repr(self.name)}, density={repr(self.density)}, unit={repr(self.unit)})"
        )

    @classmethod
    def from_data(
        cls,
        data: NDArray[np.float64],
        bins: Union[int, NDArray[np.float64]] = 10,
        density: bool = False,
        name: Optional[str] = None,
        unit: str = "",
    ) -> "HistogramResult":
        """
        Create a HistogramResult directly from data using numpy's histogram function.

        Args:
            data: Array of values to histogram
            bins: Either number of bins or array of bin edges
            density: If True, result is normalized to form a probability density
            name: Optional name for the histogram
            unit: Optional unit for the data values

        Returns:
            HistogramResult instance
        """
        counts, bin_edges = np.histogram(data, bins=bins, density=density)
        return cls(bin_edges=bin_edges, counts=counts, name=name, density=density, unit=unit)

    @classmethod
    def from_histogram_values(
        cls,
        counts: NDArray[np.float64],
        bin_edges: NDArray[np.float64],
        density: bool = True,
        name: Optional[str] = None,
        unit: str = "",
    ) -> "HistogramResult":
        """
        Create a HistogramResult directly from pre-computed histogram values and bin edges.

        Args:
            counts: Array of pre-computed histogram values/frequencies for each bin
            bin_edges: Array of bin edges (length must be len(counts) + 1)
            density: Whether the counts represent density (normalized) or absolute counts
            name: Optional name for the histogram
            unit: Optional unit for the data values

        Returns:
            HistogramResult instance
        """
        return cls(bin_edges=bin_edges, counts=counts, name=name, density=density, unit=unit)
