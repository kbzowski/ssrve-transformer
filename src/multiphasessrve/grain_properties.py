from dataclasses import dataclass
from typing import Optional


@dataclass
class GrainProperties:
    """Lightweight class for storing grain properties."""

    phase_id: int
    compatibility: float
    threshold: float
    nucleation_ratio: float  # Fill ratio (0-1) when grain should nucleate
