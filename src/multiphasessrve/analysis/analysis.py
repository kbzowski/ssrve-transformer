from typing import Any, Dict, Set
import importlib
from pathlib import Path
from multiphasessrve.ssrve import Ssrve
from multiphasessrve.analysis.registry import registry


class SSRVEAnalysis:
    """Manager class for SSRVE analysis functions."""

    # Class-level cache of loaded modules
    _loaded_modules: Set[str] = set()

    @classmethod
    def _load_module(cls, module_name: str) -> None:
        """Load a single analysis module if not already loaded.

        Args:
            module_name: Name of the module to load
        """
        if module_name not in cls._loaded_modules:
            importlib.import_module(f"multiphasessrve.analysis.{module_name}")
            cls._loaded_modules.add(module_name)

    @classmethod
    def _load_all_modules(cls) -> None:
        """Load all analysis modules from the analysis directory."""
        analysis_dir = Path(__file__).parent
        for file in analysis_dir.glob("*.py"):
            if file.stem not in ["__init__", "registry", "analysis"]:
                cls._load_module(file.stem)

    @classmethod
    def _ensure_analysis_loaded(cls, analysis_name: str) -> None:
        """Ensure analysis module is loaded if it exists."""
        if analysis_name not in registry._registry:
            # Try direct module load first
            try:
                cls._load_module(analysis_name)
            except ImportError:
                # If direct load fails, try loading all modules
                cls._load_all_modules()

    def __init__(self, ssrve: Ssrve):
        """Initialize the analysis manager.

        Args:
            ssrve: SSRVE instance to analyze
        """
        self.ssrve = ssrve

    def analyze(self, analysis_name: str, **kwargs) -> Any:
        """Perform specified analysis.

        Args:
            analysis_name: Name of the analysis to perform
            **kwargs: Additional parameters passed to the analysis method

        Returns:
            Result of the analysis

        Raises:
            KeyError: If analysis name is not found
            ValueError: If required parameters are missing or invalid
        """
        self._ensure_analysis_loaded(analysis_name)
        analysis_class = registry.get_analysis(analysis_name)
        return analysis_class.analyze(self.ssrve, **kwargs)

    @classmethod
    def get_available_analyses(cls) -> list[str]:
        """Get list of available analyses."""
        cls._load_all_modules()
        return list(registry._registry.keys())
