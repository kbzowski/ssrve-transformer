from typing import Dict, Type, Callable
from threading import Lock


class AnalysisRegistry:
    """Thread-safe registry for SSRVE analysis functions."""

    def __init__(self):
        """Initialize the registry with thread-safe storage."""
        self._registry: Dict[str, Type] = {}
        self._lock = Lock()

    def register(self, name: str) -> Callable:
        """Decorator to register an analysis class in a thread-safe manner.

        Args:
            name: Name to register the analysis under

        Returns:
            Decorator function
        """

        def decorator(analysis_class: Type) -> Type:
            with self._lock:
                self._registry[name] = analysis_class
            return analysis_class

        return decorator

    def get_analysis(self, name: str) -> Type:
        """Get analysis class by name in a thread-safe manner.

        Args:
            name: Registered name of the analysis

        Returns:
            Registered analysis class

        Raises:
            KeyError: If analysis name is not found
        """
        # No lock needed for read-only access to dict
        if name not in self._registry:
            available = list(
                self._registry.keys()
            )  # Make a copy to avoid race conditions
            raise KeyError(
                f"Analysis '{name}' not found. "
                f"Available analyses: {', '.join(available)}"
            )
        return self._registry[name]

    def get_available_analyses(self) -> list[str]:
        """Get list of registered analysis names in a thread-safe manner.

        Returns:
            List of analysis names
        """
        # Make a copy to avoid race conditions
        return list(self._registry.keys())


# Create singleton instance
registry = AnalysisRegistry()
