"""
Event Logger for Google Gen AI Agents
A comprehensive toolkit for tracking, analyzing, and mining interaction data.
"""

__version__ = "0.3.1"

# Import main classes for easy access
from .logger import EventLogger, serialize_event_to_dict
from .processor import EventProcessor
from .insights import InsightExtractor

# Lazy import for visualizer (optional dependency)
def _get_visualizer():
    """Lazy import EventVisualizer to avoid requiring matplotlib on import."""
    try:
        from .visualizer import EventVisualizer
        return EventVisualizer
    except ImportError as e:
        def _raise_import_error(*args, **kwargs):
            raise ImportError(
                "EventVisualizer requires visualization dependencies. "
                "Install with: pip install 'event_logger[viz]' or pip install 'event_logger[full]'"
            ) from e
        return _raise_import_error

# Make EventVisualizer available but don't fail if dependencies missing
EventVisualizer = _get_visualizer()

__all__ = [
    "EventLogger",
    "serialize_event_to_dict",
    "EventProcessor",
    "InsightProcessor",
    "EventVisualizer",
    "__version__",
]