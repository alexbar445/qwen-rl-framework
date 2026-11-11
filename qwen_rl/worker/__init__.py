"""Standalone vLLM generation worker and optional verl HTTP client."""

__all__ = ["HTTPGenerationManager"]


def __getattr__(name: str):
    # Keep the standalone Worker app independent from importing verl/Ray. The
    # Learner imports the client explicitly when it needs TokenOutput support.
    if name == "HTTPGenerationManager":
        from .client import HTTPGenerationManager

        return HTTPGenerationManager
    raise AttributeError(name)
