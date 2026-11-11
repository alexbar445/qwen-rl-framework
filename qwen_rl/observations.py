"""The pinned toolkit owns model-visible observation text and its strict decoder."""
from .prompt import load_toolkit_runtime


def encode_observation(value: dict) -> str:
    load_toolkit_runtime()
    from arc_toolkit.model_wire import format_observation
    return format_observation(value)


def decode_observation(text: str, *, schema: str | None = None) -> dict:
    load_toolkit_runtime()
    from arc_toolkit.model_observation import OBSERVATION_SCHEMA
    from arc_toolkit.model_wire import parse_observation
    return parse_observation(text, OBSERVATION_SCHEMA if schema is None else schema)
