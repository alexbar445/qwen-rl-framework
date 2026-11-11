"""Lazy access to the shared gradient payload codec."""
from typing import Any


def tensor_codec() -> tuple[Any, Any, str, str, Any]:
    from qwen_rl.transport.tensor_codec import (
        TENSOR_PAYLOAD_CONTENT_TYPE, TENSOR_PAYLOAD_TRANSPORT,
        decode_tensor_payload, encode_tensor_payload, max_encoded_size,
    )
    return (encode_tensor_payload, decode_tensor_payload, TENSOR_PAYLOAD_CONTENT_TYPE,
            TENSOR_PAYLOAD_TRANSPORT, max_encoded_size)
