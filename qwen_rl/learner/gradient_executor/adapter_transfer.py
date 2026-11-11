"""Exact adapter installation with bounded pinned readback and stable parameters."""
import torch

READBACK_BYTES = 64 * 1024**2


def copy_and_verify(parameters, staged):
    devices = {p.device for p in parameters.values() if p.device.type == 'cuda'}
    try:
        _copy_and_verify(parameters, staged)
    finally:
        # Include failed allocation/copy paths: neither source nor pinned scratch
        # may be released while their owning stream still has transfers.
        for device in devices:
            torch.cuda.current_stream(device).synchronize()


def _copy_and_verify(parameters, staged):
    # Source snapshots remain alive until every stream completes. Direct copy
    # avoids allocating a second device tensor for each source.to(device).
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(staged[name], non_blocking=True)
    cuda_parameters = [(n, p) for n, p in parameters.items() if p.device.type == 'cuda']
    for name, parameter in parameters.items():
        if parameter.device.type != 'cuda' and not torch.equal(parameter.detach().cpu(), staged[name]):
            raise RuntimeError(f"installed adapter tensor {name!r} differs from its source")
    if not cuda_parameters:
        return
    size = min(READBACK_BYTES, sum(p.numel()*p.element_size() for _, p in cuda_parameters))
    scratch = torch.empty(size, dtype=torch.uint8, device='cpu', pin_memory=True)
    pending = []
    used = 0
    device = None

    def flush():
        nonlocal used
        if pending:
            torch.cuda.current_stream(device).synchronize()
            for name, actual, expected in pending:
                if not torch.equal(actual, expected):
                    raise RuntimeError(f"installed adapter tensor {name!r} differs from its source")
            pending.clear()
        used = 0

    try:
        for name, parameter in cuda_parameters:
            if device != parameter.device:
                flush()
                device = parameter.device
            actual = parameter.detach().reshape(-1)
            expected = staged[name].reshape(-1)
            width = parameter.element_size()
            offset = 0
            while offset < actual.numel():
                # Align each typed view even if future layouts mix dtypes.
                used = (used + width - 1) // width * width
                count = min((size-used)//width, actual.numel()-offset)
                if not count:
                    flush()
                    continue
                view = scratch[used:used+count*width].view(parameter.dtype)
                view.copy_(actual[offset:offset+count], non_blocking=True)
                pending.append((name, view, expected[offset:offset+count]))
                used += count*width
                offset += count
        flush()
    finally:
        # A failed equality/copy must not release pinned storage with DMA in flight.
        for _, parameter in cuda_parameters:
            if device != parameter.device:
                torch.cuda.current_stream(parameter.device).synchronize()
        if device is not None:
            torch.cuda.current_stream(device).synchronize()
