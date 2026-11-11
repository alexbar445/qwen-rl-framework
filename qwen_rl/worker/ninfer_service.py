"""Private child entry for the Worker HTTP process supervised by NInfer."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence


def _inspect_selected_gpu(expected_uuid: str) -> dict[str, object]:
    """Read physical identity through NVML; this frontend never initializes CUDA.

    NVIDIA_VISIBLE_DEVICES is a container-creation hint, not evidence that the
    already-running native Engine can use a GPU. The launcher pins that Engine's
    CUDA_VISIBLE_DEVICES to the configured physical UUID independently.
    """
    from ..cluster.host import probe_gpus

    if not expected_uuid or os.environ.get("GPU_UUID") != expected_uuid:
        raise RuntimeError("Worker GPU identity differs from its launch contract")
    matches = [gpu for gpu in probe_gpus()
               if gpu.uuid.removeprefix("GPU-") == expected_uuid.removeprefix("GPU-")]
    if len(matches) != 1:
        raise RuntimeError("selected physical GPU UUID is absent or ambiguous in NVML inventory")
    selected = matches[0]
    return {
        "gpu_uuid": expected_uuid,
        "gpu_name": selected.name,
        "gpu_memory_total_mib": selected.memory_total_mib,
        "gpu_compute_capability": selected.compute_capability,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwen-rl-ninfer-worker-service")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8241)
    args = parser.parse_args(argv)

    from ..cluster.server import create_worker_server
    from ..cluster.settings import load_settings_json

    config = load_settings_json(args.config)
    from .residency import parse_residency_settings
    model_loader = None
    if parse_residency_settings(config).engine_lifecycle == "restart":
        from .ninfer_process import NInferProcess
        from .ninfer_backend import load_ninfer_backend
        contract_path = os.environ.get("QWEN_NINFER_ENGINE_CONTRACT")
        if not contract_path:
            raise RuntimeError("restart residency requires launcher-owned Native process contract")
        native = NInferProcess.from_file(contract_path)

        async def fence_failed_child():
            runtime = app.state.worker_app.state.runtime
            async with runtime.generation_state:
                runtime.ready = False
                runtime.policy_ready = False
                runtime.admission_open = False
                runtime.generation_state.notify_all()

        native.failure_callback = fence_failed_child

        async def model_loader(inference, *, lora_rank, lora_alpha):
            await native.start()
            try:
                backend = await load_ninfer_backend(inference, lora_rank=lora_rank, lora_alpha=lora_alpha)
                backend.engine_process = native
                return backend
            except BaseException:
                native.close()
                raise

    app = create_worker_server(
        config,
        model_loader=model_loader,
        token_file=os.environ.get("ARC_ENV_TOKEN_FILE"),
        gpu_probe=_inspect_selected_gpu,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
