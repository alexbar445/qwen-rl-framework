# Contributing

Contributions are welcome. Keep changes focused, explain the behavior they change, and include tests or a reproducible evidence note when the change affects runtime behavior.

## Before opening a change

- Read the affected public interfaces and existing CI workflow. CI is the source of truth for required checks; do not infer supported behavior from a private deployment or an unrecorded local run.
- Keep environment, model-backend, deployment, and training-core changes within their stated boundaries.
- Do not add model weights, datasets, credentials, local host paths, private run outputs, or third-party code whose provenance and redistribution terms are not verified.
- Preserve existing copyright, attribution, license, and NOTICE files when changing or redistributing bundled third-party material.
- For performance or capability claims, include the exact source revision, profile, model/artifact, hardware, workload, result, and limitations. Keep CPU, GPU, and multi-host evidence separate.
- Keep public examples runnable against the exported package, and state whether they need external assets, services, or optional dependencies. Use the [support matrix](docs/SUPPORT.md) for evidence scope and the [architecture](docs/ARCHITECTURE.md) and [extension](docs/EXTENDING.md) guides for the public interfaces.

## Local checks

Install the lightweight development dependencies, then run CPU and documentation
tests without requiring a prebuilt wheel:

```bash
python -m pip install '.[test]'
python -m pytest -q tests --ignore=tests/test_public_packaging.py --ignore=tests/test_small_model_engine.py
```

The two installed-artifact tests need a prepared wheelhouse. They create their
own isolated virtual environment and install offline:

```bash
test_wheelhouse="$(mktemp -d)"
python -m pip wheel --wheel-dir "$test_wheelhouse" '.[test]'
PUBLIC_WHEELHOUSE="$test_wheelhouse" \
PUBLIC_WHEEL_PATH="$test_wheelhouse/qwen_rl_framework-0.1.0-py3-none-any.whl" \
python -m pytest -q tests/test_public_packaging.py
```

These commands exercise the public CPU scope. GPU and physical multi-host
qualification require separately prepared profiles and evidence; see
[Support](docs/SUPPORT.md). CI repeats the checks against a committed source
export rather than relying on additional files in a developer's checkout.

## Pull requests

Describe the problem and the change, note compatibility or configuration effects, and link the relevant CI or runtime evidence. For documentation-only updates, state which source interfaces or release records the text reflects; keep relative links within the exported public tree.

Unless a contribution is explicitly marked otherwise, contributions submitted for inclusion are offered under the repository's Apache-2.0 license, subject to the terms in [LICENSE](LICENSE). Third-party contributions or assets keep their own terms and must be identified separately.

For release qualification, build from the committed export described in
[Packaging](docs/PUBLIC_RELEASE.md). Incremental `build/` output can retain files
that have been deleted from source; the installed-artifact tests explicitly
reject the retired experiment modules. A clean export avoids carrying such
local build products into a new wheel.

## Small-model training checks

Use a separate CPU environment and the installation commands in
[the small-model recipe](examples/small_model/README.md). Install `.[test]`
there too, then run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q tests/test_small_model_engine.py tests/test_training_runner.py tests/test_training_config.py
```

These tests build a tiny local Transformer and exercise actual LoRA gradients,
AdamW, publication and complete-state recovery, including fresh CLI processes.
They require no downloaded model. The pretrained 135M qualification is a separate
bounded integration run documented with the recipe; CI does not claim to repeat it.
