# Support and evidence

Support is tied to an exact source revision and execution profile. The presence of code, a backend name, an optional dependency, or a historical private run does not qualify other models or hardware.

## Evidence levels

- **Qualified here** means the release evidence names the environment, command, and result. The claim stays within that recorded scope.
- **Implemented** means a public code path exists. A matching runtime receipt is still needed before calling a model or deployment profile validated.
- **External setup required** means datasets, models, runtime images, credentials, or services must be supplied and checked separately.
- **No qualification recorded** means this snapshot provides no evidence for that scope.

## Current matrix

| Area | Public code or evidence | What the evidence supports |
|---|---|---|
| CPU minimal example | Qualified on Linux x86_64 with Python 3.12.8 in the release evidence. It uses a scalar Bernoulli policy, real Adam state, and a local checkpoint/recovery comparison. | The toy coordinator/update/recovery path works for that recorded source and environment. It is not LLM training or model-quality evidence. |
| SmolLM2 small-model CPU recipe | The pinned 135M model completed four real CPU LoRA/AdamW updates. A fresh-process 2+resume-to-4 run matched continuous training in adapter, optimizer, scheduler, RNG, cursor and next sampling; see the [source-bound receipt](../examples/small_model/QUALIFICATION.json). | Single-process CPU float32 and constrained one-token yes/no only. No model-quality gain, free-form generation, GPU or multi-host qualification. |
| Python 3.11 | The public package CI workflow has 3.11 and 3.12 targets. The release evidence records local qualification on 3.12 only. | A configured CI target is not a local qualification receipt for every environment. Check the CI result for the source revision being used. |
| Generic `Environment` / `RewardScorer` boundary | The contracts and application injection path are present; the minimal example exercises the coordinator with a toy adapter. | The shared interface and toy CPU path, not every custom adapter or application configuration. Custom environments cannot use ARC-only BEPA or held-out evaluation in this snapshot. |
| ARC | Optional reference adapter. Its external toolkit, authorized dataset, and evaluator are not bundled in the minimal path. | Contract-level or fixture checks do not establish a result on an external ARC dataset, verifier, or model. Supply exact run evidence for any score claim. |
| Bonsai | Reference profile notes exist; model weights, adapter, tokenizer, runtime image, and credentials are separate. | No general claim about Bonsai-specific behavior or transfer to other models. Require a receipt for the exact profile and workload. |
| GPU Learner/Worker profiles, including speculative MTP paths | Optional backend code and dependencies exist for selected profiles. | No new GPU qualification is recorded by the release evidence. A CPU run or a different GPU/backend profile does not qualify them. |
| Physical multi-host training | Cluster and deployment code is present. | No physical multi-host qualification is recorded. Same-host connectivity or code-path checks do not establish multi-host training or recovery. |

The initial CPU qualification is preserved in [the initial evidence](../provenance/initial-20261111/RELEASE_EVIDENCE.json). [Release review](RELEASE_REVIEW.md) describes the subsequent cleanup and its checks; [Packaging](PUBLIC_RELEASE.md) explains source-to-artifact binding. An old receipt does not automatically qualify a newer tree.

## CLI prerequisites

The root CLI groups represent different roles; most still need an explicit deployment or external integration profile. `train` is the single-process JSONL-based small-model recipe described above.

| Command group | Prerequisites beyond the base package |
|---|---|
| `train` | Install the `small-model` extra and its CPU Torch wheel as shown in the [recipe guide](../examples/small_model/README.md); the pinned model weights are downloaded separately. |
| `runtime` | Explicit runtime configuration; a model backend requires its corresponding learner or worker dependencies and external model assets. |
| `gradients` | Explicit server/executor configuration and credentials; real compute needs the selected learner backend. |
| `cluster` | Explicit controller/site/Worker settings, credentials, reachable hosts and any selected images or binaries. No cluster is provisioned by base installation. |
| `dashboard` | Display dependencies where requested; offline input directories or an explicitly configured live source. |
| `instance` | Prepared runtime YAML for `setup --runtime-config`; subsequent operations require the resulting `--deployment` JSON. No default model template is bundled. |
| `local` | A prepared `--deployment PATH` before the command, plus the specified container/runtime assets. No default deployment is bundled. |
| `learner` | A prepared deployment for managed roles, or an explicit T2 compute profile and separately qualified runtime for `compute`. |
| `vast` | Prepared provider/deployment settings and external role packages/images. This front end is not an automatic cloud rental workflow. |

The `small-model`, `learner`, `worker`, `teacher`, `runtime`, `constrained` and `ternary-tp` extras identify optional Python dependencies, not model/runtime qualification. Only install the extras required by a chosen integration; the CPU minimal example needs none of them.

The `ternary-deps` extra contains public Python helper dependencies. It does **not** supply the separately qualified `qwen-rl-ternary-kernels==0.1.0` native extension. This release does not bundle a wheel or build recipe for that extension. A T2 integrator must provide a matching, independently qualified Torch/CUDA extension; installing the extra alone does not enable the backend. Missing extensions fail explicitly rather than using a different numerical implementation.

## What to include in a support report

For a reproducible runtime issue, include the source revision, install extras, command, configuration/profile, model and artifact identifiers, runtime image, hardware and topology, and the relevant output or error. Remove credentials and private evaluation data. For performance or capability claims, include workload and result details as well. Keep CPU, GPU, and multi-host evidence separate.

Before relying on a custom adapter, read [the extension contract](EXTENDING.md). For checkpoint errors, select a fresh empty directory for the toy example; the example rejects non-empty destinations and inconsistent pointers rather than overwriting or silently resetting them.
