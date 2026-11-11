# Getting started

## Requirements

- Linux
- Python 3.11 or 3.12

The minimal CPU example uses only the base package. The real-model CPU recipe
is separate: it installs the `small-model` extra and downloads its pinned
135M-parameter model from Hugging Face. Neither path needs a GPU, ARC toolkit,
or Teacher service.

## Install and run

Download/unpack the source archive or clone the repository, then open its root. Use an isolated environment to avoid changing system or existing training packages:

```bash
git clone https://github.com/alexbar445/qwen-rl-framework.git
cd qwen-rl-framework
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m qwen_rl.examples.minimal
```

The example uses a temporary checkpoint directory and removes it after the recovery comparison. To keep the files for inspection, pass a new or empty path:

```bash
python -m qwen_rl.examples.minimal --checkpoint-dir ./minimal-run
```

The command prints a JSON object. Its main fields mean:

| Field | Meaning |
|---|---|
| `framework_namespace` | Identifies the framework package used by the example. |
| `path` | Shared coordinator, rollout, batch, and update interfaces exercised by the toy engine. |
| `environment` / `backend` | The example's two-action environment and standard-library scalar Bernoulli policy with Adam. |
| `optimizer_updates` | Three updates counted across the initial run and the two comparison branches; this is not three sequential updates on one branch. |
| `first_update_policy_version` | Policy version published after the initial update. |
| `continued_policy_version` / `restored_policy_version` | Final versions after the uninterrupted and restored branches each perform one update. |
| `recovery_match` | Whether the restored branch's resulting toy state exactly matches the uninterrupted branch. |
| `group_size` | Two trajectories are generated in each toy group. |
| `checkpoint` | The restored branch's snapshot path, or a note that the temporary files were removed. |
| `checkpoint_persisted` | Whether the caller-selected checkpoint directory was retained. |
| `scope` | Explicitly labels the run as a CPU toy protocol smoke, with no LLM, GPU, or multi-node claim. |

`completed_updates` and `restored_run_status` describe the restored branch's last coordinator request. That request contains one update and should finish with status `succeeded`.

## Train a real model on CPU

This path trains a constrained yes/no task with the real SmolLM2-135M causal
language model. It downloads about 272 MB from the model repository on the
first run. The recipe pins its exact Hugging Face revision and uses Apache-2.0
model weights; the weights are not bundled with this project. From the source
tree root, install the base package, CPU Torch wheel, and the isolated
small-model dependencies:

```bash
python -m pip install .
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.10.0"
python -m pip install '.[small-model]'
```

Then start the two-update run:

```bash
qwen-rl train --config examples/small_model/train.yaml --run-dir ./runs/small-model --steps 2
```

The run uses two CPU threads and a real PyTorch AdamW update over a rank-4 LoRA
adapter. Its groups contain four actual one-token model samples with sampled
behavior log-probabilities. The included twelve arithmetic tasks and their
exact-match reward make the first run self-contained; see the
[recipe guide](../examples/small_model/README.md) to change the JSONL tasks or
provide a custom reward.

Resume the same run directory by giving `--resume`. `--steps` is the total
number of committed updates to reach, not an additional count. An incomplete
run continues its original work; increasing the target after completion adds
a continuation segment in the same lineage and carries its optimizer state:

```bash
qwen-rl train --resume --run-dir ./runs/small-model --steps 4
```

This recipe demonstrates a bounded model-training path. It is not evidence of
free-form language ability or reasoning improvement, GPU training, or
multi-host execution.

## Checkpoints and errors

With `--checkpoint-dir`, the example writes `first/`, `continued/`, and `restored/` branches beneath the selected directory. Each branch contains the toy engine's snapshot and latest-snapshot pointer. The initial snapshot includes the scalar policy, Adam moments and step, random-generator state, and logical update context. The comparison checks that restoring the snapshot produces the same next state as continuing from its copy.

The selected directory must be empty if it already exists. A non-empty directory is rejected with `FileExistsError`; choose another path rather than mixing example output with existing files. Inconsistent or corrupted snapshot pointers are rejected instead of silently initializing a fresh engine.

This recovery check covers the toy engine and its checkpoint format. It does not validate a production model checkpoint, optimizer implementation, GPU restart, remote Worker recovery, or multi-host recovery. See [support and evidence](SUPPORT.md) for those boundaries.

## Next guides

- [Architecture](ARCHITECTURE.md): task selection, grouped rollout scheduling, update, and policy publication.
- [Extending the framework](EXTENDING.md): the `Environment` and `RewardScorer` contracts.
- [Support matrix](SUPPORT.md): what the public evidence does and does not establish.
- [Minimal example source notes](../examples/minimal/README.md): implementation details and limits.

## Choose the next path

| Goal | Start here | Current boundary |
|---|---|---|
| Understand and verify the framework | The CPU example above | No model or deployment assets required. |
| Supply JSONL tasks and a reward callback | `qwen-rl train` and the [small-model guide](../examples/small_model/README.md) | Typed `Task` plus `reward(task, completion) -> float`; the local recipe accepts two single-token choices. |
| Integrate a different generation environment | [Environment guide](EXTENDING.md) and the Python application factory | Advanced programmatic integration with explicit model and Worker services. |
| Train a real model on CPU | The [SmolLM2 recipe](../examples/small_model/README.md) | One-token yes/no arithmetic training on a single CPU process; no general capability or GPU claim. |
| Operate an already prepared single-host deployment | `instance` with explicit profiles | Requires the external configuration and assets described in [Support](SUPPORT.md). |
| Integrate remote compute or Workers | `gradients` / `cluster` | Advanced integration; a local toy result does not qualify physical multi-host execution. |

The first path is the current reproducible quick start. The other paths need
explicit integration work; passing the CPU demo is not evidence that they are
configured automatically.
