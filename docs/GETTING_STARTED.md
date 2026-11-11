# Getting started

## Requirements

- Linux
- Python 3.11 or 3.12

The documented CPU path uses only the base package. It does not need a model, dataset, GPU, ARC toolkit, or Teacher service.

## Install and run

Download/unpack the source archive or clone the repository, then open its root. Use an isolated environment to avoid changing system or existing training packages:

```bash
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
| Implement a custom task and reward | [Environment guide](EXTENDING.md) and the Python application factory | This is a programmatic integration; the standard runtime CLI does not yet select arbitrary custom adapters from YAML. |
| Train a real model | Supply a qualified model backend, Worker managers and runtime configuration | This release has no turnkey, public small-model training recipe. |
| Operate an already prepared single-host deployment | `instance` with explicit profiles | Requires the external configuration and assets described in [Support](SUPPORT.md). |
| Integrate remote compute or Workers | `gradients` / `cluster` | Advanced integration; a local toy result does not qualify physical multi-host execution. |

The first path is the current reproducible quick start. The other paths need
explicit integration work; passing the CPU demo is not evidence that they are
configured automatically.
