# Small-model CPU recipe

This source-tree recipe runs a real causal language model on CPU through the
root `qwen-rl train` command. It is intentionally bounded: the task is binary
yes/no arithmetic, the answer is one constrained token, and the adapter is a
small LoRA. It is a training-path example, not evidence of free-form language
ability or general reasoning improvement.

## Model and license

The recipe fetches `HuggingFaceTB/SmolLM2-135M` at the immutable revision
[`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2).
At that revision, the [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/README.md)
declares [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).
The repository is about 272 MB; the first run downloads it into the Hugging
Face cache. This project does not include or redistribute the weights, and
`trust_remote_code` is disabled.

## Install and train

On Linux, run these commands from the project root with Python 3.11 or 3.12.
Use an isolated environment. Installing Torch from the official CPU wheel
index keeps this recipe separate from the GPU-oriented training extras:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.10.0"
python -m pip install '.[small-model]'
qwen-rl train --config examples/small_model/train.yaml --run-dir ./runs/small-model --steps 2
```

The recipe uses CPU float32, two Torch threads, seed 7, and LoRA rank 4 on
`q_proj` and `v_proj` (alpha 8, dropout 0). It uses PyTorch AdamW. Each update
uses groups of four actual model samples and their sampled behavior
log-probabilities; duplicate answers are allowed, and the runner does not
manufacture a diverse group. The one-token choices are `" yes"` and `" no"`.
Their leading spaces match the model's tokenization; the runner checks that
both choices are single tokens. The update is GRPO-style, with group-relative
rewards, a 0.2 clipping ratio, and a 3.0 importance-ratio cap. This bounded
objective does not claim an unbiased policy-gradient estimator.

The `small-model` extra pins Torch 2.10.0, Transformers 4.57.6, and PEFT
0.20.0. Sampling temperature is fixed at 1.0 and LoRA dropout is fixed at
0.0; these are recipe values, not additional tuning fields. `epsilon: 0.5`
controls exploration, but it does not force a group to contain both answers.

`examples/small_model/tasks.jsonl` contains twelve original arithmetic
examples. Each line has exactly `task_id`, `prompt`, and `answer`; answers are
the lowercase labels `yes` or `no`. The runner validates each input row at the
file boundary and passes a typed `Task` to the reward callback. The recipe's
built-in exact-match reward strips surrounding whitespace before comparing
the generated completion with the answer.

## Resume or extend a run

Use the same run directory and add `--resume`. `--steps` is the total target
number of committed updates, not a number to add. If the two-update run was
interrupted, this continues it toward four updates. If it already completed,
raising the target starts a continuation segment in the same lineage with the
same optimizer state:

```bash
qwen-rl train --resume --run-dir ./runs/small-model --steps 4
```

Keep the run directory with its checkpoints and state files to resume later.
Use a different run directory to start a separate lineage.

## Supply a custom reward

Write each JSONL row with `task_id`, `prompt`, and `answer`. After the input
boundary check, a reward callback receives the typed `Task` and generated
completion, then returns one float:

```python
from qwen_rl.training.tasks import Task


def reward(task: Task, completion: str) -> float:
    return float(completion.strip() == task.answer.strip())
```

The same function is shown in `reward.py`. To use that checked-in callback from
the CLI, set `task.reward` to `reward:reward` and explicitly expose its
directory when launching the command:

```bash
PYTHONPATH=examples/small_model qwen-rl train --config examples/small_model/train.yaml --run-dir ./runs/small-model --steps 2
```

For another custom callback, set `task.reward` to `module:function` and make
that module explicitly importable in the environment where `qwen-rl` runs.
The CLI does not guess or modify `sys.path`. Otherwise, the bundled recipe
uses `qwen_rl.training.tasks:exact_match_reward`.

## Scope

This recipe runs a single CPU process. It does not exercise GPU training,
multi-host execution, or free-form generation. A successful update only shows
that this bounded task and training path ran; it is not evidence that the base
model or adapter gained general language or reasoning ability. Consult
[Support and evidence](../../docs/SUPPORT.md) for the qualification status of
this source revision.

## Verified integration run

The [qualification receipt](QUALIFICATION.json) records a CPU float32 run of the
actual pinned 134,515,008-parameter model: four committed LoRA/AdamW updates,
compared with two updates followed by a fresh CLI process resuming to four.
Adapter tensors, optimizer, scheduler, Python/NumPy/Torch RNG, task cursor and
next sampled group matched exactly. A zero-variance group was genuinely sampled
and filtered; recovery also preserved the cursor across that refill.
The receipt binds the relevant source files and dependency versions. This is
recovery evidence for this bounded recipe, not a model-quality benchmark.

CI uses an independently generated tiny local Llama to test actual gradients and
fresh-process recovery offline. It does not download or re-qualify the pretrained
135M model on every change.
