# Minimal CPU example

Install the package on Linux with Python 3.11 or 3.12, then run:

```bash
python -m pip install .
python -m qwen_rl.examples.minimal
```

The command uses a temporary checkpoint and removes it after checking recovery. To retain inspectable files, choose a new or empty directory:

```bash
python -m qwen_rl.examples.minimal --checkpoint-dir ./minimal-run
```

The scalar Bernoulli policy uses real Adam state. The example sends complete two-trajectory groups through `RunCoordinator → AsyncGroupStream → RolloutBatch → UpdateTransaction`, saves parameters, optimizer moments and step, random-generator state, and logical progress, then compares the next uninterrupted update with a restored copy.

The JSON field `recovery_match` reports whether those two toy states agree. `optimizer_updates` counts the initial update plus one update on each comparison branch; it does not mean one branch performed three updates. With an explicit checkpoint path, the JSON `checkpoint` points to a retained toy snapshot. Existing non-empty directories and inconsistent checkpoint pointers are rejected.

This demonstrates the shared control path and a local toy recovery check. It does not train a language model or validate a production optimizer, GPU, remote Worker, or physical multi-host recovery. See [Getting started](../../docs/GETTING_STARTED.md) for output details and [Support](../../docs/SUPPORT.md) for the evidence boundary.
