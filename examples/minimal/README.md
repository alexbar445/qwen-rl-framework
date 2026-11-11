# Minimal CPU example

Install the public package on Linux with Python 3.11 or 3.12, then run:

```bash
python -m pip install .
python -m qwen_rl.examples.minimal
```

No model, dataset, GPU, ARC toolkit or paid provider is required. The command
uses a temporary checkpoint directory and removes its own temporary data after
checking recovery. To retain inspectable checkpoints, choose a new directory:

```bash
python -m qwen_rl.examples.minimal --checkpoint-dir ./minimal-run
```

The example uses a scalar Bernoulli policy and actual Adam updates. It runs the
shared `RunCoordinator → AsyncGroupStream → RolloutBatch → UpdateTransaction`
path, saves parameters, Adam moments and step, RNG and logical progress, then
compares the next uninterrupted update with a new engine restored from disk.
Existing nonempty directories and inconsistent checkpoint pointers are rejected.

The backend and local policy-ACK closures are teaching implementations. This is
neither a language-model quality result nor a test of GPU or physical multi-host
training. See `qwen_rl/examples/minimal/` for the environment, reward scorer and
engine interface implementations; retain the production owner, policy and
checkpoint fences when substituting a real backend.
