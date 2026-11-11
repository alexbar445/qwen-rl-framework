# Distributed LLM RL Framework

This repository develops a distributed reinforcement-learning workflow for language-model generation. The codebase keeps its existing `qwen_rl` Python namespace and `qwen-rl` command while making the training core usable beyond the original ARC and Bonsai experiments.

The main training path is built around grouped rollouts, GRPO/DAPO-style updates, LoRA policies, remote gradient execution, and checkpoint recovery. Support is profile-specific: the presence of a backend or deployment command does not mean every model, algorithm, or topology has been validated.

## Quick start

Linux and Python 3.11 or 3.12 are required. The CPU example uses a small scalar policy and Adam to demonstrate the training and recovery flow without downloading a model, dataset, or Teacher service:

```bash
python -m pip install .
python -m qwen_rl.examples.minimal
```

This is a teaching example. It does not train a language model and does not demonstrate GPU or multi-host execution. See [the minimal example](examples/minimal/README.md) for its exact scope.

## Training and deployment scope

The repository contains Learner/Worker, grouped rollout, policy-version, remote-gradient, deployment, and recovery code. The production path depends on the selected backend, model artifact, runtime image, hardware, and configuration. These integrations are optional and must be selected explicitly; installing the base package does not supply model weights, ARC data, cluster credentials, or external services.

ARC and Bonsai are reference integrations, not requirements for the minimal example:

- [ARC example](examples/arc/README.md)
- [Bonsai example](examples/bonsai/README.md)
- [Public release and evidence scope](docs/PUBLIC_RELEASE.md)

The project does not claim support for arbitrary Gym environments, every model backend, PPO, or every GPU and multi-host layout. CPU, GPU, and multi-host evidence must be read separately and tied to the exact source, profile, and hardware that produced it.

## External assets and services

Models, datasets, credentials, cluster resources, and paid Teacher/provider calls are supplied and licensed separately. The minimal example does not start or contact them. ARC and model-specific examples require explicit setup and must not be treated as authorization to download or use third-party assets.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Original project contributions are offered under [Apache-2.0](LICENSE); bundled components retain their own notices and terms.
