# Qwen RL Framework

A Python framework for grouped rollouts and reinforcement-learning updates for language-model tasks. The repository includes a small CPU teaching example and optional, profile-specific integrations.

## Try the CPU example

On Linux with Python 3.11 or 3.12:

```bash
python -m pip install .
python -m qwen_rl.examples.minimal
```

This runs a scalar-policy example through the coordinator, performs Adam updates, and checks that a restored checkpoint follows the same next update as an uninterrupted copy. It needs no model download, dataset, GPU, ARC toolkit, or Teacher service. It is a framework-path example, not language-model training or a GPU performance result.

Start here for [installation, output, and checkpoint behavior](docs/GETTING_STARTED.md). The other guides cover [architecture](docs/ARCHITECTURE.md), [custom environments and rewards](docs/EXTENDING.md), and [support and evidence](docs/SUPPORT.md).

## Integrations and evidence

ARC and Bonsai are optional reference integrations; neither is required by the CPU example. Model weights, datasets, runtime images, credentials, and external services are supplied separately. Support is specific to a configuration and its evidence. Code or an optional dependency being present does not validate every model, backend, GPU, or multi-host setup.

See the [ARC](examples/arc/README.md) and [Bonsai](examples/bonsai/README.md) notes and the [public release evidence scope](docs/PUBLIC_RELEASE.md). Contributions and third-party materials are covered by [CONTRIBUTING.md](CONTRIBUTING.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
