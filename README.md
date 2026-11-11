# Qwen RL Framework

A Python framework for grouped rollouts and reinforcement-learning updates for language-model tasks. The repository includes a small CPU teaching example, a real-model single-CPU recipe, and optional, profile-specific integrations.

## Try the CPU example

First download/unpack the source archive or clone this repository and open its root directory. On Linux with Python 3.11 or 3.12:

```bash
git clone https://github.com/alexbar445/qwen-rl-framework.git
cd qwen-rl-framework
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m qwen_rl.examples.minimal
```

This runs a scalar-policy example through the coordinator, performs Adam updates, and checks that a restored checkpoint follows the same next update as an uninterrupted copy. It needs no model download, dataset, GPU, ARC toolkit, or Teacher service. It is a framework-path example, not language-model training or a GPU performance result.

Start here for [installation, output, and checkpoint behavior](docs/GETTING_STARTED.md). The other guides cover [architecture](docs/ARCHITECTURE.md), [custom environments and rewards](docs/EXTENDING.md), and [support and evidence](docs/SUPPORT.md).

## Train a real model on CPU

The [small-model recipe](examples/small_model/README.md) adds a bounded
SmolLM2-135M path with a pinned model revision, CPU float32, one-token
yes/no generation, and LoRA updates. It downloads about 272 MB of external
weights and does not require a GPU. From the project root, run it after
installing the documented `small-model` extra:

```bash
qwen-rl train --config examples/small_model/train.yaml --run-dir ./runs/small-model --steps 2
```

This example trains on twelve included arithmetic prompts. It demonstrates a
real model update on this bounded task; it does not establish free-form
language ability, general reasoning improvement, GPU support, or multi-host
qualification. See [Support](docs/SUPPORT.md) for the evidence boundary.

## Integrations and evidence

ARC and Bonsai are optional reference integrations; neither is required by either CPU example. The SmolLM2 weights are fetched separately from Hugging Face at the revision pinned by the recipe. Other model weights, datasets, runtime images, credentials, and external services are supplied separately. Support is specific to a configuration and its evidence. Code or an optional dependency being present does not validate every model, backend, GPU, or multi-host setup.

See the [ARC](examples/arc/README.md) and [Bonsai](examples/bonsai/README.md) notes and the [public release evidence scope](docs/PUBLIC_RELEASE.md). Contributions and third-party materials are covered by [CONTRIBUTING.md](CONTRIBUTING.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The linked guides are distributed in the source archive/repository. A standalone
wheel contains the runtime package and declared resources, not the documentation
tree. Read the [hosted getting-started guide](https://github.com/alexbar445/qwen-rl-framework/blob/main/docs/GETTING_STARTED.md)
when using an installed wheel.
