# Public release scope

This page separates implemented code paths from runtime evidence. A package, backend, or command being present in source is not by itself proof that a target configuration runs successfully.

## Evidence matrix

Each release should record the exact source revision, command, profile/configuration, model and artifact identity, runtime image, hardware, topology, result, and known limits. Keep the three scopes below separate:

| Scope | What counts as evidence | Current statement |
|---|---|---|
| CPU teaching example | The scalar-policy example and its checkpoint recovery, on the named CPU environment | The example is intended to exercise a small end-to-end path. It is not model-training evidence; use the linked CI or release receipt for a pass result. |
| GPU training | A named backend/profile run on the stated GPU model, software image, model artifact, and workload | No GPU result is implied by the CPU example. Check for a matching release receipt before treating a profile as GPU-validated. |
| Multi-host training | A run across independently identified hosts, with the stated network, worker/learner placement, and recovery scope | No multi-host result is implied by a same-host run or by the presence of cluster code. Check for a matching release receipt. |

Older local test counts and receipts from other revisions, profiles, or private environments are historical evidence. Do not carry them forward as results for the public snapshot. A bounded mechanism or connectivity check is not a model-capability or generalization result.

## Minimal CPU example

The example is deliberately small and self-contained:

```bash
python -m pip install .
python -m qwen_rl.examples.minimal
```

It uses a scalar policy and Adam to demonstrate the training flow and recovery using a fresh instance. It does not download or load a language model, use ARC data, start a Teacher, access a GPU, or span physical hosts. Its result must not be described as a language-model training run.

## Optional integrations

GPU backends and deployment paths require their own optional dependencies, runtime images, configuration, model artifacts, and compatible hardware. Their support is profile-specific; optional installation does not mean every profile has been tested on every supported device.

ARC and Bonsai are reference integrations. Data, task catalogs, verifiers, models, adapters, and inference services are not bundled by the minimal path. Before using an integration, verify that its dependencies and input assets are part of the selected public release and that their separate terms allow the intended use.

Model downloads and external Teacher/provider calls are explicit setup choices. They are not started by the base installation or minimal example. A paid provider can incur charges and must be configured separately by the operator.

## Environment and reward boundary

The Learner coordinator accepts a `Task`-shaped task catalog, an `Environment` that yields complete rollout groups, and a `RewardScorer` that returns one `Reward` per trajectory. `create_learner_app(..., environment=..., reward_scorer=...)` uses that injected path without loading the ARC task source, ARC service, prompt builder, or ARC AgentLoop. Omitting those arguments selects the existing ARC adapter and its trusted scoring path.

The shared pipeline validates one environment group at its boundary, attaches scorer values to trajectories, then continues through group admission, `RolloutBatch`, `UpdateTransaction`, and the selected learner engine. A custom adapter must provide a stable `source_identity`; resume rejects a missing or changed identity. The adapter owner must change that value when task content or rules change. The string is an operator supplied semantic version, not a content hash or proof of task contents. The current CPU example exercises this shared path with a scalar policy; it does not validate a language model, GPU backend, distributed hardware, or multi-host recovery. BEPA and held-out ARC evaluation remain available only through the ARC adapter.

## Source and package contents

Public source exports use an explicit allowlist. Private journals, run outputs, checkpoints, local deployment credentials, and research worktrees are not part of the product documentation or package. Components with unresolved redistribution provenance are excluded from the public source snapshot until their source and license are verified; the Apache-2.0 license for original project material does not extend to excluded or separately licensed components.

The current source tree may still contain research-only copies that are omitted by the release exporter. Check the exported file manifest, the root [LICENSE](../LICENSE), and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the exact contents of a release artifact.

## Release evidence checklist

Before publishing a release, attach evidence for the artifact being published:

- CPU CI and the minimal example, with exact source revision and command.
- Any claimed GPU profile, with its model, image, artifact, hardware, and workload.
- Any claimed multi-host profile, with actual host topology and recovery boundary.
- Source and wheel file manifests, dependency metadata, and retained third-party licenses/notices.
- A final source scan for credentials, private paths, and files outside the allowlist.

If evidence is absent, label that scope unverified rather than inferring support from another scope.

## Building a clean source snapshot

Commit the reviewed changes before exporting. The exporter reads the selected
Git revision, never dirty/untracked files, and refuses existing output directories
or symlinked source entries:

```bash
python tools/build_public_snapshot.py --ref HEAD --output /tmp/qwen-rl-public-new
```

The resulting `PUBLIC_SOURCE_MANIFEST.json` binds every selected file to its
source commit and content hash. It does not certify licensing, model behavior or
absence of secrets. Review third-party notices and scan the actual exported tree
before publishing. CI builds and installs its wheel from that same source tree.

Gitleaks8.30.1 is used for the release source scan. The small per-rule allowlist
only recognizes reviewed public signing-key metadata and code expressions; it
does not exempt whole files from scanning. A changed fake token in the same file
was used as a negative control. Private research Git history is not part of the
new public repository.
