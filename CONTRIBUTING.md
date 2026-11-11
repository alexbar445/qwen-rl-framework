# Contributing

Contributions are welcome. Keep changes focused, explain the behavior they change, and include tests or a reproducible evidence note when the change affects runtime behavior.

## Before opening a change

- Read the affected public interfaces and existing CI workflow. CI is the source of truth for required checks; do not infer supported behavior from a private deployment or an unrecorded local run.
- Keep environment, model-backend, deployment, and training-core changes within their stated boundaries.
- Do not add model weights, datasets, credentials, local host paths, private run outputs, or third-party code whose provenance and redistribution terms are not verified.
- Preserve existing copyright, attribution, license, and NOTICE files when changing or redistributing bundled third-party material.
- For performance or capability claims, include the exact source revision, profile, model/artifact, hardware, workload, result, and limitations. Keep CPU, GPU, and multi-host evidence separate.

## Pull requests

Describe the problem and the change, note compatibility or configuration effects, and link the relevant CI or runtime evidence. For documentation-only updates, state which source interfaces or release records the text reflects.

Unless a contribution is explicitly marked otherwise, contributions submitted for inclusion are offered under the repository's Apache-2.0 license, subject to the terms in [LICENSE](LICENSE). Third-party contributions or assets keep their own terms and must be identified separately.
