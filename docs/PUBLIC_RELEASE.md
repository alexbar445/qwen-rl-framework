# Packaging and release qualification

Start with [Getting started](GETTING_STARTED.md), [Architecture](ARCHITECTURE.md),
and [Support](SUPPORT.md). This page describes how maintainers build and qualify
an artifact, rather than repeating the user tutorials.

## Export committed source

Commit reviewed changes, then choose a new output directory outside the checkout:

```bash
python tools/build_public_snapshot.py --ref HEAD --output /tmp/qwen-rl-public-new
```

The exporter reads committed files from an explicit allowlist. It excludes dirty
and untracked edits, model/data payloads, credentials, research outputs, external
vendor trees and native patches. It rejects symlinked sources and existing output
directories. The exported `PUBLIC_SOURCE_MANIFEST.json` records the source commit
and each selected file's size and hash. It certifies file identity only.

New user documentation and focused public tests must be added to that allowlist
when appropriate. Source-only and wheel tests run against the exported tree in
CI, so private checkout files cannot quietly satisfy a missing dependency.

## Qualify the actual artifact

The CI workflow builds a base/test wheelhouse, installs the wheel with
`--no-index --find-links`, runs the public tests, and checks CLI, adapter and demo
imports from a separate virtual environment outside the checkout. It verifies
that unknown vendor/toolkit identity files and removed experiments are absent
from the wheel, while license files and declared package resources are present.

Record the source commit, exact artifact hash, interpreter, commands and results.
For this review, see [Release review](RELEASE_REVIEW.md). A newly edited tree must
not inherit an old PASS merely because its package version is unchanged.

## Source scanning and provenance

CI scans the exported source using checksum-pinned Gitleaks8.30.1. Its exceptions
are limited to reviewed public metadata and exact code expressions, using both
path and content conditions for a specific rule. They do not exempt whole files.
The source review also checks a changed fake token as a detection negative control.
No scan is a general guarantee that every possible secret is absent.

The initial source import and its qualification record are preserved unchanged in
[the initial manifest](../provenance/initial-20261111/PUBLIC_SOURCE_MANIFEST.json)
and [the initial evidence](../provenance/initial-20261111/RELEASE_EVIDENCE.json).
Those are historical records, not current-tree manifests. The original private
implementation commit and the public initial commit differ because the public
repository was created from a clean source export; file hashes bind that import.
Subsequent edits have their own Git history and qualification.

## Runtime claims

[Support](SUPPORT.md) separates CPU qualification, optional backend code, external
assets, GPU execution and physical multi-host execution. Only report a capability
when its evidence names the matching source, profile, model/artifact, image,
hardware, topology and workload. The scalar CPU example does not qualify a
language model, GPU backend, or multi-host recovery.

Original project code uses [Apache-2.0](../LICENSE). Preserve the separate terms in
[Third-party notices](../THIRD_PARTY_NOTICES.md). Do not add excluded source or
assets before their provenance and redistribution terms are established.
