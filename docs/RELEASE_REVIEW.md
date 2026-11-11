# Public release scope review

This review starts from public commit `4fa2cab`. It changes the product surface,
not the original research checkout or its saved experiments and checkpoints.
Qualification results for the reviewed tree are recorded after the final tests;
the older [initial evidence](../provenance/initial-20261111/RELEASE_EVIDENCE.json)
remains a historical receipt.

## Removed research-only code

- `span_pivot`: the self-contained ARC trajectory rewrite/dedup/replay pilot.
- Fixed 9B pilot evaluation/receipt validators and the specialized rank-32
  evaluation-transition helper. General evaluation and model identity checks stay.
- The separate compression-recovery training objective, its six training modules,
  and its exclusive MLA recovery model helper. Normal T2 trajectory/self-smoke
  compute remains; the experiment-only `--check` mode is gone.

The removals follow consumer searches in the public package, tests and CLI.
Training-state recovery is unrelated to that removed compression objective and
remains intact. Historical compression provenance checks on already produced
model artifacts also remain intact. NInfer/vLLM backends, distributed owner/lease
fences, remote gradient execution and checkpoint handling are retained.

## Fixed misleading entry points

Instance setup now requires an explicit runtime template and local deployment
requires an explicit profile. Neither points at an absent bundled default.
The T2 kernel error no longer recommends a build script absent from this tree.
The unresolved native wheel is not an installable public dependency; the renamed
`ternary-deps` extra contains only its public Python prerequisites.

## Documentation

The guides now separate first run, architecture, extending an environment, support
and packaging. The complete extension example is executable without a model, GPU
or Teacher. CI checks local documentation links, retired module/CLI absence,
configuration failure before allocator setup, wheel contents and installed usage.

## Qualification boundaries

CPU tests, wheel installation, documentation examples and source scans qualify
only those scopes. This cleanup adds no GPU, model-quality or physical multi-host
claim. Python3.11 is a CI target; local qualification uses Python3.12 on Linux.
