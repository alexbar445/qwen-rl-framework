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

## Recorded checks

The implementation at `1b07324560c12e357c59223104e1662b7d8f1920` passed:

- 100 CPU/core/retirement/documentation tests in the review checkout.
- 102 tests against the committed public export, including independent offline
  wheel installation, outside-checkout execution, and retired-module absence.
- Gitleaks8.30.1 source scan: zero findings after the existing reviewed exceptions.

The complete extension example produced `[0.0, 1.0]`; relative documentation links
were checked locally. The source archive carries its own generated manifest and
qualification record. These results bind the stated code revision; this final
documentation entry records the results without changing runtime code.

## Final architecture and first-user review

The shared task/reward, group-admission, update and recovery boundaries are useful
for framework developers. The outer user workflow still needs refinement:

1. The CPU demonstration is reproducible, but there is no complete public
   small-model training recipe with configuration, dataset preparation, model
   backend and Worker startup. ARC/Bonsai notes do not provide that recipe.
2. Generic adapters are currently injected through the Python application
   factory; the standard runtime CLI does not select them from configuration.
   A recommended high-level training entry point remains a product gap.
3. `Environment.generate_group` receives engine, manager and Worker-pool details.
   That is an integration interface, rather than a small task-only API. A future
   public runner should hide those runtime details for ordinary task authors.
4. The eight CLI groups have distinct responsibilities, but choosing among
   deployment and runtime front ends still requires framework knowledge.

Two concrete defects were repaired during this review: dashboard help imported
a removed local-deployment constant; and a custom environment named `arc` could
accidentally enable ARC catalog logic. The dashboard now has explicit source
selection, and domain sampling uses an explicit composition flag. Regression
checks cover all eight help pages without importing model runtimes, offline
dashboard collection, and the complete generic update/recovery path with a
colliding environment name.

Quick start now starts from acquired source and an isolated virtual environment.
Wheel-only readers are told that the full guides are in the source distribution;
a hosted documentation URL still depends on publication of the remote repository.
This is a developer walkthrough, not a usability study of external users.
