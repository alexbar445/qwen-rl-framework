# ARC reference integration

ARC is an optional reference environment for the framework. The CPU minimal example does not depend on ARC, and the base install does not download ARC tasks, start a verifier, or call a Teacher.

Before using an ARC integration, confirm that the selected public release includes the adapter and its declared optional dependencies. Supply a dataset and evaluator that you are authorized to use. Data files and task catalogs are not licensed by the framework's Apache-2.0 license; check their own provenance and terms.

Treat an ARC score as scoped to its exact dataset split, task source, verifier, model, prompt, and run configuration. A diagnostic run, finite sample, replay, or toy example is not evidence of broad ARC generalization. Keep private evaluation answers and credentials out of logs, examples, and public issues.

This page intentionally provides no command copied from the private ARC deployment workflow. Use commands only when the selected release's adapter documentation and CI or release receipts identify a runnable configuration.
