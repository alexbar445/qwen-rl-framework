# Bonsai reference model profile

Bonsai is a reference model/profile associated with the original training experiments. The example folder does not include model weights, adapters, tokenizer files, runtime images, or deployment credentials. Installing the framework does not download or launch them.

Use only a model artifact and quantization/runtime combination whose terms and technical requirements you have checked. Model licenses, gated access, provider terms, and hardware requirements are separate from this repository's Apache-2.0 license. The framework does not make a model license compatible or grant access to an external host.

Backend and topology support is profile-specific. A T2/NInfer, NVFP4, or other optional backend must be selected with a matching configuration and runtime; one backend's receipt does not validate the others. Read [the release evidence matrix](../../docs/PUBLIC_RELEASE.md) before making GPU or multi-host claims.

Paid Teacher/provider calls are external and are not started by this example or the base install. Configure a provider only when you have explicitly chosen it, supplied your own credentials, and reviewed any cost and data-handling terms.

This guide avoids commands and private deployment paths until a public Bonsai configuration and its exact supported runtime are included in a release.
