# Vendored NInfer artifact writer closure

The files in `ninfer_artifact/` are copied unchanged from the `tools/artifact/`
package shipped in the pinned NInfer Worker image
`398328bc124097f194a305a329e43fb2eed3688a304a9b931f423e5bf66fcace`.
The matching local upstream source is `inference/ninfer-rl-worker`, commit
`7afcb6c2e165674d888383cabcdfef43fb9b144d`; every vendored Python file was
SHA-256 compared with that source and matched. Only the unchanged dependency
closure for `ArtifactWriter` is included: `writer`, `schema`, `layouts`,
`formats`, `file_io`, `framing`, and the package initializer.

The upstream project license is Apache License 2.0; its full text is included
in `LICENSE`.
