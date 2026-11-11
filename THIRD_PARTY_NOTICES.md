# Third-party notices

This is a source-level inventory of bundled third-party material, not a complete license report for every optional or transitive dependency. Python package dependencies have their own licenses; consult the metadata distributed with each dependency and the exact release artifact.

## NInfer artifact writer

The files under `qwen_rl/_vendor/ninfer_artifact/` are an unchanged copy of the `tools/artifact/` writer closure from the pinned NInfer Worker source recorded in [`UPSTREAM.md`](qwen_rl/_vendor/ninfer_artifact/UPSTREAM.md). That file records the upstream commit, image digest, and the local SHA-256 comparison. The upstream license is Apache-2.0; retain the accompanying [`LICENSE`](qwen_rl/_vendor/ninfer_artifact/LICENSE) with these files. This notice does not claim ownership of the upstream code.

## Apache ECharts 6.1.0

The dashboard bundles `qwen_rl/dashboard/static/vendor/echarts.min.js`, version 6.1.0. Preserve the original [`ECHARTS-LICENSE.txt`](qwen_rl/dashboard/static/vendor/ECHARTS-LICENSE.txt), [`ECHARTS-NOTICE.txt`](qwen_rl/dashboard/static/vendor/ECHARTS-NOTICE.txt), and [`manifest.json`](qwen_rl/dashboard/static/vendor/manifest.json) with the bundled file. The upstream notice states:

> Apache ECharts
> Copyright 2017-2026 The Apache Software Foundation
>
> This product includes software developed at The Apache Software Foundation (https://www.apache.org/).

The ECharts license file identifies d3.js-derived subcomponents under BSD 3-Clause. The applicable d3.js notice is reproduced below, with its copyright and conditions, so that the dashboard bundle's distribution retains that attribution.

### d3.js BSD 3-Clause notice

Copyright 2010-2016 Mike Bostock
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the author nor the names of contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Excluded pending provenance review

The private source tree contains research copies at `vendor/paired-factory-v1/` and `vendor/arc-taskgen-v2/`, and the patch `deploy/worker/native-shared-prefix.patch`. Their redistribution provenance and license scope have not been established for this release. They are excluded from the public source export; they are not licensed by the root Apache-2.0 file. Do not add them to a release artifact until their source, notices, and redistribution terms have been reviewed.

Models, datasets, checkpoints, adapters, external inference runtimes, and paid Teacher/provider services are also outside this project's license grant. Their owners' terms apply separately.
