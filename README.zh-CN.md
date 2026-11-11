# Qwen RL Framework

面向语言模型任务的分组 rollout 与强化学习更新框架。仓库包含一个 CPU 教学示例、一个真实模型单机 CPU recipe，以及按具体 profile 选择的可选集成。

## 运行 CPU 示例

Linux、Python 3.11 或 3.12：

```bash
git clone https://github.com/alexbar445/qwen-rl-framework.git
cd qwen-rl-framework
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m qwen_rl.examples.minimal
```

示例用标量策略经过 coordinator、执行 Adam 更新，并检查从 checkpoint 恢复后，下一步结果是否与未中断副本一致。不需要下载模型或数据，也不使用 GPU、ARC toolkit 或 Teacher 服务。这只是框架流程示例，不是语言模型训练或 GPU 性能结果。

参见[安装、输出和 checkpoint 说明](docs/GETTING_STARTED.md)、[架构](docs/ARCHITECTURE.md)、[接入自定义环境与奖励](docs/EXTENDING.md)和[支持范围与证据](docs/SUPPORT.md)。

## 在 CPU 上训练真实模型

[小模型 recipe](examples/small_model/README.md) 使用固定 revision 的
SmolLM2-135M，以 CPU float32 完成单 token 的 yes/no 生成和 LoRA 更新。首次
运行会从 Hugging Face 下载约 272 MB 权重，不需要 GPU。按 recipe 安装
`small-model` extra 后，在项目根目录运行：

```bash
qwen-rl train --config examples/small_model/train.yaml --run-dir ./runs/small-model --steps 2
```

示例使用仓库内 12 道算术题。它展示这个有限任务上的真实模型更新，不证明
自由文本能力或一般推理能力提高，也不代表 GPU 或多机已获验证。

## 集成与证据

ARC 与 Bonsai 是可选参考集成，CPU 示例不依赖它们。SmolLM2 权重由 recipe 按固定 revision 从 Hugging Face 单独下载；其他模型权重、数据、运行镜像、凭证和外部服务须由使用者单独准备。支持范围绑定具体配置及其证据；源码中存在某段代码或可选依赖，不代表所有模型、backend、GPU 或多机布局都已验证。

参见 [ARC](examples/arc/README.md)、[Bonsai](examples/bonsai/README.md)、[公开发行证据范围](docs/PUBLIC_RELEASE.md)、[贡献指南](CONTRIBUTING.md)和[第三方声明](THIRD_PARTY_NOTICES.md)。

请先取得并解压源码（或克隆仓库），再在项目根目录运行以上命令。完整指南随源码分发；单独安装 wheel 不会安装文档目录。

已安装 wheel 的用户可阅读[在线入门指南](https://github.com/alexbar445/qwen-rl-framework/blob/main/docs/GETTING_STARTED.md)。
