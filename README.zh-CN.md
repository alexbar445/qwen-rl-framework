# 分布式 LLM 强化学习框架

本仓库提供面向语言模型生成任务的分布式强化学习工作流，保留现有 `qwen_rl` Python 命名空间与 `qwen-rl` 命令。主要训练路径包含分组 rollout、GRPO/DAPO 风格更新、LoRA policy、远端梯度执行和 checkpoint 恢复；具体支持范围由 backend、profile、模型制品、运行环境和硬件共同决定。

Python 3.11 或 3.12 下可运行 CPU 教学示例：

```bash
python -m pip install .
python -m qwen_rl.examples.minimal
```

示例使用小型标量策略与 Adam 演示训练和恢复流程；它不训练语言模型，也不证明 GPU 或多机能力。它不会下载模型或数据，也不会调用付费 Teacher。

ARC 与 Bonsai 是参考集成，不是基础示例的依赖：见 [ARC](examples/arc/README.md)、[Bonsai](examples/bonsai/README.md) 和[发行说明](docs/PUBLIC_RELEASE.md)。模型、数据、凭证、集群资源和外部服务须由使用者单独准备并确认各自许可。

项目不宣称支持任意 Gym 环境、所有模型 backend、PPO 或所有 GPU/多机拓扑。CPU、GPU、多机结果须分别按对应源码、配置和硬件收据判断。详见 [英文首页](README.md)、[贡献指南](CONTRIBUTING.md)、[安全政策](SECURITY.md) 与[第三方声明](THIRD_PARTY_NOTICES.md)。原创项目内容按 [Apache-2.0](LICENSE) 授权，第三方组件保留各自声明与许可。
