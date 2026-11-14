# AI Infra 讲义

从 Transformer 骨架一路到推理系统的 AI Infra 讲义，每篇含依赖图、数值例子、面试要点与常见坑。
篇目表由发布清单自动生成（见下），编号即阅读顺序。

## 怎么读

1. 左侧按编号顺序读（编号即清单顺序），每篇开头的「依赖图」写明前置
2. 顶部搜索框直接搜术语（例如 `KV cache`、`PD 分离`、`roofline`）
3. 右上角可切深色，手机上同样可用
4. 配套的 [手算卡](worksheets/index.md) 是要打印出来动笔的

<!-- ENTRIES:BEGIN（以下篇目由 publish_notes.py 按 publish.yaml 生成，勿手改）-->

## 篇目（共 29 篇，按发布清单生成）

| # | 篇 |
|---|---|
| 01 | [第一层 · 骨架 —— 从文本到矩阵](notes/01_skeleton.md) |
| 02 | [第二层 · 心脏 —— Self-Attention 的 Q/K/V](notes/02_attention.md) |
| 03 | [第三层 · 加强 —— Multi-Head、残差、LayerNorm、FFN](notes/03_block.md) |
| 04 | [第四层 · 组装 —— Encoder / Decoder / 掩码 / Cross-Attention](notes/04_architecture.md) |
| 05 | [第五层 · 现代变体 —— 从原论文到 LLaMA/GPT](notes/05_modern.md) |
| 06 | [第六层 · 跑起来 —— 训练/推理、KV Cache、完整 forward](notes/06_inference.md) |
| 07 | [第七层 · 预训练 —— 目标、数据与 Scaling Law](notes/07_pretraining.md) |
| 08 | [第八层 · 对齐 —— SFT、RLHF、DPO 与安全](notes/08_alignment.md) |
| 09 | [第九层 · 评测 —— benchmark、污染、judge 与可复现](notes/09_evaluation.md) |
| 10 | [第十层 · 解码 —— 从 logits 到 token](notes/10_decoding.md) |
| 11 | [第十一层 · 应用 —— Prompt、RAG、工具与 Agent](notes/11_applications.md) |
| 12 | [第十二层 · 多模态 —— VLM、扩散与语音](notes/12_multimodal.md) |
| 13 | [第十三层 · 推理模型 —— 测试时计算、长 CoT 与 RLVR](notes/13_reasoning.md) |
| 14 | [第十四层 · 长上下文 —— 位置外推、注意力模式与 KV 账本](notes/14_long_context.md) |
| 15 | [第十五层 · PEFT 与 LoRA —— 用 0.1% 参数做微调](notes/15_peft_lora.md) |
| 16 | [第十六层 · Agent RL —— rollout、环境、奖励与异步训练](notes/16_agent_rl.md) |
| 17 | [第十七层 · 数据工程 —— 从原始语料到数据飞轮](notes/17_data_engineering.md) |
| 18 | [第十八层 · 可解释性 —— 从探针到电路与 SAE](notes/18_interpretability.md) |
| 19 | [第十九层 · 多模态深化 —— VLM、视频/音频与扩散](notes/19_multimodal_deep.md) |
| 20 | [M2-0 · 地图与坐标系](notes/20_ai_infra_map.md) |
| 21 | [M2-1 · 单机与算子（地基）](notes/21_single_node_ops.md) |
| 22 | [M2-3 · 推理系统](notes/22_inference_systems.md) |
| 23 | [M2-2 · 训练系统](notes/23_training_systems.md) |
| 24 | [M2-4 · 分布式底座](notes/24_distributed_base.md) |
| 25 | [M2-5 · 平台与稳定性](notes/25_platform_sre.md) |
| 26 | [M2-6 · 前沿与端侧](notes/26_frontier_edge.md) |
| 27 | [第二十七层 · Agent 工程化：Context / Harness / Loop / Graph](notes/27_agent_engineering.md) |
| 28 | [第二十八层 · vLLM v1 源码 trace：一次请求怎么走](notes/28_vllm_source_trace.md) |
| 29 | [第二十九层 · Agent 评测 Infra：从打分到可复现的评测平台](notes/29_agent_eval_infra.md) |

<!-- ENTRIES:END -->

## 关于

这些是我自己学习时写的讲义，内容会持续修订。有错欢迎在仓库里提 issue。
