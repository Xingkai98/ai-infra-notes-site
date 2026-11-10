---
title: 第十六层 · Agent RL —— rollout、环境、奖励与异步训练
module: 大模型基础
layer: 16
date: 2026-10-22
status: 完成
figures: [52a_agent_rl_loop, 52b_sync_async, 52c_multiturn_mask, 52d_grpo_agent, 52e_agent_cost]
---

# 第十六层 · Agent RL —— rollout、环境、奖励与异步训练

> 本层回答:Agent 的 RL 和普通 RLHF 差在哪?为什么 rollout 是新的数据生产线?多轮工具调用怎么做 loss mask、奖励与异步训练?沙箱和成本怎么算?

> **前置(按节)**
> - §1–§2:先看 `notes/11_applications.md §5`(Agent 循环)与 `notes/26_frontier_edge.md §2`(Agent Runtime)。
> - 另:§2.2 会直接用「重要性比率/截断」,先扫一眼 `notes/08_alignment.md §2`(PPO/KL)的 ratio 与 clip 即可(只需知道:新旧策略概率之比,偏离太多就截断),否则那段只能跳读。
> - §3–§4:先看 `notes/13_reasoning.md §4`(RLVR/GRPO)与 `notes/08_alignment.md §2`(PPO/KL)。
> - §5–§6:先看 `notes/26_frontier_edge.md §3`(RL Infra / rollout)。
> - §7–§8:先看 `notes/09_evaluation.md`(评测口径)。
>
> **建议路径(3 次 × 45 分钟)**
> - 第 1 次(必学):§1 Agent RL 的问题形态 + §2 rollout 引擎(52a/52b)。
> - 第 2 次(必学):§3 环境扩展 + §4 reward/verifier + §5 多轮 GRPO/loss mask(52c/52d)。
> - 第 3 次(加分):§6 异步 RL/权重同步 + §7 沙箱/成本(52e)+ §8 评测与安全。
>
> **与 26 的分工**:26 §2–§3 从平台/基础设施视角讲 Agent Runtime、沙箱选型与 rollout 闭环;
> 本篇从 RL 训练视角讲问题形态、reward/信用分配、多轮 loss mask、GRPO/PPO、异步与权重同步。
> 26 给系统口径与容量账本,本篇给训练机制与数值例子;实现细节两边互相引用,不重复展开。

## 0. 依赖图(学习顺序)
> **注意:这张图是「依赖关系」,不是阅读顺序。** `C→B`(环境先于轨迹)、`D→H`/`F→H`(异步依赖 rollout 与策略更新)、`I⇢D`/`I⇢F`(成本/沙箱是约束,不是产出)只是说明「没有这些前提,后面那节会讲不透」,正文仍按建议路径展开:§2 同步 rollout → §3 环境 → §4 reward → §5 策略更新 → §6 异步。
> 因此 §2.2 的异步段第一遍**只需读到「采样与训练重叠、旧策略数据要用比率修正」**,
> 涉及 `π_old`/clip 的**完整公式**在 §5.2(§2.2 已把 on/off-policy、stale、`ρ`、`ε=0.2` 的 clip 给全);§6.1 只讲异步架构与 staleness 控制,不重述公式;不要在这段停下来补前置。

```mermaid
graph TD
  A["Agent 循环<br/>plan/act/observe"] --> B["多步 trajectory"]
  C["环境/工具/沙箱"] --> B
  B --> D["rollout 引擎<br/>采样轨迹"]
  D --> E["reward/verifier<br/>规则/单测/PRM"]
  E --> F["策略更新<br/>PPO / GRPO"]
  F --> G["更好的 Agent"]
  G --> D
  D --> H["异步/权重同步"]
  F --> H
  I["成本/沙箱/安全"] -.-> D
  I -.-> F
  G --> J["§8 评测<br/>pass^k / 成功率"]
```

## 1. Agent RL 和普通 RLHF 差在哪

普通 RLHF 的一条样本通常是“一个 prompt → 一个回答 → 一个奖励”。Agent RL 的样本是**一条多步 trajectory**:

```
state_0 (prompt)
  → action_0 (tool call / code)
  → observation_0 (tool result)
  → action_1 ...
  → final answer / task state
```

差别:

- **多步信用分配**:最终成功/失败要归因到哪一步?中间工具调用错了,后面全错。
- **环境不可微**:工具/代码/网页不是可微函数,只能通过 rollout 采样。
- **动作空间混合**:自然语言 token + 结构化工具调用 + 代码执行。
- **成本高**:一条轨迹可能几十步、几万 token、几秒到几分钟;rollout 是瓶颈。
- **可验证性**:很多任务有客观结果(单测通过、网页状态、答案匹配),适合 RLVR。
这里的 **RLVR**(RL with Verifiable Rewards)指的是「用可自动验证的结果当奖励」,它是一种**训练范式**(规定奖励从哪来),不等于 §4.1 的某一种奖励类型:奖励来自单测/状态/精确匹配这类**程序化判定**,不需要训练奖励模型。这与 ORM(训练出来的结果奖励模型)是两回事:ORM 会分布外失真、会被 reward hacking,程序化验证器的短板是覆盖不足——这正是 §4.3 要多验证器/隐藏测试的前提。本层在 §4 展开它;想先看它作为推理训练主线的用法,再补 `notes/13_reasoning.md §4`(已在 §3–§4 的前置里,§1–§2 阶段可以先跳过)。

> 一句话:Agent RL 把“模型输出”变成“模型在环境里执行一串动作”,奖励来自**任务是否完成**,而不是单个回答的偏好分。

## 2. Rollout 引擎:新的数据生产线

![Agent RL 循环:策略 → rollout → 轨迹 → 奖励 → 更新](../diagrams/52a_agent_rl_loop.landscape.png)

### 2.1 一次 rollout 的组成

- **策略(policy)**:当前模型,按 token/动作采样;
- **环境(environment)**:工具、代码沙箱、网页、数据库、模拟器;
- **执行器**:解析工具调用、执行、把 observation 写回上下文;
- **轨迹存储**:state/action/observation/reward/token logprob;
- **验证器**:任务完成判定(单测、状态检查、精确匹配、LLM judge)。

### 2.2 同步 vs 异步

![同步 vs 异步 rollout:GPU 空转与重叠](../diagrams/52b_sync_async.landscape.png)

**同步**:采样一批轨迹 → 全部完成后训练 → 再采样。简单、on-policy;但 rollout 的长尾会让训练 GPU 空转。

**异步**:rollout  worker 持续采样,训练器从队列取数据;采样与训练重叠,吞吐更高。代价是**策略滞后**(stale policy),需要用重要性比率/截断/版本控制处理。
**这几个词先给最小定义**(本层只需要到这个程度,完整公式见 §5.2):

- **on-policy**:训练用的就是当前策略刚采的数据,采样分布与目标分布一致;
- **off-policy**:训练数据来自别的策略(旧版本 `π_old`),分布对不上,直接算梯度会有偏;
- **stale policy**:异步里最轻的一种 off-policy——`π_old` 只是几版之前的自己,不是无关策略;
- **重要性比率**:`ρ = π_new(a|s) / π_old(a|s)`,把「旧策略能采到这条轨迹」折算成「新策略下它该有的权重」,让旧数据近似当 on-policy 用;
- **截断**:`ρ` 离 1 太远说明数据太旧,用 `clip(ρ, 1-ε, 1+ε)`(常用 `ε=0.2`)只让超出区间的更新不再带来额外收益(不是把梯度里的比率永远夹住);完整写法是 `min(ρ·A, clip(ρ,1-ε,1+ε)·A)`,见 §5.2。

数值示意:rollout 10 s、train 4 s。
先定义两种部署形态:**colocated** = rollout 与训练共用同一组 GPU(换权重在同组卡内完成);**disaggregated(分离池)** = rollout 与训练各用独立的一组 GPU(权重跨节点传输)。同步每轮 14 s,训练阶段占墙钟时间比例约 `4/14≈29%`(这不是 GPU 占用率:colocated 时 rollout 的 10 s 也在 GPU 上生成;只有 §7.2 那种训练 8 卡、rollout 8 卡的分离池,29% 才近似训练池的占用);异步稳定后每 10 s 产出一批(训练 4 s 藏在 10 s 内),吞吐约 `14/10=1.4×`。如果 rollout 更长(30 s),按同一算式 `(r+t)/max(r,t)` 只有 `34/30≈1.13×`,比 1.4× **更小**:重叠收益的峰值出现在 rollout≈train 附近,rollout 远长于 train 时收益趋近 1;真正随 rollout 变长而变大的是**同步模式的空转占比**(`10/14≈71%` → `30/34≈88%`),以及异步的队列/滞后。

### 2.3 rollout 的工程要点

- **并发**:每个环境一个沙箱;并发受 CPU/内存/网络/工具配额限制,不一定受 GPU 限制。
- **长尾**:少数轨迹跑很久;要设 step/token/时间上限,超时截断并给部分奖励。
- **可复现**:记录随机种子、工具版本、环境状态;否则 reward 不可复现。
- **缓存**:相同 prompt/工具结果可缓存,但要注意训练分布与缓存一致性。

## 3. 环境扩展:从玩具到真实任务

### 3.1 环境类型

| 类型 | 例子 | 验证方式 | 难点 |
|---|---|---|---|
| 代码/终端 | SWE-bench、repo 修复 | 单测/隐藏测试 | 环境依赖、长轨迹、沙箱 |
| 工具/API | 搜索、数据库、订票 | 状态/结果检查 | 工具 schema、错误处理 |
| 网页 | WebArena | 页面状态/答案 | 动态页面、反爬、延迟 |
| 对话/客服 | tau-bench | 规则/数据库状态 | 用户模拟、多轮 |
| 游戏/模拟 | 文本游戏 | 环境 reward | reward 设计、探索 |

### 3.2 环境扩展的难点

- **任务生成**:人工写任务贵;可从 issue/工单/日志合成,但要保证可验证、无泄漏。
- **状态隔离**:每个 rollout 独立沙箱,避免互相污染;启动速度很关键(Firecracker/gVisor/容器)。
- **工具版本**:工具 API 变更会让旧轨迹失效;要锁定版本。
- **可验证性**:不是所有任务都有单测;需要规则/状态/LLM judge 组合。

## 4. Reward / verifier:多步任务怎么打分

### 4.1 奖励类型

- **结果奖励(ORM / 程序化规则)**:任务完成/单测通过/答案匹配;稀疏但客观;ORM 是训练出来的结果打分模型,程序化规则不需要训练,二者别混为一谈。
- **过程奖励(PRM)**:每步打分;密集但标注贵,可能奖励“看起来对”。
- **LLM judge**:灵活,适合开放任务;有偏差、可被 hack。
- **复合奖励**:结果 + 格式 + 效率(步数/token)+ 安全约束。

缩写对照:ORM = Outcome Reward Model(结果奖励模型,训练出来的打分器);PRM = Process Reward Model(过程奖励模型,常训成步骤打分器);GAE = Generalized Advantage Estimation(广义优势估计,§5.3 展开);TD 残差 = 单步时序差分误差 `δ_t`;`k3` = 第 3 个 token(§5.2 与动手清单里那个)。

### 4.2 多步信用分配

最终奖励 `R` 只在轨迹末尾;要把它分配给各步:

- **轨迹级**:整条 trajectory 一个 advantage(GRPO 用组内标准化,PPO 用 critic 当 baseline,见 §5.3);简单,但难定位错步。
- **turn 级**:每个 assistant turn 一个奖励(用 PRM/规则);密集,但需要步骤标签。
- **token 级**:PPO 的 token advantage;最细,但需要 value/critic 或 GAE。
**advantage 是什么**:先把同组样本的奖励标准化,`A_i = (r_i - mean(r)) / (std(r)+ε_s)`(下标 s 表示数值稳定项,常取 1e-4 量级,只为防除零,与 §2.2/§5.2 裁剪的 `ε=0.2` 无关;按 0.2 代入会得 0.714,和下面的 `A=[1,-1,1,-1]` 对不上),它表示「这条轨迹比同组平均好多少」,不是原始分数;§5.2 给的就是轨迹级这一个。
这里的「组」= 同一个任务(prompt)采出的 G 条 trajectory,G 的取值与标准化细节见 §5.2。上面三档不是三个并列算法,而是**同一个奖励往下切的粒度**:轨迹级=一条轨迹一个 `A`(GRPO 用这个,见 §5.2);turn 级=每个 assistant turn 一个 `A`(把该 turn 之后的奖励或 PRM 分数当这一 turn 的奖励,再在组内标准化);token 级=每个 token 一个 `A`,靠 critic `V(s_t)` + GAE 算(见 §5.3「GAE 展开」)。粒度越细越能定位错步,但要的信息也越多(步骤标签 / value 网络),所以一般先用轨迹级跑通。

> 面试口径:Agent RL 的难点不是“算法”,而是**奖励是否可靠 + 轨迹是否可复现 + rollout 是否够快**。

### 4.3 Reward hacking

Agent 会找捷径:跳过必要步骤、伪造工具结果、只改格式、刷单测、利用环境漏洞。防御:隐藏测试、工具权限、状态校验、多验证器、KL/预算、红队。

## 5. 多轮 GRPO / PPO:loss mask 与优势

### 5.1 多轮序列的 loss mask

一条轨迹的 token 序列通常包含:

```
system + user + assistant(tool call) + tool observation + assistant(final) + ...
```

训练时**只对 assistant 生成的 token 算 loss**;tool observation/user 输入要 mask 掉:

```
labels[tool_obs] = -100
labels[user] = -100
labels[assistant] = token id
```

![多轮轨迹的 loss mask:只学 assistant 动作](../diagrams/52c_multiturn_mask.landscape.png)

数值例:一条轨迹 3 个 assistant 段,分别 4/3/2 个 token;tool observation 共 7 个 token。按 system/user 前缀为 0 token 计(只算已列出的 token):总 token 16,有效 loss token 9;真实轨迹要把 system/user 计入总长(总数>16),但它们同样 mask、不进 loss;平均 loss 只对这 9 个算。

### 5.2 GRPO for agents

GRPO 的组内比较可以直接用在轨迹级奖励上:

```
对同一任务采 G 条 trajectory
r_i = 任务是否完成(0/1) 或 复合奖励
A_i = (r_i - mean(r)) / (std(r)+ε_s)
```

数值(`G=4`,`r=[1,0,1,0]`):`mean=0.5`,`std=0.5`,`A=[1,-1,1,-1]`。
口径要说清:这里的 `std` 是总体标准差(ddof=0);若按样本标准差(ddof=1),`std≈0.577`、`A≈[0.866,-0.866,0.866,-0.866]`,方向不变但数值不同。然后对每条轨迹的 assistant token 做 PPO 比率/裁剪 + KL。
公式里的 `A_t` 就是该 token 所属轨迹的 `A_i`:同一条轨迹的所有 assistant token 共用这一个数;要逐 token 的值就得走 §5.3 的 critic/GAE。
**这些符号补全了,这条才算得出来**(细节见 `notes/08_alignment.md §2`):

- **ratio**:`ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t)`,工程上就是 `exp(logp_new - logp_old)`,按 token 算,比的是「当前策略」与「采这条轨迹时的策略」;
- **clip**:`L_pg = -min(ρ_t·A_t, clip(ρ_t, 1-ε, 1+ε)·A_t)`,常用 `ε=0.2`;`A_t>0` 时封顶 `1+ε`,`A_t<0` 时封底 `1-ε`,截断截的就是这个权重;
- **KL**:`L = L_pg + β·KL(π_θ ‖ π_ref)`,`β` 是系数(常用 0.01–0.1 或自适应),把策略拴在参考策略附近,所以 §4.3 把它列进防御清单;`π_ref` 是**参考策略**(通常是冻结的 SFT 模型),与采样用的 `π_old` 是两个不同分布,不能默认 `π_ref=π_old`;单 token 近似 `KL≈logp_θ-logp_ref`,要用参考模型的 logprob;
- **critic / value**:`V(s_t)` 估计「从这一步往后还能拿多少奖励」;GRPO 省掉的就是这个 `V`(不训练 critic)。

![Agent GRPO:轨迹级奖励与 token 级更新](../diagrams/52d_grpo_agent.landscape.png)

**关键细节**:

- 轨迹长度不同:按 token 平均还是按轨迹平均?按 token 平均会偏爱长轨迹;要长度归一化或按轨迹加权。
- 工具 observation 的 token 不算策略动作,不能进 ratio/loss。
- 同一任务的多条轨迹要一起采样(组),否则没有组内比较。
- 全成功/全失败组优势为 0;要难度课程/过滤。

### 5.3 PPO for agents

PPO 需要 critic/value;多步任务里 value 可以按 turn 或 token 估。优势可用 GAE;但环境奖励稀疏时,critic 很难学。工程上 GRPO 更常用(省 critic),PPO 适合有密集奖励/复杂信用分配的场景。
**GAE 展开**:`δ_t = r_t + γV(s_{t+1}) - V(s_t)` 是单步 TD 残差,`A_t = Σ_k (γλ)^k δ_{t+k}`。`γ` 是折扣,`λ∈[0,1]` 调偏差-方差:`λ=0` 只用一步(偏差大、方差小),`λ=1` 退化成整段回报减 baseline(方差大)。按 turn 估 value 就是每个 turn 一个 `V`,回归目标取该 turn 之后的累计奖励;按 token 估就是每个 token 一个 `V` 再配上式。GRPO 不做这一步,组内标准化直接替代了 value。

## 6. 异步 RL 与权重同步

### 6.1 异步架构

```
rollout workers (policy v) → trajectory queue → trainer (policy v+k) → new weights → rollout workers
```

- **staleness**:rollout 用的是旧策略,训练时策略已更新;用重要性比率 `π_new/π_old` 修正,或限制队列长度/版本差。
- **权重同步**:训练器更新后,把权重广播给 rollout workers(NCCL/共享内存/对象存储);colocated(同池,定义见 §2.2)可在同一组 GPU 上换权重,disaggregated(分离池)需要跨节点传输。同步开销要隐藏在 rollout 中,否则会吃掉异步收益。
- **故障**:worker 挂掉/环境超时;要重试、丢弃坏轨迹、记录原因。

### 6.2 同步 vs 异步的取舍

- 同步:on-policy、稳定、易复现;吞吐低。
- 异步:吞吐高;但 off-policy、staleness、实现复杂。
- 常见折中:小滞后(1–2 个版本)+ 重要性裁剪 + 队列上限。

## 7. 沙箱与成本

![Agent RL 成本:rollout、环境、训练](../diagrams/52e_agent_cost.landscape.png)

### 7.1 成本构成

- **GPU rollout**:策略前向生成 token;长轨迹/高并发吃 GPU。
- **CPU/环境**:沙箱、工具、编译、浏览器;常是瓶颈。
- **训练**:PPO/GRPO 更新;通常比 rollout 短,但大模型+长序列也贵。
- **存储/网络**:轨迹、日志、权重同步。

### 7.2 账本示例(相对量级)

一条轨迹 2k token、5 个工具调用、30 s 环境时间;1k 条轨迹 = 30k s 环境时间。若 64 个并发沙箱,理论上 `30000/64≈469 s`;实际受长尾/工具限流影响更大。训练 8 卡、rollout 8 卡时,同步模式会有一侧空转;异步可把空转压下去,但需要更多 rollout 并发/队列。

### 7.3 降本

- 限制 step/token/时间;早停(任务完成/失败);
- 缓存工具结果/前缀;复用沙箱;
- 小模型 draft/蒸馏;只对困难任务用大模型;
- 按任务难度分配 rollout 预算。

## 8. 评测与安全

### 8.1 评测

- **任务成功率**:SWE-bench resolved rate、tau-bench pass^k、WebArena success。
- **效率**:步数、token、环境时间、成本。
- **轨迹质量**:工具调用正确率、错误恢复、是否绕过必要步骤。
- **鲁棒性**:工具失败/超时/空结果;对抗注入。
- **安全**:越权、沙箱逃逸、数据泄漏、危险操作。
**注意 `pass^k` 与前置清单里的 `pass@k` 是两个方向**:`pass@k` = k 次采样里**至少一次**成功,k 越大越容易;`pass^k` = k 次**独立重复全部**成功,若单次成功率 `p`,则 `pass^k ≈ p^k`(p=0.6、k=4 时只剩约 0.13),k 越大越难。所以 tau-bench 报 `pass^k` 测的是多轮稳定性/一致性,不是「能不能做对一次」;报这个数必须同时交代 k、采样温度、重试与状态重置方式,否则和 pass@k 没法比。

### 8.2 安全

- 沙箱隔离(容器/Firecracker/gVisor)、最小权限、网络白名单;
- 工具 allowlist、参数校验、速率限制;
- 危险操作人工确认;审计日志;可回滚;
- 红队:注入、越狱、工具滥用、资源耗尽。

## 9. 面试要点

1. **Agent RL 和 RLHF 的区别?** → 多步 trajectory、环境不可微、信用分配、rollout 成本。
2. **rollout 引擎是什么?** → 策略在环境里采样轨迹,存 state/action/observation/reward,供训练。
3. **同步 vs 异步?** → 同步 on-policy 但 GPU 空转;异步重叠吞吐高但有 staleness/off-policy。
4. **多轮 loss mask?** → 只对 assistant token 算 loss;tool/user token 设 -100。
5. **GRPO 怎么用于 Agent?** → 同一任务采 G 条轨迹,轨迹级奖励算组内优势,再对 assistant token 做 PPO clip+KL。
6. **多步信用分配怎么做?** → 轨迹级/ turn 级/ token 级;PRM 密集但贵,ORM 稀疏但客观。
7. **reward hacking 怎么防?** → 隐藏测试、多验证器、权限/沙箱、KL/预算、红队。
8. **权重同步?** → trainer 更新后广播给 rollout workers;队列/版本差控制 staleness。
9. **成本瓶颈?** → 环境/CPU/长尾常比 GPU 更瓶颈;并发、早停、缓存、难度分层。
10. **怎么评测 Agent?** → 任务成功率、效率、轨迹质量、鲁棒性、安全;SWE-bench/tau-bench/WebArena。

## 常见坑

- 把 Agent RL 当成“单轮 RLHF 加长”:忽略多步信用分配与环境成本。
- 把 tool observation 算进 policy loss:模型会去“预测工具输出”,不是学动作。
- 异步队列无限长:策略滞后严重,重要性修正失效。
- 只看任务成功率,不看步数/成本/安全:Agent 可能用危险捷径刷分。
- 沙箱不复用/启动慢:环境启动成为吞吐瓶颈。
- 任务不可复现:工具版本/随机性导致 reward 抖动。

## 动手清单

1. **多轮 mask**:构造一条 tool-call 轨迹,标出 assistant/tool token,算有效 loss 数;验收:tool token 全为 -100。
2. **GRPO 手算**:用 `r=[1,0,1,0]` 复算 mean/std/advantage(`mean=0.5`、`std=0.5`、`A=[1,-1,1,-1]`);再算一个 token 的目标,输入按此口径取:同一个 token 的 `logp_new=-0.30`、`logp_old=-0.35`,`ρ=exp(0.05)≈1.05`,`A=1`,`ε=0.2`(clip 区间 0.8–1.2),`β=0.01`,k3 的 KL 要用参考策略的 logprob:`KL≈logp_new-logp_ref`(题面缺 `logp_ref`,需补一个;取 `logp_ref=-0.32` 时 `KL≈0.02`),不能拿 `π_new/π_old` 的 `ρ` 代入;目标 `L≈-min(1.05,1.2)×1 + 0.01×0.02 ≈ -1.05`(策略项按奖励口径看是 `+1.05`,取负后才进 loss);验收:四个 advantage 对上 `[1,-1,1,-1]`(G=4),token 目标对上这串算式(52d 只是示意图,数值以本步给出为准,不是从图上反推)。
3. **rollout 时间账**:给定 rollout 10s/train 4s,算同步/异步吞吐;验收:解释 1.4× 与滞后。
4. **沙箱并发**:估算 1k 条轨迹、64 并发、30s/条的环境时间;验收:给出瓶颈与降本建议。

## 相关图

`52a_agent_rl_loop` · `52b_sync_async` · `52c_multiturn_mask` · `52d_grpo_agent` · `52e_agent_cost`
(均在 `diagrams/`,横竖双版本)

## 下一步

第十七层 · 数据工程:数据源/配比、MinHash/LSH 去重、质量分类器、curriculum、合成数据/self-instruct/蒸馏、许可与污染、数据飞轮。
