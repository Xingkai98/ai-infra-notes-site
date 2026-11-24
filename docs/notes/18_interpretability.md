---
title: 第十八层 · 可解释性 —— 从探针到电路与 SAE
module: 大模型基础
layer: 18
date: 2026-10-22
status: 完成
figures: [54a_probing, 54b_patching, 54c_circuit, 54d_sae]
---

# 第十八层 · 可解释性 —— 从探针到电路与 SAE

> 本层回答:我们怎么从“模型输出看起来对”走到“知道模型内部哪部分在算什么”?探针、logit lens、attention、activation patching、circuit、SAE 各自能回答什么、不能回答什么?

> **前置(按节)**
> - §1–§3:先看 `notes/02_attention.md`(attention)与 `notes/03_block.md`(残差/LayerNorm/FFN)。
> - §4–§6:先看 `notes/06_inference.md §4`(完整 forward)与 `notes/13_reasoning.md §2`(长 CoT)。
> - §7:只需基础线性代数;正文给数值例子。

> **建议路径(2 次 × 45 分钟)**
> - 第 1 次(必学):§1 行为 vs 机制 + §2 probing + §3 logit lens + §4 attention(54a)。
> - 第 2 次(必学):§5 patching(54b)+ §6 circuit(54c)+ §7 SAE/steering(54d)。

## 0. 依赖图(学习顺序)

```mermaid
graph TD
  A["模型是可分析的函数<br/>有内部激活"] --> B["探针 / logit lens<br/>读出信息"]
  C["行为实验可重复<br/>输入-输出可比较"] --> D["activation patching<br/>干预激活"]
  D --> E["circuit<br/>组件 + 边"]
  B --> E
  F["激活可稀疏分解"] --> G["SAE / steering"]
  G --> E
  E --> H["有证据的机制解释"]
```

根节点是“可以把模型当函数分析、可以做干预”;汇点是“机制解释”,不是“一张好看的热图”。

## 1. 为什么需要可解释性:行为解释 vs 机制解释

- **行为解释(behavioral)**:只描述输入-输出关系。例如“给 few-shot 例子,准确率上升”“加 CoT,数学题更好”。它有用,但不知道内部为什么。
- **机制解释(mechanistic)**:把模型拆成组件(attention head、MLP 神经元、残差方向),找出**因果路径**。例如“这个 induction head 把前一个 A 后面的 token 复制到当前位置”。

两类解释都要有**可证伪的预测**:

| 问题 | 行为证据 | 机制证据 |
|---|---|---|
| 模型会不会做指代消解? | 换句子测准确率 | 哪个 head/路径负责? |
| 去掉某组件会怎样? | 输出变化 | ablation/patching 的因果效应 |
| 这个特征是什么? | 输入输出相关 | SAE 特征 + steering 干预 |

用途:安全审计(找拒答/越狱路径)、调试(定位幻觉/复制)、科学理解(电路/特征)、模型编辑(ROME/steering)。

## 2. Probing:信息在不在?

**做法**:冻结模型;取第 `l` 层的 hidden `h_l`;训练一个**小分类器**(常用线性层)预测标签(词性、实体、情感、真值)。

```
h_l [T, d_model] → 线性探针 W [d_model, C] → logits [T, C] → softmax
```

只训练 `W`(和 bias),不更新模型。**layer-wise**:对每一层都训一个探针,看准确率随层数怎么变。

![线性探针:冻结模型,只训小分类器;右图是按层的准确率示意](../diagrams/54a_probing.landscape.png)

**数值例子**:`h=[1.0, 2.0]`,`W=[1.0, -1.0]`,`b=0`:

```
z = 1×1 + 2×(-1) + 0 = -1.0
p = sigmoid(-1.0) = 0.2689 < 0.5 → 预测类别 0
```

```python
import torch
h = torch.tensor([1.0, 2.0])
W = torch.tensor([1.0, -1.0])
b = torch.tensor(0.0)
z = W @ h + b          # -1.0
p = torch.sigmoid(z)   # 0.2689
```

**能回答**:某层是否**线性可读**出某个属性;信息出现得早还是晚。
**不能回答**:模型是否**使用**了这个信息;探针容量大时可能“自己学会”任务。

**控制**:用 control task(随机标签/打乱标签)看探针是否只是记住;用 selectivity(只对目标属性敏感)和 causal probe(把探针方向注入,看输出)补强。

> 常见误区:把探针准确率 95% 说成“模型 95% 依赖这个特征”。探针是**相关**,不是因果。

## 3. Logit lens:中间层“想说什么”

把第 `l` 层的 residual `h_l` 直接乘 unembedding `W_U`,得到词表分布:

```
logits_l = h_l · W_U        # [T, vocab]
```

看 top-k token 随层数怎么变化。**tuned lens** 会为每层学一个小变换再 unembed,缓解“中间层表示不在最终基”的问题。

**能回答**:哪一层开始出现答案候选;模型是否“早就知道”。
**不能回答**:这个候选是否被后续层使用;中间层表示与最终 unembedding 的基不一定对齐。

> 与 13 的衔接:CoT/推理模型里,logit lens 可看“中间答案”是否在 CoT 之前出现;但**看到候选 ≠ 证明它是推理依据**。

## 4. Attention analysis:权重只是线索

常见做法:

- 看 attention 热力图:行=query,列=key;
- head ablation:去掉一个 head,看任务指标;
- attention rollout:把多层 attention 复合,追踪信息流;
- 找 induction head:`A B … A → B` 的复制模式。

![Circuit 与注意力模式:热图只是线索,要用 ablation/patching 验证](../diagrams/54c_circuit.landscape.png)

**关键警告**:

- attention 权重高 **≠** 因果重要(Jain & Wallace 2019 等)。
- 注意力权重是“混合比例”,不是“信息量”;FFN/残差也在写信息。
- 存在 attention sink、冗余 head、backup head;单个 head 去掉可能被补偿。

**能回答**:模型在哪些位置/头上有可读的模式。
**不能回答**:这个 head 是否必要、是否唯一。

## 5. Activation patching / causal tracing:做干预

**固定一对 prompt**:

- clean:`The Eiffel Tower is in` → 正确答案 Paris,logit 4.0;
- corrupt:`The Colosseum is in` → Paris logit 1.0。

把 clean 在 `(层 l, 位置 pos)` 的 hidden state **替换**进 corrupt 的前向,得到 patched logit。用**恢复率**衡量:

```
recovery = (logit_patched - logit_corrupt) / (logit_clean - logit_corrupt)
         = (3.2 - 1.0) / (4.0 - 1.0) = 0.733
```

![Activation patching:只替换一个 (层, 位置),看 logit 恢复多少](../diagrams/54b_patching.landscape.png)

**读法**:补回 73% 的 logit 差 → 这个 activation 携带关键信息。但它仍可能是**多条路径之一**;要做 head/neuron 级 patching、noising(把 clean 的激活换成 corrupt)和 path patching(只允许特定边传递)。

**代码骨架**(库名仅示意,不同框架 API 不同):

```python
clean_cache = model.run_with_cache(clean_ids)
def patch_pos(act, hook):
    act[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return act
patched_logits = model.run_with_hooks(
    corrupt_ids,
    fwd_hooks=[(f"blocks.{l}.hook_resid_post", patch_pos)],
)
```

**能回答**:哪一层/哪个位置对输出有因果影响;信息在哪一步被写入。
**不能回答**:所有交互路径(需要 path patching/电路分析);分布外是否仍成立。

## 6. Circuit:组件 + 边 + 因果实验

一个 circuit 是**子图**:

- 节点:attention head、MLP 神经元/方向、residual stream 位置;
- 边:QK 组合(query 从某处读)、OV 写回、MLP 读出/写入;
- 验证:ablation、patching、path patching,而不是只看权重。

典型例子(IOI,Indirect Object Identification,arXiv:2211.00593 Fig.2):`A B … A → B`。一个简化路径:

```
L2 previous-token head:看前一个 token(L2 不是 induction head)
L5–6 induction head:匹配“前一个 A 后面的 token”
L7–8 S-inhibition head:压低竞争名字
L9–10 name mover head:把 B 写入 residual,提升 B 的 logit
```

> 层号按论文图示;不同模型/任务会有偏移。图 54c 是“简化 IOI 路径”示意,不是唯一电路。

**能回答**:哪些组件/边足以复现行为;是否有 backup。
**不能回答**:circuit 是否完整、是否唯一;不同 prompt/模型可能换路径。

> 面试口径:circuit 不是“画出 attention 热图”,而是“提出组件假设 + 做因果干预 + 检查反例”。

## 7. SAE 与 steering:把激活拆成稀疏特征

**动机**:单个神经元常是**多义**的(polysemantic)。一个关键解释是 **superposition**:模型要表示的特征数可能**远大于**神经元/维度数;在特征稀疏激活的假设下,多个特征可以叠加在同一个激活方向里,单个神经元因此看起来“什么都管”。SAE 用**过完备字典**(`d_feature >> d_model`)+ 稀疏约束,试图把叠加的特征拆开;但拆开不保证每个特征单义,也不保证覆盖全部概念。

```
f = ReLU(W_e (x - b_dec) + b_e)      # 编码,得到稀疏特征
x_hat = W_d f + b_dec                 # 解码/重建
L = ||x - x_hat||² + λ ||f||_1        # 重建 + 稀疏(L1)
```

`W_d = W_e^T` 在部分实现里用作**转置初始化/参数绑定**;常见实现把编码 `W_e` 与解码 `W_d` **分开训练**(绑定只是可选的省参/正则)。稀疏项除了 L1,也有 L0/其他稀疏惩罚;λ 控制“重建 vs 稀疏”的权衡。

**玩具数值**(`W_e = W_d = I`,只保留正激活):

```
x = [0.8, -0.2, 0.5, -0.1]
f = ReLU(x) = [0.8, 0, 0.5, 0]        # L0 = 2
x_hat = [0.8, 0, 0.5, 0]
e = x - x_hat = [0, -0.2, 0, -0.1]
||e||_2 = sqrt(0.2² + 0.1²) = 0.224
```

![SAE:编码 → 稀疏特征 → 解码;玩具例子用单位字典展示稀疏与重建](../diagrams/54d_sae.landscape.png)

**Steering**:找到某个特征方向 `d_i`,在推理时加:

```
x' = x + α · d_i
```

看输出是否按预期变化。**能回答**:SAE 给“特征是否可分离”;steering/ablation 才能给“是否因果可控制”。
**不能回答**:特征是否“单义”、是否覆盖所有概念;SAE 会有 feature splitting/absorption,steering 可能副作用。

**代码骨架**:

```python
# 这里 W_e 存成 [d_model, d_feature];若存成 [d_feature, d_model] 则写 W_e @ (x-b_dec)
f = torch.relu((x - b_dec) @ W_e.T + b_e)
x_hat = f @ W_d + b_dec
loss = ((x - x_hat) ** 2).sum(-1).mean() + lam * f.abs().sum(-1).mean()
```

## 8. 方法对比:能回答什么、不能回答什么

| 方法 | 能回答 | 不能回答 | 主要风险 |
|---|---|---|---|
| Linear probe | 某层是否有线性可读信息 | 模型是否使用 | 探针容量/虚假相关 |
| Logit lens | 中间层答案候选 | 候选是否被使用 | 基不对齐 |
| Attention analysis | 模式/头的位置 | 因果必要性 | 权重≠解释 |
| Patching | 某层/位置的因果效应 | 全路径/分布外 | 单点干预、补偿 |
| Circuit | 组件+边的机制假设 | 完整性/唯一性 | 只验证了部分路径 |
| SAE | 稀疏特征/可 steering 假设 | 单义性/全覆盖/因果作用(需 steering 验证) | 特征分裂/吸收 |

**一句话**:probe/logit lens/attention 主要给**相关/读出**证据;activation patching/circuit 通过**干预**给因果证据;SAE 是**表示分析**,特征是否因果地控制行为要靠 steering/ablation 验证。方法越靠因果,成本越高;好的解释是**多种证据交叉**。

## 9. 与 01–06、13 的衔接

- **01–06**:residual stream 是“共享黑板”;attention head 读/写黑板;FFN 可看作 key-value memory。可解释性把这些结构变成可检验的假设。
- **13**:CoT/推理模型给我们“行为轨迹”;可解释性问“这些中间步骤是否真的参与计算”。两者结合:行为 + 机制,而不是只看 CoT 文本。
- **安全**:找拒答/越狱/欺骗的路径;但**不要**把一次 circuit/SAE 结果当成完整证明。

## 面试要点

1. **行为解释和机制解释的区别?** → 行为只描述输入输出;机制给组件/因果路径。
2. **linear probe 能证明什么?** → 某层线性可读出属性;不能证明模型使用它。
3. **logit lens 是什么?** → 中间层 residual 直接乘 unembedding 看词表分布;注意基不对齐。
4. **attention 权重能当解释吗?** → 不能单独;权重≠因果,要 ablation/patching。
5. **activation patching 怎么算恢复率?** → `(patched-corrupt)/(clean-corrupt)`;数值例 0.733。
6. **circuit 是什么?** → 组件+边+因果实验;不是热图,也不保证完整/唯一。
7. **SAE 的损失?** → 重建误差 + L1 稀疏;`f=ReLU(W_e(x-b)+b_e)`,`x_hat=W_d f+b`。
8. **SAE 特征一定单义吗?** → 不;有 feature splitting/absorption,多义神经元问题仍在。
9. **steering 怎么做?** → `x'=x+αd`;要测副作用与分布外。
10. **为什么相关≠因果?** → 探针/热图只给相关;patching/ablation 才是干预。

## 常见坑

- 把探针准确率当成“模型依赖该特征”。
- 把 attention 权重当成“解释”;忽略 FFN/残差/backup。
- 只做单点 patching,不做 path patching/多种子,误判唯一路径。
- 把 SAE 特征名当“真概念”;特征名是人给的标签,不是模型内部的唯一语义。
- 把 circuit 说成“完整机制”;论文通常只验证一部分路径。
- 在分布外/对抗样本上直接外推机制结论。

## 动手清单

1. **探针**:取一个小模型某层 hidden,训线性探针预测词性;验收:报告 layer-wise 准确率、control task、训练/验证差距。
2. **Patching**:构造 clean/corrupt prompt 对,替换一层一个位置,复算 recovery;验收:画出层×位置的恢复热图,并做至少一个 head ablation。
3. **SAE/steering**:用开源 SAE 或玩具字典,找一个特征方向做 `x+αd`;验收:报告 α 扫描、输出变化与副作用。

## 相关图

`54a_probing` · `54b_patching` · `54c_circuit` · `54d_sae`(均在 `diagrams/`,横竖双版本)

## 下一步

第十九层 · 多模态深化:VLM 训练目标、视觉 tokenizer、跨模态融合、视频/音频、扩散与评测安全——把 12 的概览推进到可推导、可复算的深度。
