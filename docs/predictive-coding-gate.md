# 预测编码记忆门控：理论依据与实现

> Memtide 的写入管线在"抽取"与"冲突消解"之间插了一道**门控**（`memtide/gating.py`）：
> 不是所有抽取出的事实都值得存。只有当新事实相对已有记忆先验产生**预测误差**
> （prediction error / surprise）时，才触发编码。

## 一、理论脉络

**预测编码（Predictive Coding）** 认为大脑是一个不断对世界做预测的生成模型，
感知与学习的驱动力不是刺激本身，而是*预测与实际的差值*：

- **Rao & Ballard (1999)**, *Predictive coding in the visual cortex*, Nature
  Neuroscience — 预测编码的奠基性计算模型：皮层只传播预测残差（误差信号）。
- **Friston (2005, 2010)**, 自由能原理 — 把预测误差最小化统一为学习与感知的目标。

**预测误差门控记忆编码** 是认知科学的成熟结论：

- **Ranganath & Rainer (2003)**, *Neural mechanisms for detecting and
  remembering novel events*, Nature Reviews Neuroscience — 新异事件检测与记忆
  编码共享神经机制。
- **van Kesteren et al. (2012)**, *How schema and novelty augment memory
  formation*, Trends in Neurosciences — 双通路模型（MRM）：与图式（schema）
  一致的信息走整合通路巩固图式，图式外的新异信息走情景通路被强烈编码。
- **Quent, Henson & Greve (2021)** — 对"预测误差增强记忆"各种理论模型的系统比较。
- **Pupillo et al. (2023)**, *The effect of prediction error on episodic memory
  encoding*, npj Science of Learning — 用计算模型导出的预测误差预测后续记忆成绩。
- **Huang et al. (2025)**, *Accurate predictions facilitate robust memory
  encoding* — 准确预测与预测误差在编码中的不同角色。

**工程先例**：

- **Itti & Baldi (2009)**, *Bayesian surprise attracts human attention*, Vision
  Research — 把 surprise 形式化为 KL(后验‖先验)，本实现用其一阶近似
  （Shannon 自信息，见下）。
- **Schmidhuber (2010)**, *Formal theory of creativity* — 压缩进度/惊喜度作为
  内在动机。
- **Park et al. (2023), Generative Agents** — 用 LLM 打"重要性分"过滤写入，
  是同一思想的工程化先例。
- **Surprise-Gated Robot Episodic Memory (arXiv 2606.03787)** — 机器人情景记忆
  按 surprise 门控写入，与 Memtide 的设计几乎同构。

## 二、Memtide 的实现

### 1. 先验 = 记忆库

已有记忆的向量集合就是 agent 的生成式先验。新事实 f 的被预测程度用
**核密度估计**：

```
p_hat(f) = max_sim²          max_sim = max_m cos(v_f, v_m)   （对所有有效记忆）
```

平方核（平方指数型）把相似度映射到 [0,1] 的概率空间，0.02 的下限保证对完全
未见的事实 log 有限（surprise 饱和约 11.3 bits）。

### 2. 预测误差 = 自信息

```
S(f) = -log2 p_hat(f)        单位：bits
```

这是 **Bayesian surprise 的退化情形**：对单点观测，KL(后验‖先验) 退化为
自信息。比完整 KL 便宜一个数量级，且单调性一致。

### 3. 三分流（对应 van Kesteren 双通路 + 容量节省）

| 预测误差 | 决策 | 依据 |
|---|---|---|
| `S ≤ 0.5 bits`（p̂ ≥ 0.71） | **REJECT** 不编码 | 完全被先验预测到 → 无误差信号 → 编码是浪费容量 |
| `0.5 < S < 2.5` | **INTEGRATE** 正常写入 | 图式一致信息 → 整合进先验，精化记忆 |
| `S ≥ 2.5 bits` | **NOVEL** 写入 + 重要度 +0.1 | 图式外新异信息 → 强情景编码（间隔效应也偏向它） |
| 同义 slot 单值冲突（规范化后相同 + cos ≥ `gate_slot_floor` 0.40） | **强制编码**（volatile-update） | 搬家/换工作本身就是记忆系统存在的意义——先验失效的位置恰是必须更新的位置；slot 为开放 hint，经别名归一判定（city == location），多值/时间限定事实不受此分支摆布，交给 LLM 消解决定 |

阈值映射回相似度：REJECT 线 ≈ cos 0.84，NOVEL 线 ≈ cos 0.42。四个阈值都在
`MemoryConfig`（`gate_*` 字段），门控可通过 `gate_enabled=False` 整体关闭。

### 4. 可审计性

每次门控决策随 `AddResult.gate` 返回，并持久化到记忆的 `metadata`：

```json
{"gate": "novel", "surprise_bits": 3.47, "slot": "preference"}
```

即"这条记忆为什么被存、当时它有多意外"永远可回溯——与 Memtide 的 Zep 式审计
日志同一哲学。

## 三、与去重的分工

预测编码门控不是替代冲突消解，而是**把二值去重升级为连续的容量分配**：

- 完全重复（cos ≈ 1.0）→ 门控直接拦下（REJECT），同时计一次 NOOP，行为与
  原来 Mem0 式去重一致；
- 近重复但超相同（0.84 < cos < 0.995）→ INTEGRATE 放行 → 消解阶段决定
  ADD/UPDATE；
- 真正的新信息 → 进入编码，且越意外记得越牢。

一个自然的推论：**记忆库越满，门控越严**。空库时一切事实都是高惊喜（先验
为空 → 全部编码），随着先验成型，只有真正偏离模型的新信息才能写入——这与
生物记忆的发育轨迹一致。
