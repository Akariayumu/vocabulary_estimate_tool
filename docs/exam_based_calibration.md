# 考纲词汇锚点校准方案

> 用中国英语考试大纲词表作为难度锚点，优化 bucket matrix 模型的认知率预测精度，解决"频率≠难度"的核心问题。

---

## 目录

1. [问题陈述](#1-问题陈述)
2. [考纲词汇收集方案](#2-考纲词汇收集方案)
3. [考纲锚点校准策略](#3-考纲锚点校准策略)
4. [损失函数设计](#4-损失函数设计)
5. [实施路线图](#5-实施路线图)
6. [与现有 bucket matrix 模型的关系](#6-与现有-bucket-matrix-模型的关系)
7. [可行性分析与预期改进](#7-可行性分析与预期改进)
8. [附录：数学推导](#8-附录数学推导)

---

## 1. 问题陈述

### 1.1 频率 ≠ 难度

当前 bucket matrix 模型的核心假设是 COCA 词频 rank 可以代表词汇难度：rank 越高的词越难，学习者应该按 rank 顺序掌握词汇。

然而，真实语言学习并非如此规律。以 **CET-6 考纲词汇**为例：

| 指标 | 数值 |
|------|------|
| 考纲词总数（在词库中） | 7,520 词 |
| rank 最小值 | 1（the） |
| rank 最大值 | 29,996（subregionalize） |
| **rank 中位数** | **6,487** |
| rank 范围 | 1 — 29,996 |

这意味着 CET-6 考纲词汇覆盖了从最常用词（rank=1）到极低频词（rank≈30,000）的**整个频谱**。一个词汇量 6,000 的学习者，按照"rank 排序"的假设，应该不认识 rank > 6000 的词而认识 rank ≤ 6000 的词。但实际情况是：

- 该学习者很可能**不认识**许多 rank ≤ 6000 的陌生低频词（如属于 1k-5k 桶的罕见专业词）
- 该学习者很可能**认识**许多 rank > 6000 的 CET-6 考纲词汇（因为背过）
- 高频功能词（the, a, is）几乎人人都认识，但某些 rank 3000-5000 的词（如 tariff, warranty, incest）可能完全陌生

### 1.2 现有模型的局限性

当前 bucket matrix 模型在合成数据上训练得到，其核心假设是：
```
学习者词汇量 = 「按 rank 顺序填充已知词集」的词汇量
```

这种假设在合成数据生成时是合理的（便于生成训练标签），但在真实学习场景中偏离真实。具体表现：

| 偏差类型 | 描述 | 影响 |
|---------|------|------|
| **rank 偏差** | 低频考纲词被低估认识率 | CET-6 级用户被低估 |
| **专业词汇偏差** | 高频专业词被高估认识率 | 非英语专业用户被高估 |
| **间隔效应** | 低频用户可能认识一些高频生僻词 | 词汇量估计方差偏大 |
| **过拟合合成数据** | 模型在合成用户上准确，但真实用户可能有不同认知模式 | 泛化能力未知 |

### 1.3 考纲锚点的核心思路

解决上述问题的方案是引入**考纲词汇作为校准锚点**：

```
核心洞察：
  考纲词汇集合 = 有意义的「难度锚点」
  CET-4 约 4,500 词 → 代表 4,000-5,000 词汇量水平
  CET-6 约 6,000 词 → 代表 6,000-7,000 词汇量水平
      ...
  考纲词表在 rank 空间中的分布提供了「什么是这个水平该认识的词」的真实约束
```

**校准目标**：调整模型参数，使得：
- "CET-4 水平"的预测学习者 → CET-4 词表覆盖率达到 80-85%
- "CET-6 水平"的预测学习者 → CET-6 词表覆盖率达到 75-80%
- 各水平过渡时，考纲词覆盖率曲线平滑且单调

### 1.4 改进目标

| 维度 | 当前（合成数据训练） | 目标（加考纲锚点后） |
|------|-------------------|--------------------|
| 训练数据 | 合成虚拟用户 ×12 | 合成数据 + 考纲锚点约束 |
| 损失函数 | 桶级 MSE | 桶级 MSE + 考纲覆盖率 MSE + 平滑正则 |
| rank 假设 | 严格 rank 顺序 | 考纲词在 rank 空间中的实际分布 |
| 对 CET-6 用户预测 | 可能低估 | 用锚点校正 |
| 对高频生僻词 | 可能高估 | 用锚点约束 |
| 可解释性 | 黑箱参数 | 每个考纲锚点提供直观校验 |

---

## 2. 考纲词汇收集方案

### 2.1 当前已有词表

以下词表已存在于 `data/exam_vocab/` 目录中：

| 词表文件 | 原始词数 | 在词库中匹配 | 匹配率 | rank 中位数 | 说明 |
|---------|:-------:|:----------:|:-----:|:-----------:|------|
| `gaokao.txt` | 3,469 | 3,370 | 97% | 3,065 | 高考考纲词汇（含中考基础词） |
| `cet6.txt` | 8,028 | 7,520 | 94% | 6,487 | 六级考纲词汇（含四级词） |
| `toefl.txt` | 3,469 | 2,502 | 72% | 12,416 | TOEFL 高频词汇 |
| `gre.txt` | 6,677 | 3,335 | 50% | 15,660 | GRE 词汇 |
| `coca20000.txt` | 20,199 | — | — | — | COCA 频率排序（非考纲词表） |

**问题**：GRE 和 TOEFL 匹配率较低（50-72%），因为这些词表中包含大量不在 COCA 前 30k 的极低频词。

### 2.2 当前缺少的词表

以下词表是校准锚点体系的关键缺失：

| 词表 | 预期词汇量 | 在词库中预期匹配数 | 获取难度 | 优先级 |
|------|:---------:|:----------------:|:--------:|:-----:|
| **中考** | ~1,500-2,500 | ~1,500-2,400 | ⭐ 高（公开） | **P0** |
| **CET-4 独立词表** | ~4,000-5,000 | ~3,500-4,500 | ⭐⭐ 需处理（从 CET-6 剥离） | **P0** |
| **考研** | ~5,500-6,000 | ~4,500-5,500 | ⭐ 高（公开） | **P0** |
| **专八 TEM-8** | ~10,000-13,000 | ~8,000-10,000 | ⭐⭐ 中等 | P1 |
| **IELTS** | ~6,000-7,000 | ~5,000-6,000 | ⭐⭐ 中等 | P1 |

其中 **P0** 词表对校准体系至关重要，应在 Phase 1 完成收集。

### 2.3 词表收集方法

#### 2.3.1 方法一：官方大纲提取（推荐）

中国英语考试均有官方《考试大纲》或《词汇表》，以 PDF/Word 格式发布。示例来源：

| 词表 | 典型来源 | 提取方法 |
|------|---------|---------|
| 中考 | 《义务教育英语课程标准》附录 | 网页/PDF 爬取 → 正则提取 |
| 考研 | 全国硕士研究生招生考试英语(一/二)大纲 | PDF 解析 → 表格提取 |
| 专八 | 《英语专业八级考试大纲》词汇表 | 网上已有整理版 → 对比验证 |
| IELTS | Cambridge IELTS 词汇表、官方 Vocabulary List | 开源数据集 |

提取流程：

```python
def extract_exam_vocab(source_path: str, output_path: str) -> set[str]:
    """
    从原始来源提取考纲词汇。
    
    对不同格式的支持：
    - .txt: 直接读取，word per line
    - .pdf: pdfplumber/docling 解析表格
    - .docx: python-docx 提取段落 + 表格
    - .html: BeautifulSoup 提取
    
    输出：去重、小写化的词集
    """
    # ... 实现细节
    pass
```

#### 2.3.2 方法二：从现有词表推导

对于某些难以获取官方大纲的词表，可以从已有词表中推导：

**CET-4 独立词表从 CET-6 推导**：

```
CET-4 ≈ CET-6 中 rank < 4000 且不在高考词表中的词
      + 高频 CET-6 词（rank < 3000 且属于典型四级难度）
```

操作步骤：

```python
def derive_cet4_from_cet6(cet6_words: set[str], gaokao_words: set[str], 
                           bank: VocabBank) -> set[str]:
    """
    从 CET-6 词表中推导 CET-4 独立词表。
    
    策略：
    1. 剔除高考词（已在高考中覆盖）
    2. 保留 rank < 4000 的词
    3. 补充 rank ∈ [4000, 6000] 中属于典型四级难度的词
    """
    derived = set()
    for word in cet6_words:
        if word in gaokao_words:
            continue
        rank = bank.get_rank(word)
        if rank is not None and rank < 4000:
            derived.add(word)
        # 补充策略...
    return derived
```

#### 2.3.3 方法三：自动难度标注

对于确定属于某考试等级但未在官方词表中的词，可用 NLP 方法自动标注：

```
难度标注流水线：
  词 + 上下文：
    │
    ▼
  特征提取：rank, 词长, 音节数, COCA 频率, CEFR 等级 (if available)
    │
    ▼
  分类器 / 规则引擎：
    - rank < 1500 → 中考概率高
    - rank ∈ [1500, 2500] → 高考概率高
    - rank ∈ [2500, 4000] → CET-4 概率高
    - rank ∈ [4000, 6000] → CET-6 概率高
    - rank > 6000 + 学术词 → 考研/TEM-4 概率高
    │
    ▼
  输出：词 → 考试等级推断
```

但自动标注仅作为补充手段，**不替代官方大纲词表**。

### 2.4 词表格式规范

所有考纲词表统一为每行一个词（小写），UTF-8 编码，放置在 `data/exam_vocab/` 目录。

```text
data/exam_vocab/
├── gaokao.txt          # 高考（已有）
├── cet6.txt            # 六级（已有，含四级）
├── toefl.txt           # TOEFL（已有）
├── gre.txt             # GRE（已有）
├── coca20000.txt       # COCA 排序（已有）
├── zhongkao.txt        # 中考（待收集）← P0
├── cet4.txt            # CET-4 独立词表（待推导）← P0
├── kaoyan.txt          # 考研（待收集）← P0
├── tem8.txt            # 专八（待收集）← P1
└── ielts.txt           # IELTS（待收集）← P1
```

### 2.5 词表质量确认

每个词表收集后，需进行以下验证：

1. **去重检查**：与已有词表对比，输出重叠率
2. **词库匹配率**：应在词库中匹配 85%+ 的词
3. **rank 分布覆盖**：确认 rank 分布与预期范围一致
4. **人工抽样审核**：随机抽 50 词，人工判断是否属于该考试等级

```python
def validate_exam_vocab(path: str, bank: VocabBank, 
                        expected_rank_range: tuple[int, int]) -> dict:
    """验证考纲词表质量。"""
    with open(path) as f:
        words = {w.strip().lower() for w in f if w.strip()}
    
    matched = 0
    ranks = []
    for w in words:
        r = bank.get_rank(w)
        if r is not None:
            matched += 1
            ranks.append(r)
    
    ranks.sort()
    in_range = sum(1 for r in ranks 
                   if expected_rank_range[0] <= r <= expected_rank_range[1])
    
    return {
        "total_words": len(words),
        "matched": matched,
        "match_rate": matched / len(words),
        "rank_median": ranks[len(ranks)//2] if ranks else None,
        "rank_min": min(ranks) if ranks else None,
        "rank_max": max(ranks) if ranks else None,
        "in_expected_range": f"{in_range}/{len(ranks)} ({100*in_range/len(ranks):.0f}%)",
    }
```

---

## 3. 考纲锚点校准策略

### 3.1 核心思想：标准学习者画像

引入考纲词表后，我们可以对每个能力水平构建 **"标准学习者画像"**——即该水平的学习者应该认识的词集：

```
水平 L 的「标准学习者」应该认识「考试等级 L」考纲词表中的 x% 的词。
```

具体锚点设定（基于中国教育体系的实际经验）：

| 学习者水平 | 对应词汇量 | 锚点考纲词表 | 预期覆盖率 | 权重 |
|-----------|:---------:|------------|:---------:|:---:|
| 初中 | ~2,000 | 中考 | 85% | 3x |
| 高中 | ~3,500 | 高考 | 82% | 3x |
| 四级 | ~4,800 | CET-4 | 80% | 3x |
| 六级 | ~6,500 | CET-6 | 75% | 3x |
| 考研 | ~8,000 | 考研 | 70% | 2x |
| 专八 | ~12,000 | TEM-8 | 65% | 2x |

> **预期覆盖率**不是 100%，因为：
> - 考纲词表包含部分超纲词（如 CET-6 中有 rank > 20000 的极低频词）
> - 即使在该水平，个别低频考纲词不认识是正常的
> - 覆盖率递减体现了"词汇量增长边际递减"的现实

### 3.2 锚点约束转化为模型约束

对于给定的 bucket matrix 参数 (θ₁, θ₂, ..., θ₉, γ)，我们可以计算一个学习者在考纲词表上的**预测覆盖率**：

```
对于考试等级 E 的考纲词表 V_E：

预测覆盖率 P_E = (1/|V_E|) × Σ_{w ∈ V_E} sigmoid(θ_bucket(w) + γ)

其中 γ 是该学习者的能力偏移参数。
```

校准目标：**对于对应水平的学习者，P_E 应接近预期覆盖率**。

这转化为以下约束条件：

```
对于词汇量 ≈ expected_vocab_size(E) 的学习者：
  P_E ≈ expected_coverage(E)
  
  约束类型：软约束（通过损失函数项实现）
  约束强度：权重 2x-5x（取决于锚点可靠性）
```

### 3.3 考纲覆盖率的认知率映射

对于不同词汇量水平，考纲词表覆盖率的递减曲线应该平滑且单调：

```
覆盖率的期望形状：

1.0 │
    │                  ┌── 中考词表（最简单）
    │                ╱ 
    │              ╱    ┌── CET-4 词表
    │            ╱    ╱
    │          ╱    ╱     ┌── CET-6 词表
    │        ╱    ╱    ╱
    │      ╱    ╱    ╱    ...
    │    ╱    ╱    ╱    ╱
    │  ╱    ╱    ╱    ╱
  0 └─────────────────────────
    2k   4k   6k   8k   10k...
         expected vocabulary size
```

每个考纲词表对应一个锚点：(expected_vocab_size, expected_coverage)

### 3.4 多锚点联合约束

对于一组考纲词表，训练时的联合约束为：

```
L_official = Σ_{E ∈ {中考, 高考, CET-4, CET-6, ...}} 
                  w_E × (P_E - target_E)²
```

其中：
- `P_E` = 使用当前模型参数预测的考纲词表 E 覆盖率
- `target_E` = 考纲词表 E 的预期覆盖率
- `w_E` = 每个词表的训练权重

这种做法的核心优势：
1. **覆盖全频谱**：从中考到专八，每个水平段都有锚点
2. **天然平衡**：低频词多的考纲（TEM-8）不会因为难以匹配就被忽略
3. **可解释性强**：可以直接检查每个锚点的预测覆盖率

### 3.5 在 bucket matrix 中集成锚点

Bucket matrix 模型中，词 w 的认知概率为：

```
P(known|w) = sigmoid(θ_bucket(w) + γ)
```

其中 θ_bucket(w) 是词 w 所在桶的参数，γ 是学习者偏移。

对于考纲词表 E 中的所有词：

```
P_E = (1/|V_E|) × Σ_{w ∈ V_E} sigmoid(θ_bucket(w) + γ)
```

训练时，P_E 作为损失函数中的一个约束项。

### 3.6 锚点与批次训练

在 Phase 2 的正式训练中，锚点约束可以按以下方式集成：

```python
def compute_official_vocab_loss(
    thetas: dict[str, float],      # 9 个 θ 参数
    gamma: float,                   # 学习者偏移
    exam_vocab_sets: dict,         # 考纲词表 {name: {words, expected_coverage, weight}}
    bank: VocabBank,               # 词汇库（用于查桶归属）
) -> float:
    """
    计算考纲锚点损失项。
    
    对于每个考纲词表，计算该词表在模型下的预测覆盖率，
    与预期覆盖率的加权 MSE。
    """
    total = 0.0
    
    for name, info in exam_vocab_sets.items():
        words = info["words"]
        expected = info["expected_coverage"]
        weight = info["weight"]
        
        if not words:
            continue
        
        # 计算每个词的认知概率
        probs = []
        for word in words:
            bucket = bank.get_bucket(word)
            if bucket and bucket in thetas:
                p = sigmoid(thetas[bucket] + gamma)
                probs.append(p)
        
        if not probs:
            continue
        
        # 预测覆盖率
        coverage = sum(probs) / len(probs)
        
        # 加权 MSE
        total += weight * (coverage - expected) ** 2
    
    return total
```

---

## 4. 损失函数设计

### 4.1 完整的损失体系

模型训练的完整损失函数为多项目标联合优化：

```
L = λ_bucket × L_bucket
  + λ_interval × L_interval        （可选的间隔组损失）
  + λ_official × L_official         （考纲锚点损失，本方案核心）
  + λ_smooth × L_smooth             （平滑正则化）
  + λ_l2 × L_l2                     （参数正则化）
```

| 损失项 | 符号 | 默认权重 | 说明 |
|-------|:----:|:-------:|------|
| 桶级 MSE | L_bucket | 1.0 | 原始 bucket matrix 损失 |
| 间隔组 MSE | L_interval | 0.3 | 间隔组认知率一致性 |
| **考纲锚点 MSE** | **L_official** | **0.5-1.0** | **本方案核心，初期低权重，后期逐步增加** |
| 平滑正则 | L_smooth | 0.01 | 认知率递减曲线平滑 |
| L2 正则 | L_l2 | 0.001 | 参数正则化，防止过拟合 |

### 4.2 损失项详解

#### 4.2.1 桶级 MSE（L_bucket）

对每个用户 j 和每个桶 b，计算预测认知率与观测认知率的 MSE：

```
L_bucket = Σ_j Σ_b w_b × (p_jb - r_jb)²

其中：
  p_jb = sigmoid(θ_b + γ_j)
  r_jb = 桶 b 的观测认知率（Beta 平滑后）
  w_b = 桶大小 / 总词数（大桶更多权重）
```

这是原有训练管线的核心损失。

#### 4.2.2 考纲锚点 MSE（L_official）← 新

```
L_official = Σ_E w_E × (P_E - target_E)²

其中：
  P_E = (1/|V_E|) × Σ_{w ∈ V_E} sigmoid(θ_bucket(w) + γ)
  target_E = 考纲词表 E 的预期覆盖率
  w_E = 每个考纲词表的权重（2x-5x）
```

对这个损失的梯度：

```
∂L_official / ∂θ_b = 2 × w_E × (P_E - target_E) × (∂P_E/∂θ_b)

∂P_E / ∂θ_b = (1/|V_E|) × Σ_{w ∈ V_E ∩ bucket_b} 
                    sigmoid(θ_b + γ) × (1 - sigmoid(θ_b + γ))
```

**关键行为**：如果某个考纲词表的预测覆盖率低于目标，梯度会推动 θ 增加（让该桶的认知率更高），反之亦然。

#### 4.2.3 平滑正则化（L_smooth）

确保了不同桶的 θ 参数之间保持合理的递减关系：

```
L_smooth = λ_s × Σ_{i=1}^{8} (sigmoid(θ_i + γ) - sigmoid(θ_{i+1} + γ))²        
```

或者，在认知率空间更直接：

```
L_smooth = λ_s × Σ_{b递增} max(0, P(known|bucket_b） - P(known|bucket_{b-1}) + ε)²
           + λ_s × Σ |(P_b - P_{b+1}) - (P_{b+1} - P_{b+2})|²
```

第一项是保序约束（高频桶认知率应该高于低频桶），第二项是曲率约束（递减速度变化应该平滑）。

### 4.3 训练中的锚点注入方式

#### 方式 A：插值法（推荐，Phase 1 使用）

在合成数据生成阶段，在标准训练数据中**插入**考纲锚点数据点：

```python
def inject_exam_anchors(synthetic_dataset, exam_vocab_dict, bank):
    """
    在合成数据中插入考纲锚点数据。
    
    对于每个合成用户，如果其词汇量接近某个考纲等级，则
    额外生成该考纲词表的覆盖率作为训练目标。
    """
    augmented = []
    for user_data in synthetic_dataset:
        vocab_size = user_data["vocab_size"]
        
        # 找到最接近的考纲等级
        anchor = find_nearest_exam_level(vocab_size, exam_levels)
        if anchor:
            # 为该用户生成考纲词覆盖率标签
            coverage_label = compute_expected_coverage(
                vocab_size, anchor
            )
            user_data["exam_anchors"] = {
                anchor.name: coverage_label
            }
        
        augmented.append(user_data)
    
    return augmented
```

#### 方式 B：显式损失函数项（推荐，Phase 2 使用）

在梯度下降循环中直接计算考纲覆盖率损失：

```python
for epoch in range(n_epochs):
    for batch in data_loader:
        # 1. 桶级 MSE
        loss_bucket = compute_bucket_mse(batch, thetas, gammas)
        
        # 2. 考纲覆盖率 MSE
        loss_official = 0.0
        for user in batch:
            gamma = user_gammas[user.id]
            for exam_name, exam_info in exam_vocab_sets.items():
                coverage = predict_exam_coverage(
                    thetas, gamma, exam_info, bank
                )
                target = exam_info.expected_coverage
                loss_official += (
                    exam_info.weight * (coverage - target) ** 2
                )
        
        # 3. 平滑正则
        loss_smooth = compute_smoothness(thetas)
        
        # 总损失
        loss = (lambda_b * loss_bucket 
                + lambda_o * loss_official 
                + lambda_s * loss_smooth)
        
        loss.backward()
        optimizer.step()
```

#### 方式 C：先正则后微调（稳健方案）

**Phase A**：仅用桶级 MSE 训练（不做任何锚点约束），得到基准参数。

```python
# Phase A: 基线训练
for epoch in range(1000):
    loss = compute_bucket_mse(...)
    loss.backward()
    optimizer.step()
```

**Phase B**：在基线基础上加入锚点约束，使用较小的学习率微调。

```python
# Phase B: 锚点微调
optimizer = Adam(lr=lr * 0.1)  # 更小的学习率
for epoch in range(500):
    loss = (lambda_b * L_bucket + lambda_o * L_official + lambda_s * L_smooth)
    loss.backward()
    optimizer.step()
```

这种方式避免了锚点约束在训练初期主导损失函数。

**推荐使用方式 C**，因为它：
1. 先达到合理的桶级拟合
2. 再用锚点约束做微调，减少对合成数据分布的影响
3. 训练过程可以监控"锚点约束是否扭曲了桶级预测"

### 4.4 考纲锚点的分级加权策略

不同考纲词表的锚点可靠性不同，因此采用分级加权：

| 等级 | 词表 | 权重 | 可靠性 | 说明 |
|:----:|------|:---:|:------:|------|
| A | 高考、CET-4 | 5x | 高 | 词表界定清晰，与考试等级对应明确 |
| B | CET-6、考研 | 3x | 中高 | 词表大但部分词超范围 |
| C | 中考、TEM-8 | 2x | 中 | 中考词表可能不全，TEM-8 低频词多 |
| D | IELTS、TOEFL | 1x | 中低 | 非中国考试体系，锚点对应性较弱 |

**权重调整原则**：
- 初期训练：全部锚点使用较低权重（1x-2x），避免主导损失函数
- 微调阶段：逐步提高高可靠性锚点的权重（3x-5x）
- 达到 1000+ 真实用户数据后：可降低合成数据权重，提高锚点权重

---

## 5. 实施路线图

### Phase 0：词表收集与验证（1-2 周）

**目标**：收集所有 P0 考纲词表，验证质量。

- [ ] **中考词表**：从《义务教育英语课程标准》附录提取
  - 预期 ~1,800-2,500 词
  - 验证：≥95% 在词库中，rank 中位数 ~1500-1800
- [ ] **CET-4 独立词表**：从 CET-6 词表剥离高考词 + 高频部分
  - 预期 ~4,000-5,000 词
  - 验证：与高考词重叠 <30%，rank 中位数 ~3000-4000
- [ ] **考研词表**：从考研英语大纲提取
  - 预期 ~5,500-6,000 词
  - 验证：与 CET-6 重叠率 60-70%，不含 CET-4 部分
- [ ] 更新 `optim/official_vocab.py` 支持所有新词表
- [ ] 更新 `optim/calibration_trainer.py` 中锚点相关逻辑

**交付物**：
- `data/exam_vocab/zhongkao.txt`、`cet4.txt`、`kaoyan.txt`
- 更新后的 `official_vocab.py`（5+ 考纲词表）
- 词表验证报告

### Phase 1：锚点损失函数实现（3-5 天）

**目标**：在现有 calibration trainer 中实现考纲锚点损失。

- [ ] 实现 `compute_exam_coverage(thetas, gamma, exam_set, bank)`
  - 计算给定模型参数下某个考纲词表的预测覆盖率
- [ ] 实现 `compute_official_loss(stats_map, thetas, gammas, exam_sets)`
  - 对所有用户的考纲锚点计算加权 MSE
- [ ] 将考纲损失集成到 `train_numpy()` 的损失函数中
- [ ] 实现 `--dry-run` 显示每个考纲词表的预测覆盖率
- [ ] 在 `train_numpy()` 中添加 `L_official` 的梯度计算（有限差分法）
- [ ] 使用 `--validate` 验证合成数据下锚点约束的有效性

**Checkpoint 1**：能在 `--dry-run` 模式下看到各考纲词表的预测覆盖率，基本逻辑正确。

### Phase 2：合成数据验证锚点有效性（2-3 天）

**目标**：在合成数据上验证考纲锚点是否改善了认知率预测。

- [ ] 生成合成数据时，确保合成用户的已知词集符合考纲模式（不严格按 rank 排序）
  - 例如：给"六级水平"用户额外添加 CET-6 考纲词
  - 这模拟了真实学习者"背了考纲词"的行为
- [ ] 运行 Phase A（无锚点训练）和 Phase B（有锚点训练）
- [ ] 对比两种训练下对上述"非纯 rank 排序"用户的预测误差
- [ ] 尝试不同的 λ_official 权重（0.1, 0.3, 0.5, 1.0, 2.0），分析锚点约束的强度影响
- [ ] 验证：加入锚点后，对考纲词覆盖率低估的用户是否得到了改善

**Checkpoint 2**：在合成数据上，锚点校准版本的预测误差↓ 10%+。

### Phase 3：正式训练与评估（1-2 天）

**目标**：在 V100 上运行完整训练，产出新参数。

- [ ] 拷贝当前训练脚本 `optim/train_bucket_matrix.py` 为锚点版本
- [ ] 集成考纲锚点损失
- [ ] 运行 Phase A（仅桶级 MSE，2000 epochs）
- [ ] 运行 Phase B（锚点微调，500 epochs，小学习率）
- [ ] 评估预测精度（合成数据 + 考纲覆盖率）
- [ ] 输出参数文件 `trained_params_bucket_anchored.json`

**Checkpoint 3**：新参数在合成数据上维持原精度，在考纲覆盖率指标上显著改善。

### Phase 4：线下验证（2-3 天）

**目标**：用真实数据或合理近似验证新参数效果。

- [ ] 准备 3 组测试数据：
  1. 合成标准用户（按 rank 填充，~12 个级别）
  2. 考纲偏差用户（按考纲填充，模拟背词人群，~6 个级别）
  3. 混合用户（部分 rank 部分考纲，~6 个级别）
- [ ] 对比新旧参数在这 3 组上的：
  - 桶级认知率拟合误差
  - 考纲覆盖率误差
  - 词汇量估计精度
- [ ] 手动检查"不合理"的预测案例
- [ ] 确认 θ 参数变化方向正确（高频桶变化小，中频桶受锚点影响大）

**Checkpoint 4**：新参数在所有测试集上表现≥旧参数，考纲覆盖率提升 5-15%。

### Phase 5：线上测试与部署（1-2 天）

**目标**：新参数到生产环境，验证实际效果。

- [ ] 在开发环境部署新参数 `trained_params_bucket_anchored.json`
- [ ] 准备 A/B 测试：随机 50% 用户使用旧参数，50% 使用新参数
- [ ] 指标监控：
  - 词汇量估计分布（不应当出现偏移）
  - 用户反馈（"我觉得估得准/不准"比率）
  - 对不同能力区间的区分度
- [ ] 如果 A/B 测试通过，新参数正式上线

**Checkpoint 5**：新参数在生产环境稳定运行，无明显回归问题。

### Phase 6：持续完善（可选）

- [ ] 每收集 50 名真实用户数据后，重新运行锚点微调
- [ ] 建立考纲词表覆盖率可视化看板
- [ ] 探索更大规模的考纲词表（CEFR 等级标注词库）
- [ ] 支持用户自报考试等级作为训练标签

---

## 6. 与现有 bucket matrix 模型的关系

### 6.1 增强而非替换

考纲锚点校准**不是**替代 bucket matrix 模型，而是**在其基础上增加新的训练约束**。

```
当前模型：
  
  bucket matrix 参数 (θ₁,...,θ₉)
        │
        ├── 定义各桶的认知率基线
        ├── 训练自合成数据（按 rank 填充）
        └── 隐含假设：词频 rank = 难度排序
               ↑ 这就是要改进的地方

加入锚点后：

  bucket matrix 参数 (θ₁,...,θ₉)
        │
        ├── 定义各桶的认知率基线（不变）
        ├── 训练自合成数据 + 考纲锚点约束
        ├── 新增：考纲覆盖率作为约束项
        └── 新增：高频生僻词会被锚点约束压低
```

**模型结构不变**：依然是 9 个桶的 sigmoid 模型 + 校准，推理管线完全不变。

**参数数量不变**：9 个 θ + 每个用户 γ，不增加参数。

**推理流程不变**：用户作答 → 按桶聚合 → 拟合 γ → 预测词汇量。

### 6.2 参数变化的方向性预期

引入考纲锚点后，θ 参数预期会发生以下变化：

| 桶 | 当前 θ | 预期变化 | 原因 |
|---|:------:|:--------:|------|
| 1k | +9.12 | 基本不变 | 高频词几乎所有人都认识，锚点不改变 |
| 2k | +6.15 | 基本不变 | 同上 |
| 3k | +4.74 | 略降 | 部分 rank 3000+ 的词在低频桶却被考试覆盖，梯度会拉低高频桶低估高频生僻词 |
| 5k | +2.78 | 略降 | 类似原因 |
| 8k | +1.24 | 略升 | 低频但属于考纲的词需要更高的认知率 |
| 10k | -0.16 | 略升 | 同上 |
| 15k | -2.65 | 略升 | 考研/TEM-8 锚点拉动 |
| 20k | -4.98 | 可能有升有降 | 高频生僻词下降，低频考纲词上升 |
| 30k | -7.92 | 略升 | 低频考纲锚点拉动 |

**总体趋势**：θ 曲线会从"严格的 rank 单调"变得更加**贴合真实学习曲线**——在中频段（5k-15k）变得更加平滑，低频段的下降不再那么陡峭。

### 6.3 校准参数的变化

当前校准参数几乎为恒等映射（k ≈ 0, piecewise slopes ≈ 1.0），锚点校准后预期变化不大，因为改变的是 θ 本身而不是校准层。

但若发现加入锚点后预测偏斜，校准参数可能需要进行一个**微调**：

```
预期 L_official 加入后：
  - L_bucket（桶级 MSE）可能略升（因为锚点给 θ 增加了约束）
  - 但整体模型在真实学习模式下的预测更准
  - 校准参数可能小幅变化以补偿
```

### 6.4 兼容性

新参数文件格式与旧版本兼容：

```json
{
  "model": "bucket_matrix",
  "theta": { ... },                       // 新训练值
  "calibration_k": ...,
  "piecewise_knots": [ ... ],
  "bucket_sizes": { ... },
  "training_epochs": 2500,                 // 2000 + 500 微调
  "training_mode": "bucket_mse+anchored",  // 新增标记
  "anchor_sets_used": ["zhongkao", "gaokao", "cet4", "cet6", "kaoyan"],
  "anchor_loss_weight": 0.5
}
```

推理代码无需更改——`trained_params_bucket.json` 的加载逻辑完全兼容新增字段。

### 6.5 回退机制

如果锚点校准后的参数在生产环境中出现问题：

```python
# 回退：只需把参数文件替换回去
cp trained_params_bucket.json trained_params_bucket_anchored.json.backup
cp trained_params_bucket_baseline.json trained_params_bucket.json
```

或在配置中支持多参数集切换：

```python
@dataclass
class EstimatorConfig:
    # ...
    theta_source: str = "anchored"  # "baseline" | "anchored"
    # 加载不同参数文件
```

---

## 7. 可行性分析与预期改进

### 7.1 对当前模型的预期影响

| 指标 | 当前值 | 目标值 | 测量方法 |
|:----|:-----:|:-----:|:--------|
| 考纲词覆盖率 MSE | 较高（未能约束） | ↓ 30-50% | 合成数据 + 真实考纲匹配测试 |
| 桶级认知率 MSE | 很低（合成数据过拟合） | 略升但可接受 | 留出测试集 |
| CET-6 用户预测偏差 | 可能低估 | 改善 10-15% | 合成考纲偏差用户测试 |
| 词汇量估计 MAE | ~200（合成数据） | ≤250（更真实场景） | 更接近真实场景的测试集 |
| θ 参数稳定性 | 稳定 | 稳定（锚点微调扰动小） | 交叉验证 |

### 7.2 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|:-----:|:----:|---------|
| 考纲词表质量差 | 中 | 高 | 收集后人工审核、词库匹配率验证 |
| 锚点约束过强 | 中 | 中 | 阶段式增加权重（Phase A→B），监控 L_bucket |
| 锚点与合成数据冲突 | 中 | 中 | 使用方式 C（先基线后微调） |
| 部分考纲词表难以获取 | 高 | 中低 | CET-4 可从 CET-6 推导，考研可网上收集 |
| 锚点校准后高频词预测偏差 | 低 | 中 | 高频词锚点权重低（中考词表的低频部分） |
| 真实用户与锚点模型行为不一致 | 不确定 | 中 | 初期仅做线下验证，仅在有信心时上线 |

### 7.3 需要的资源

| 资源 | 说明 |
|------|------|
| 考纲词表收集 | 1-2 周人工搜索 + 提取 |
| 计算资源 | V100（已有），训练时间约 30 分钟 |
| 验证数据 | 合成数据（已有），线下测试集 |
| 代码改动 | 主要涉及 `optim/calibration_trainer.py` 和 `optim/official_vocab.py` |
| 文档 | 本文档 + 更新 TECHNICAL.md |

### 7.4 何时认为锚点校准"成功"

综合判断指标：

1. ✅ 考纲覆盖率 MSE 下降 30%+（在合成锚点测试集上）
2. ✅ θ 参数变化符合方向预期（高频基本不变，中频略升）
3. ✅ 桶级 MSE 上升不超过 10%（锚点约束未过度扭曲）
4. ✅ 新模式对"背词学习者"的预测更准（合成用户测试）
5. ✅ 词汇量估计整体分布不偏移（生产监控）

---

## 8. 附录：数学推导

### 8.1 考纲覆盖率的梯度

对于考纲词表 E，预测覆盖率 P_E：

```
P_E = (1/|V_E|) × Σ_{w ∈ V_E} σ(θ_{b(w)} + γ)

其中 σ(x) = 1/(1+e⁻ˣ)
```

梯度 wrt θ_b（桶 b 的参数）：

```
∂P_E / ∂θ_b = (1/|V_E|) × Σ_{w ∈ V_E ∩ bucket_b} σ(θ_b + γ) × (1 - σ(θ_b + γ))
```

也就是说：只有桶 b 中的考纲词才对该梯度有贡献，贡献大小等于 sigmoid 的导数（最大 0.25）。

梯度 wrt γ（学习者偏移）：

```
∂P_E / ∂γ = (1/|V_E|) × Σ_{w ∈ V_E} σ(θ_{b(w)} + γ) × (1 - σ(θ_{b(w)} + γ))
```

### 8.2 锚点损失 wrt θ 的完整链式法则

锚点损失对 θ_b 的总梯度：

```
∂L_official / ∂θ_b = Σ_E 2 × w_E × (P_E - target_E) × ∂P_E / ∂θ_b
```

代入 ∂P_E/∂θ_b：

```
∂L_official / ∂θ_b = Σ_E 2 × w_E × (P_E - target_E) ×
                     (1/|V_E|) × Σ_{w ∈ V_E ∩ bucket_b} σ(θ_b + γ) × (1 - σ(θ_b + γ))
```

### 8.3 综合梯度

完整的梯度下降更新方向：

```
∇_θ_b L = ∇_θ_b L_bucket + λ_official × ∇_θ_b L_official + λ_smooth × ∇_θ_b L_smooth
```

### 8.4 与当前训练框架的集成

现有 `train_numpy()` 使用有限差分法计算 gradients。对于考纲损失，有限差分法同样适用：

```python
# 在 _loss_with_params() 中新增 L_official 计算
def _loss_with_params(..., include_official=True):
    # ... 原有损失
    if include_official:
        for uid, stats in stats_map.items():
            gamma = fitted_gammas[uid]
            for exam_name, exam_info in exam_sets.items():
                coverage = predict_exam_coverage(thetas, gamma, bank, exam_info)
                loss += exam_info.weight * (coverage - target) ** 2
    return loss
```

### 8.5 学习曲线监控

训练过程中应监控以下曲线以判断锚点训练是否健康：

```
Epoch loss curves:
  L_bucket:     ──── 应缓慢下降，如果突然上升 >10% 说明锚点权重过大
  L_official:   ──── 应持续下降
  L_smooth:     ──── 趋于平稳小值
  
Coverage per exam set:
  中考:     0.80 ─ 0.90 （目标：0.85）
  高考:     0.75 ─ 0.85 （目标：0.82）
  CET-4:    0.70 ─ 0.85 （目标：0.80）
  CET-6:    0.65 ─ 0.80 （目标：0.75）
  
  (随 γ 偏移变化，此处为中等能力学习者对应的数值)
```

---

## 附录 A：相关文件

| 文件 | 用途 | 需要修改 |
|------|------|:--------:|
| `data/exam_vocab/*.txt` | 考纲词表存储 | ✅ 新增中考/CET-4/考研词表 |
| `optim/official_vocab.py` | 考纲词表加载与元数据 | ✅ 添加新词表定义和覆盖率函数 |
| `optim/calibration_trainer.py` | 校准训练器核心 | ✅ 添加 L_official 计算 |
| `optim/train_bucket_matrix.py` | bucket matrix 训练脚本 | ✅ 集成锚点训练模式 |
| `vocab_estimator/trained_params_bucket.json` | 参数存储 | ✅ 新训练后更新 |
| `docs/calibration_pipeline.md` | 校准管线文档 | 交叉引用本方案 |
| `docs/TECHNICAL.md` | 技术文档 | 更新模型说明 |
| `docs/exam_based_calibration.md` | **本文档** | 新文档 |

## 附录 B：词表匹配工具

快速检查考纲词表与词库的匹配情况：

```bash
# 查看词表中有多少词在词库中
python3 -c "
from vocab_estimator.config import DEFAULT_CONFIG
from vocab_estimator.vocab_bank import VocabBank
bank = VocabBank(DEFAULT_CONFIG)

with open('data/exam_vocab/gaokao.txt') as f:
    words = {w.strip().lower() for w in f if w.strip()}
matched = sum(1 for w in words if bank.get_rank(w) is not None)
ranks = [bank.get_rank(w) for w in words if bank.get_rank(w) is not None]
ranks.sort()
print(f'Gaokao: {len(words)} words, {matched} matched ({100*matched/len(words):.0f}%)')
print(f'  rank median: {ranks[len(ranks)//2]}')
print(f'  rank range: [{min(ranks)}, {max(ranks)}]')

# 查看各桶中有多少考纲词
for b, ws in bank.words_by_bucket.items():
    in_bucket = sum(1 for w in ws if w in words)
    if in_bucket:
        print(f'  {b}: {in_bucket}/{len(ws)} = {100*in_bucket/len(ws):.0f}%')
"
```

## 附录 C：预期 θ 变化模拟分析

假设我们使用考纲锚点进行训练，预期的 θ 变化可以通过以下简化模拟进行预估：

```python
def simulate_theta_adjustment(
    bank: VocabBank,
    current_thetas: dict[str, float],
    exam_vocab_sets: dict,
    gamma: float = 0.0,
    lambda_official: float = 0.5,
    lr: float = 0.01,
    n_steps: int = 500,
):
    """
    简化的 θ 调整模拟。
    
    模拟在考纲锚点约束下 θ 的变化趋势。
    这是示意性的——实际训练使用梯度下降。
    """
    import copy
    thetas = copy.deepcopy(current_thetas)
    
    for step in range(n_steps):
        # 计算考纲覆盖率
        for name, info in exam_vocab_sets.items():
            P_E = predict_exam_coverage(thetas, gamma, info, bank)
            target = info.expected_coverage
            weight = info.weight
            
            err = P_E - target
            
            # 对于每个桶中的考纲词，更新 θ
            for bucket_name, theta in thetas.items():
                bucket_words = bank.words_by_bucket.get(bucket_name, [])
                exam_words_in_bucket = [
                    w for w in bucket_words 
                    if w.lower() in info["words_lower"]
                ]
                if not exam_words_in_bucket:
                    continue
                
                n_match = len(exam_words_in_bucket)
                n_total = len(info["words"])
                
                # 简化的梯度
                sigmoid_val = 1.0 / (1.0 + np.exp(-(theta + gamma)))
                grad = 2 * weight * err * (sigmoid_val * (1 - sigmoid_val)) * (n_match / n_total)
                
                thetas[bucket_name] -= lr * lambda_official * grad
    
    return thetas
```

---

## 附录 D：训练数据改造方案分析与推荐

### D.1 问题重述

当前 `generate_synthetic_users()` 生成的合成用户按**词频 rank 顺序**填充 known set：
- 词汇量 6000 的用户 = 认识 rank 1-6000 的所有词
- 训练标签 = 在各桶中严格按 rank 递减的认知率

**真实情况完全不同**：一个六级 6000 词汇量的学生，认识的是 CET-6 考纲中的约 6000 个词，这些词分布在 rank 78-28079 的广泛区间（p10=917, p50=6487, p90=18474），不是前 6000 个词频词。

**问题表现**：
- 估算结果偏高：简单答对几个高频词就把词汇量推到 4000-5000（旧参数约 4700）
- 原因：模型训练数据中的认知率模式是“按 rank 严格单调递减”，而在真实 CET-6 场景中，中低频桶的认知率远高于纯词频排序预期

---

### D.2 方案分析

#### 方案 A：CET-6 随机抽样替换

**做法**：
- 对 vocab_size = 1000, 1010, 1020, ..., 6000（步进 10，共 ~501 个用户）
- 每个用户从 CET-6 词表（全量 8028 词，词库匹配 ~7520 词）中随机抽 N 个词作为 known set
- 混合原有词频填充用户（500~20000）保持边界覆盖

**优点**：
- ✅ 直接解决 rank 分布不匹配问题
- ✅ 训练数据认知率模式与 CET-6 考生一致
- ✅ 词表来源可靠（教育部考试大纲），无需猜测学习者行为

**缺点**：
- ❌ 纯随机抽样忽略了“学习者优先学高频词”这一真实规律
- ❌ 501 个 CET-6 用户 + 12 个词频用户 → 原始词频用户几乎被淹没
- ❌ 500-1000 区间（低于 CET-6 门槛）缺乏 CET-6 用户覆盖

**适用场景**：快速验证 CET-6 假设是否正确的第一轮实验

---

#### 方案 B：加权 CET-6 抽样

**做法**：
- 从 CET-6 词表抽样时，按 `weight = 1/√(rank)` 加权
- 高频 CET-6 词（如 the, be, have）更可能被“认识”
- 低频 CET-6 词（如 subregionalize）只有高词汇量用户才可能认识
- 设置基础高频词池（top-500 frequency words）作为所有用户的保底

**优点**：
- ✅ 更符合认知规律：高频先学、低频后学
- ✅ 生成的学习曲线更平滑真实
- ✅ 在保持 CET-6 词汇分布的同时模拟了真实学习路径

**缺点**：
- ❌ 权重参数（1/√rank vs 1/log rank vs 1/rank）需要实验调优
- ❌ 实现稍复杂
- ❌ 难以验证加权方案是否“更真实”——缺乏真实用户 ground truth

---

#### 方案 C：混合频率 + CET-6 锚点增强（现状调参）

**做法**：
- 不改造训练数据
- 只调 anchor_weight（从 0.5 提到 2.0-5.0）
- 让考纲锚点损失在训练中起主导作用

**优点**：
- ✅ 改动最小，只需改一个超参
- ✅ 保留原有合成数据的桶级信号

**缺点**：
- ❌ 锚点损失只是弱约束。损失是 “预测覆盖率 vs 期望覆盖率” 的 MSE，而对每个用户的覆盖率只有 4 个数据点（中考、高考、四级、六级等），不足以主导 9 个 θ 参数的优化方向
- ❌ 覆盖率约束是**间接**的——锚点只约束词表整体的平均认知率，不约束单个桶的认知率
- ❌ 如果原始合成数据强烈偏向“rank 单调”，锚点可能根本无法掰正

---

#### 方案 D：全 CET-6 基准训练

**做法**：
- 完全废弃词频填充用户
- 只用 CET-6 词表生成用户：vocab = 500, 1000, 2000, ..., 7532（全量 CET-6）
- 词频用户全部移除

**优点**：
- ✅ 训练数据完全匹配 CET-6 考试场景
- ✅ 模型不存在任何“rank 单调”偏见

**缺点**：
- ❌ 500 以下区间（初学者）无训练数据
- ❌ 7532 以上区间（词汇量超过 CET-6 全量）无训练数据
- ❌ 对于不背考纲、只用英语的学习者（如外企员工、留学生），CET-6 模式可能不适用
- ❌ 无法处理 TOEFL/GRE 等高阶场景

---

### D.3 推荐方案：方案 B（加权 CET-6 抽样 + 混合词频用户）

**核心策略**：
```
训练用户池 = 
  [501 个 CET-6 用户]   // vocab_size 从 1000 到 6000，步进 10
  + [12 个词频填充用户]  // vocab_size = 500, 1000, ..., 20000
  + [2-4 个极低词汇量用户]  // vocab_size = 100, 300（词频填充，覆盖初学者）
```

**CET-6 用户生成具体方案**：
1. 加载全量 CET-6 词表（`data/exam_vocab/cet6.txt`，~7520 词匹配到词库）
2. 对每个目标词汇量 N（1000-6000，步进 10）：
   - 以权重 `1/√(rank+1)` 从 CET-6 词表中**无放回**抽样 N 个词作为 known set
   - 若 N ≥ 7520，直接 known = 全量 CET-6
   - 计算各桶的 known 比例 → bucket_true_rates
3. 同原始方法一样采样测试题 response 和 bucket_obs_rates
4. bucket_counts 使用相同的按桶比例采样逻辑

**保留词频填充用户的原因**：
- vocab_size ≤ 500 的初学者：CET-6 用户无覆盖，需要词频填充
- vocab_size ≥ 10000 的高阶用户：超过 CET-6 范围，需要词频填充
- 混合训练 = 模型同时学到“纯 rank 模式”和“CET-6 模式” → 泛化能力更强

**为何选择加权而非纯随机**：
- CET-6 词表中 rank=1 的词（the）和 rank=28000 的词（subregionalize）都在同一个词表里
- 纯随机抽样下，一个词汇量 1000 的用户可能知道 subregionalize 而不知道 the → 完全违反常识
- 加权抽样确保：
  - 高频 CET-6 词在所有用户中认知率较高
  - 低频 CET-6 词只有高词汇量用户才可能认识
  - 整体认知率曲线是**连续、平滑、向下倾斜**的，但斜率远小于纯 rank 填充

---

### D.4 预期效果

| 场景 | 旧参数 | 新参数（预期） | 说明 |
|:----|:-----:|:------------:|:-----|
| “只知道 5 个高频词”估测 | ~4700 | ~2000~3000 | 5 个高频词匹配 1k 桶高认知率，但 2k-5k 桶低认知率拉低整体 |
| CET-6 考生估测 | ~5500 | ~6500（接近真实） | CET-6 用户的桶级模式与训练数据一致 |
| 纯 rank 填充用户的 MAE | ~200 | ~300~400（可接受） | 混合训练后词频用户的拟合精度略有下降 |
| 考纲覆盖率 MSE | 较高 | ↓ 40-60% | 模型学到 CET-6 词表的内部分布 |

---

### D.5 实施优先级
1. **立即（本次）**：实现方案 B，跑一轮训练看 θ 变化
2. **短期**：对比新旧 θ 在“5 个高频词”场景下的估算值
3. **中期**：收集真实 CET-6 考生的答题数据（5-10 人），验证新模型
4. **长期**：按相同的加权 CET-6 逻辑扩展至其他考纲（高考、考研、专八）

---

### D.6 关键数据支持

CET-6 词表在频率桶中的分布（`data/exam_vocab/cet6.txt`，7520 词匹配到词库）：

| 桶 | CET-6 词数 | 占比 |
|:--|:---------:|:---:|
| 1k | 790 | 10.5% |
| 2k | 661 | 8.8% |
| 3k | 578 | 7.7% |
| 5k | 1027 | 13.7% |
| 8k | 1284 | 17.1% |
| 10k | 718 | 9.5% |
| 15k | 1173 | 15.6% |
| 20k | 685 | 9.1% |
| 30k | 607 | 8.1% |

这意味着一个 CET-6 词汇量 6000 的用户，其 known set 在桶中的**期望分布**是：
- 1k 桶 ~630 词 / 约占 10.5%
- 8k 桶 ~1026 词 / 约占 17.1%
- ...
- 全部 9 个桶都有分布，并非只在前几个桶

对比旧方案（6000 词 = 认识 rank 1-6000）：所有已知词集中在 1k+2k+3k 桶（~3000 词）和 5k 桶的前 3000 词，8k 桶及以后的认知率为 0。

**新旧方案的认知率曲线对比（示意图）**：
```
认知率
1.0 │  ████ ← 旧方案：1k/2k/3k全满，8k+为0
    │  ████
    │  ████
    │  ████                                    
    │  ████  ████ ← 新方案：均匀分布，缓慢递减
    │  ████  ████  ████  ████  ████  ████  ████  ████  ████
    │  ████  ████  ████  ████  ████  ████  ████  ████  ████
  0 └──────────────────────────────────────────────────────────
      1k    2k    3k    5k    8k    10k   15k   20k   30k
```

---

### D.7 挖坑："知道 5 个高频词"为什么旧模型估到 ~4700？

这是一个极端但揭示问题的测试用例：用户只在 C 类（高频词测试 card）中答对 5 个极高频词（the, a, is, are, have）。

**旧模型的推理路径**：
1. 5 个高频词都答对 → 1k 桶认知率 ≈ 100%
2. 模型发现 1k 桶全满 → 推断 γ 很大（1k 桶 θ=+9.12，要 P≈1.0 需要 γ=某个大值）
3. γ 跨桶共享 → 所有桶的认知率都被推高
4. 各桶认知率求和 → raw_vocab ≈ 15000+ → 校准后 ≈ 4700

**问题根因**：
- 训练数据中，任何一个 vocab_size=6000 的用户，其 1k 桶都是 100% 认知率（因为 6000 词覆盖了前 6 个桶）。
- 模型学到的关联是：“1k 桶 100% → 总体词汇量至少 6000”
- 但真实世界中，1k 桶 100% 只是“我认识所有最常见的 1000 词”，不代表我的词汇量就是 6000

**新模型期望的推理路径**：
1. 5 个高频词都答对 → 1k 桶认知率 ≈ 100%
2. 但训练数据中的 CET-6 用户，即使词汇量只有 2000，1k 桶也可能是 100%（因为 they, be, have 都在 CET-6 词表里）
3. 所以 γ 不会被推到很大
4. raw_vocab 更保守 → 校准后 ≈ 2000-3000

这才是真实学习者应该得到的估值。

---

*附录 D 版本：1.0*
*最后更新：2026-06-23*
*作者：akunai 的代码助手*
