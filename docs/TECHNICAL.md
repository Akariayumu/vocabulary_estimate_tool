# 英语词汇量估算工具 — 技术文档

## 文档信息

| 项目 | 内容 |
|------|------|
| 项目名称 | 英语词汇量估算工具 (English Vocabulary Size Estimator) |
| 部署地址 | [https://vocab.akuai.life](https://vocab.akuai.life) |
| 目标 URL（需要 Host header） | `154.9.242.48`（nginx 反向代理 → localhost:7860） |
| 开发端口 | `127.0.0.1:7860` |
| 最后更新 | 2026-06-23 |
| 当前估算模型 | 分桶矩阵参数模型（Version 2） |

---

## 1. 项目概述

### 1.1 目标

通过**选择题测试**或**英文文章分析**，快速估算用户的英语词汇量（以词族 / lemma 为单位），并映射到中国英语学习者水平等级（初中、高中、四级、六级、考研、母语级）。

### 1.2 核心能力

- **词汇量选择测试**：从 9 个频段桶分层抽样，用户做选择题（二进制认识/不认识或 4 选 1 中文翻译），然后评估词汇量。
- **文章词汇量估算**：用户输入英文文章，系统 tokenize → lemmatize → rank 分布 → 加权公式估算。
- **群组比较**：支持 C/F/P/K 四类（正确、错误、部分、不知道）的群组对比，带 PAVA 保序回归。
- **两阶段自适应测试**：Stage 1 均匀抽样 → Stage 2 聚焦边界桶精化采样。

### 1.3 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | Python FastAPI + uvicorn |
| 前端 | 静态 HTML + JS + CSS（FastAPI StaticFiles mount） |
| 数据持久化 | SQLite（学生记录和测试记录） |
| 词频数据 | `wordfreq` 库（fallback 为内置列表） |
| 词形还原 | spaCy `en_core_web_sm`（fallback 为规则匹配） |
| 机器翻译 | 自建英→中字典，22,887 条（全覆盖词库） |
| 参数训练 | PyTorch / NumPy，在 V100（192.168.100.104）上训练 |

### 1.4 部署架构

```text
用户浏览器
    │
    ▼
nginx 反向代理 (154.9.242.48)
    │
    ▼
systemd: vocab-estimator
    │
    ├── uvicorn (localhost:7860)
    │   ├── FastAPI endpoints (/api/...)
    │   └── StaticFiles mount (/web/ → web/)
    ├── SQLite DB (data/vocab_estimator.sqlite3)
    └── Trained params (trained_params_bucket.json)
```

---

## 2. 数据层

### 2.1 VocabBank — 词库

词库使用 `wordfreq` 库获取英语单词的频率排名，将排名 ≤ 30,000 的词保留下来。如果 `wordfreq` 不可用，则 fallback 到一个内置的约 660 词的紧凑列表（`FALLBACK_WORDS`）。

#### 词形还原（Lemmatization）

- 主实现：spaCy `en_core_web_sm`（禁用 parser 和 ner 以轻量化）
- Fallback：基于规则的保守还原（处理常见的 -ing, -ed, -ies, 不规则变形等）
- 兼容 `is_proper_like`（大写专有名词跳过）、`is_abbreviation`（全大写缩写跳过）

#### Lemma 词族合并

词库按 **lemma 词族**去重。同一个 lemma 的不同单词形式（如 run, runs, running, ran）被合并为一条记录，取最低的 frequency rank。

#### 过滤规则

- 最小词长：2 个字符
- 仅保留纯字母词（去掉数字、标点）
- 排除常见专有名词（人名、地名如 John, London, China）
- 排除全大写缩写（长度 ≤ 5，如 NASA）
- 排除非常用专有名词-like 大写词

#### 词库统计（当前版本）

| 桶标签 | 排名范围 | 词数 |
|--------|---------|------|
| 1k     | 1–1,000 | 840 |
| 2k     | 1,001–2,000 | 775 |
| 3k     | 2,001–3,000 | 743 |
| 5k     | 3,001–5,000 | 1,457 |
| 8k     | 5,001–8,000 | 2,146 |
| 10k    | 8,001–10,000 | 1,435 |
| 15k    | 10,001–15,000 | 3,526 |
| 20k    | 15,001–20,000 | 3,588 |
| 30k    | 20,001–30,000 | 7,228 |
| **总计** | | **21,738** |

### 2.2 Translations — 英→中翻译字典

文件 `server/translations.py` 导出一个 22,887 条记录的 Python 字典 `TRANSLATIONS: dict[str, str]`。键为小写单词或 lemma，值为中文翻译。

- 覆盖词库全部 21,738 个 lemma（覆盖率 100%）
- 还包含部分词库外的低频/罕用词的翻译

在选择题模式下，如果目标词没有翻译，自动降级为 binary（认识/不认识）模式。

### 2.3 SQLite 数据库

位置：`data/vocab_estimator.sqlite3`

两张表：

- **students**：`id, name, cet_score (可选), created_at`
- **test_records**：`id, student_id (FK), estimate, level, confidence, range_low, range_high, responses_json, created_at`

启动时自动调用 `init_db()` 建表（如果表不存在）。

---

## 3. 测试引擎

### 3.1 词汇量选择测试（/api/vocabulary/quiz）

工作流程：

1. **调用 `/api/vocabulary/quiz?per_bucket=12`** 获取分层抽样题目
   - 从每个桶抽取 `per_bucket` 个词
   - 每个词以 30% 概率生成"陷阱题"（所有选项都是错误翻译，正确答案是"没有正确答案"）
   - 70% 概率正常 4 选 1（从翻译字典取正确翻译，从同桶或全词库中取干扰项）
   - 如果目标词无翻译，降级为 binary 题（认识/不认识）

2. **用户作答后，调用 `/api/estimate`** 提交 `[{word, known}, ...]`
   - 如果 `trained_params_bucket.json` 存在，使用分桶矩阵模型
   - 否则 fallback 到 2 参数 Logistic 回归模型

3. **可选的 Stage 2 自适应细化**（/api/vocabulary/quiz-stage2）
   - 根据 Stage 1 的回答，计算每个桶的贝叶斯平滑已知率
   - 已知率在 0.20–0.80 之间的桶为"边界桶"
   - 从边界桶中再抽取额外词汇（default 8 个/桶）

### 3.2 分层抽样策略

#### Stage 1：均匀分层

`VocabularySampler.balanced_sample(per_bucket=12)`：
- 从 9 个桶各抽 12 个词，共 108 题
- 词内随机抽样（不重复）

#### Stage 2：自适应精化

`VocabularySampler.adaptive_sample(previous_responses, total_count=40)`：
- 根据已知率计算每个桶到 0.5 的"距离"
- 距离越小的桶获得更多抽样（信息量最大）
- 保证每个非空桶至少 1 题

#### 题目采样策略（优化管线中的合成数据用）

在 `optim/interval_sampler.py` 中实现了多种采样策略供训练使用：

- **均匀间隔采样**（interval=50, per_group=2）：将 30,000 的 rank 范围分为 50-rank 间隔，每个间隔抽 2 词
- **幂律加权采样**（weighted, power=0.7）：高频率桶按 size × (1/median_rank)^power 加权
- **智能采样**（smart, exam-level-aware）：基于多个考试等级（中考/高考/四级/六级/考研/母语级）的 sigmoid p(known) 方差（信息量），分配更多样本到区分度高的桶

### 3.3 文章词汇量估算（/api/estimate/article）

输入文章文本，进行：

1. **Tokenize**：用正则 `[A-Za-z]+(?:'[A-Za-z]+)?` 提取单词
2. **预处理**：跳过含数字的词、专有名词、全大写缩写
3. **Lemmatize**：还原到 lemma 形式并去重
4. **Rank 收集**：对每个 lemma 查询在 VocabBank 中的 frequency rank
5. **Rank 分布公式估算**（核心算法）：

```python
p50_rank  = ranks 的第 50 百分位
p90_rank  = ranks 的第 90 百分位
p95_rank  = ranks 的第 95 百分位
max_rank  = ranks 的最大值

raw_estimate = p50_rank
             + (p90_rank - p50_rank) × 0.7
             + (p95_rank - p90_rank) × 0.4
             + (max_rank - p95_rank) × 0.15
```

**为什么不使用 logistic 回归？** 因为文章中的词全部被当作"已知"（正样本），logistic 回归在全是正样本时会退化（sigmoid 趋向无穷大）。所以使用 rank 分布加权公式。

### 3.4 群组对比（/api/estimate/groups）

支持 C/F/P/K 四类（Correct/Failed/Partial/Don't Know）的群组对比：

1. 对每组分别执行 `estimate_single()`
2. 使用 PAVA（Pool Adjacent Violators Algorithm）保序回归强制 **C ≥ F ≥ P ≥ K** 的单调递减顺序
3. 可选 sklearn 或 NumPy 手写实现

---

## 4. 估算模型

系统经历了两个版本的模型演进。

### 4.1 ⛔ Version 1（已退役）：2 参数 Logistic 回归

#### 模型形式

```python
P(known|rank) = sigmoid(α + β × log(rank))

raw_vocab = Σ [ sigmoid(α + β × log(r)) for all ranks r in 1..30000 ]
```

- **α**（截距）：每个用户单独拟合，反映该用户的整体能力
- **β**（斜率）：全局共享参数，控制曲线的陡峭程度
- 默认值：α=0.0（启动值）, β=-0.30（固定或训练得来）

#### 拟合方式

优先使用 sklearn 的 `LogisticRegression`（L-BFGS，L2=1.0）。若 sklearn 不可用，fallback 到 NumPy 梯度下降（800 次迭代，lr=0.05）。

#### 加权拟合（方案 A）

从 2026-06-22 版起，支持 rank 加权拟合：

```python
weight(w) = 1 / (1 + log₂(max(rank, 10) / 10))
```

- rank=1 → weight≈1.0（高频词答错惩罚最大）
- rank=5,000 → weight≈0.100
- rank=20,000 → weight≈0.083

加权后模型更关注高频词的准确性，产生更保守的词汇量估算。启用配置项 `enable_weighted_fitting: True`。

#### 退役原因

- **低词汇量用户严重高估**：高频词大部分认识 → logistic 模型外推成大量的低频词也认识 → 词汇量被拉高 ~4000 词
- 2 个全局参数表达能力有限，无法捕捉不同频段的异质性

### 4.2 ✅ Version 2（当前）：分桶矩阵参数模型

#### 模型形式

```python
对每个桶 b:
    P(known|bucket_b) = sigmoid(θ_b + γ_u)

raw_vocab = Σ [ bucket_size_b × sigmoid(θ_b + γ_u) ]
```

- **θ_b**（桶参数）：9 个参数，每个桶一个，在 V100 上训练得到
- **γ_u**（用户偏移）：针对每个用户在推理时通过最小化 MSE 拟合

#### 已训练的参数（trained_params_bucket.json）

| 桶 | θ_b | 特征说明 |
|----|-----|---------|
| 1k  | +9.12 | 几乎所有人都认识（p≈1.0） |
| 2k  | +6.15 | 非常高频多数认识 |
| 3k  | +4.74 | 高频大多数认识 |
| 5k  | +2.78 | 过渡区开始 |
| 8k  | +1.24 | 部分认识 |
| 10k | -0.16 | 边界（p≈0.5） |
| 15k | -2.65 | 低频逐渐不认识 |
| 20k | -4.98 | 低频大多数不认识 |
| 30k | -7.92 | 非常低频几乎不认识 |

校准参数：
- `k` ≈ 0（tanh 几乎为恒等映射）
- 分段线性斜率 `ks` ≈ [0.99, 1.00, 1.01]（几乎不起作用）

#### 推理时的 γ 拟合

1. 对用户的 responses 按桶聚合，计算每个桶的观测已知率
2. 在 [-15, 15] 范围内网格搜索 γ（步长 0.5）
3. 在最佳网格点附近用二分搜索精化（30 步）
4. 使用找到的 γ 计算各桶认知概率 → 加权求和 → 校准

#### 训练流程（在 V100 上运行）

1. **合成数据生成**：生成 12 个不同词汇量水平（500–20,000）的"虚拟用户"
   - 从高频到低频填充已知词集至目标词汇量
   - 按桶大小按比例分配 300 道测试题
   - 记录每个用户的"真实认知率"（无噪声训练标签）
2. **Phase 1：训练 θ 和 γ**
   - 使用 PyTorch Adam 优化器（fallback NumPy Adam）
   - Loss = MSE(预测认知率, 真实认知率) + L2 正则
   - epoch=2000, lr=0.05, l2_θ=0.001, l2_γ=0.001
3. **Phase 2：训练校准参数 k 和 ks**
   - 有限差分法计算梯度，Adam 更新
   - k 的学习率极小（1e-9），ks 学习率 0.001
   - 模型收敛到 k≈0, ks≈[1,1,1]（恒等校准）

#### 性能指标

- **预测误差**：~200 词（MAE）
- **低词汇量用户不再高估**
- 校准几乎为恒等映射（k→0, knots→1），说明分桶矩阵本身的预测已经非常准确

---

## 5. 校准体系

### 5.1 两阶段校准管线

#### Stage 1：tanh 平滑饱和

```python
calibrated = max_v × tanh(k × raw_vocab)
# max_v = 20,000 (native speaker ceiling)
# k = 0.0000691 (旧版) / ≈0 (新版桶矩阵)
```

设计目的：防止纯 logistic 模型的高估。tanh 在低频区产生饱和，锚定母语者上限 ~20,000 词族。

推荐校准曲线（旧版 k=0.0000691 时）：
| 原始估计 | 校准后 | 对应水平 |
|----------|--------|---------|
| 21,303（全对） | 18,049 | 母语者水平 |
| 10,000 | 10,335 | 专业/母语级 |
| 8,000 | 7,877 | 六级+/考研 |
| 6,000 | 5,181 | 四级 |
| 4,000 | 4,076 | 高中/四级过渡 |
| 2,000 | 2,747 | 初中/高中过渡 |

#### Stage 2：分段线性压缩

```python
knots = [(3000, 1.00), (8000, 0.45), (22000, 1.28)]
```

- [0, 3000]：slope=1.00（保持原样）
- [3000, 8000]：slope=0.45（压缩中段，防止四级被高估）
- [8000, 22000]：slope=1.28（扩展高端，让母语级达到 18000+）

### 5.2 当前版本与旧版本对比

| 特性 | 旧版（2-param logistic） | 新版（分桶矩阵） |
|------|------------------------|-----------------|
| 校准必要性 | 强校准（k>0, knots≠1） | 几乎不需要校准 |
| k 值 | 0.0000691 | ≈0（无 tanh 效果） |
| 分段斜率 | [1.00, 0.45, 1.28] | [0.99, 1.00, 1.01] |
| 最大输出 | ~18,000 | ~21,000 |
| 低水平误差 | ~4,000 高估 | ~200（MAE） |

### 5.3 校准参数训练器

`optim/calibration_trainer.py` 提供了基于真实用户数据的校准参数训练管线：

- **Loss 组成**（多目标联合优化）：
  - `L_bucket`：桶级 MSE（原版，权重 1.0）
  - `L_interval`：间隔组 MSE（权重 0.3）— 在 rank 间隔内按组计算预测一致性
  - `L_official`：官方考试词汇覆盖率 MSE（权重 0.3）— 锚定 CET-4/6、高考等考试词汇的已知率
  - `L_smooth`：光滑性正则（权重 0.01）— 相邻 rank 预测概率的平方差和

- **官方考试词表锚点**（`optim/official_vocab.py`）：内置中考、高考、四级、六级、考研的代表性词表，其预期覆盖率为校准提供锚定参考。

---

## 6. 参数训练管线

### 6.1 合成数据生成策略

核心思想：用明确知道某词汇量水平的"虚拟用户"来训练模型。

#### 生成过程

1. **确定目标词汇量**（如 5,000）
2. **填充已知词集**：从高频桶到低频桶逐个填充，直到达到目标词汇量
3. **生成测试题**：从各桶按大小比例抽样（共 ~300 题）
4. **标注**：如果测试词在已知词集中 → known=True；否则 → known=False
5. **记录真实认知率**：每个桶实际已知词数 ÷ 桶大小（作为无噪声训练标签）

#### 采样加权策略

为训练数据生成设计的多种采样权重：

- **幂律加权**（`weighted_sample_words`）：`weight_b = size_b × (1/median_rank_b)^power`
  - power=0 → 按桶大小均匀分配
  - power=0.7 → 高频桶多抽（推荐默认）
  - power=1.0 → 强高频偏置
- **智能采样**（`smart_sample_words`）：基于考试等级 sigmoid 的 p(1-p) 信息量加权
  - 对 k 个考试等级计算 p(known) 方差 Σp(1-p)
  - 信息量大的桶（过渡区）获得更多样本
  - `info_exponent` 控制高低区分度桶的差异放大程度

### 6.2 优化细节

| 参数 | Phase 1（θ, γ） | Phase 2（k, ks） |
|------|-----------------|-----------------|
| 优化器 | Adam | Adam |
| 学习率 | 0.05 | lr_k=1e-9, lr_ks=0.001 |
| Epochs | 2,000 | 500 |
| L2 θ | 0.001 | — |
| L2 γ | 0.001 | — |
| 梯度计算 | 自动求导（PyTorch） | 有限差分法 |
| 训练用 | 12 个虚拟用户 | 12 个虚拟用户 |

### 6.3 参数文件

`vocab_estimator/trained_params_bucket.json` 包含：

- `theta`：9 个桶参数
- `calibration_k`：tanh 率（当前 ≈0）
- `piecewise_knots`：分段线性斜率
- `bucket_sizes`：9 个桶的词数
- `gammas`：训练时 12 个虚拟用户的 γ 值
- `accuracy`：每个虚拟用户的原始预测、校准后预测和误差

---

## 7. 采样策略详解

### 7.1 贝叶斯平滑

使用 Beta 先验对每个桶的已知率进行平滑：

```python
alpha, beta = bucket_beta_prior(bucket_index, n_buckets, config)
smoothed_rate = (observed_known + alpha) / (observed_total + alpha + beta)
```

先验参数在高低频桶之间线性内插：
- 高频桶（index≈0）：prior_known ≈ 4.0（倾向于高认知率）
- 低频桶（index≈8）：prior_known ≈ 0.5（先验不确定）

### 7.2 自适应采样

`VocabularySampler.adaptive_sample()` 的计算逻辑：

1. 根据响应计算每个桶到 0.5 的"距离"：`|known_rate - 0.5|`
2. 权重 = `1.0 / (0.05 + distance)`（距离越近权重越大）
3. 每个非空桶至少 1 题
4. 剩余题目按权重比例分配到各桶
5. 排除已见过的单词

### 7.3 Bootstrap 置信区间

`VocabEstimator.bootstrap_interval()`：

- 对 prepared responses 进行 300 次有放回重采样
- 每次计算一次 logistic_estimate
- 取 5% 和 95% 分位作为 90% 置信区间
- 如果 bootstrap 无法正常计算，fallback 到 baseline_estimate 的单一值

---

## 8. 部署架构

### 8.1 部署拓扑

```text
[用户] → [DNS: vocab.akuai.life]
    ↓
      nginx (154.9.242.48, Ubuntu 22.04)
    ↓
      systemd vocab-estimator service
    ↓
      uvicorn (localhost:7860)
    ↓
      FastAPI app (server/main.py)
```

### 8.2 启动方式

使用 `run.sh`：

```bash
#!/bin/bash
cd /home/akuai/stu/vocab_estimator
python3 -m uvicorn server.main:app --host 127.0.0.1 --port 7860 --reload
```

生产环境使用 systemd 服务 `vocab-estimator.service`。

### 8.3 SQLite 数据库

- 位置：`data/vocab_estimator.sqlite3`
- 表：`students` 和 `test_records`
- 启动时自动建表

### 8.4 API 端点汇总

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 前端首页 |
| `/api/estimate` | POST | 提交选择题作答，返回词汇量估算 |
| `/api/estimate/groups` | POST | C/F/P/K 群组对比 |
| `/api/estimate/article` | POST | 文章词汇量估算 |
| `/api/vocabulary/stats` | GET | 词库统计信息 |
| `/api/vocabulary/sample` | GET | 词汇抽样（不带翻译） |
| `/api/vocabulary/quiz` | GET | 词汇选择题（带中文翻译） |
| `/api/vocabulary/quiz-stage2` | POST | 两阶段自适应测试 Stage 2 |
| `/api/tests/save` | POST | 保存测试记录 |
| `/api/tests/records` | GET | 查询历史记录 |

---

## 9. 文件结构

```
vocab_estimator/
│
├── server/
│   ├── main.py                  # FastAPI 应用入口，所有端点
│   ├── translations.py          # 英→中翻译字典（22,887 条）
│   ├── database.py              # SQLite 数据库操作
│   └── __init__.py
│
├── vocab_estimator/
│   ├── __init__.py              # 包入口，导出 VocabBank / VocabEstimator
│   ├── config.py                # EstimatorConfig 数据类，所有调参在此集中
│   ├── vocab_bank.py            # VocabBank 词库构建与查询
│   ├── vocab_model.py           # VocabEstimator 旧版 2-param logistic 模型
│   ├── bucket_model.py          # 新版分桶矩阵参数模型
│   ├── sampler.py               # VocabularySampler 采样器
│   ├── coverage.py              # DocumentCoverageAnalyzer 文章覆盖率分析
│   ├── lemmatizer.py            # Lemmatizer 词形还原（spaCy / 规则 fallback）
│   └── trained_params_bucket.json  # 训练好的分桶矩阵参数文件
│
├── optim/
│   ├── __init__.py
│   ├── train_bucket_matrix.py   # 分桶矩阵参数训练（V100 上运行）
│   ├── calibration_trainer.py   # 校准参数训练器（β, k, piecewise knots）
│   ├── synthetic_generator.py   # 合成数据生成 + 训练验证
│   ├── interval_sampler.py      # 间隔/幂律/智能采样策略
│   └── official_vocab.py        # 官方考试词表锚点
│
├── web/
│   ├── index.html               # 前端页面
│   ├── app.js                   # 前端逻辑代码
│   └── styles.css               # 样式表
│
├── scripts/
│   ├── train_quick.py           # 快速训练脚本（已归档）
│   ├── train_calib.py           # 校准训练脚本
│   ├── train_bucket_matrix.py   # 桶矩阵训练脚本
│   ├── fill_all_translations.py # 翻译填充脚本
│   ├── fill_remaining.py        # 剩余翻译填充
│   ├── generate_translations.py # 翻译生成脚本
│   ├── gen_samples.py           # 采样生成
│   ├── fix_main.py              # 代码修复
│   └── analyze_fit.py           # 拟合分析
│
├── data/
│   └── vocab_estimator.sqlite3   # SQLite 数据库
│
├── docs/
│   └── TECHNICAL.md              # 本文档
│
├── run.sh                       # 启动脚本
└── requirements.txt             # Python 依赖
```

---

## 10. 版本历史

| 日期 | 变更 |
|------|------|
| 2026-06-22 | **初始部署**：2 参数 logistic 模型 + 文章分析功能，部署在 154.9.242.48 |
| 2026-06-23 09:50 | **迁移到 ~/stu/**，翻译补全至 100%（22,887 条） |
| 2026-06-23 11:42 | **校准参数调整**：全对→18,000，压缩中段扩展高端 |
| 2026-06-23 13:41 | **在 V100 训练分桶矩阵参数**：合成数据 12 用户，PyTorch Adam，2000 epochs |
| 2026-06-23 14:24 | **分桶矩阵模型部署上线**：误差从 ~4000 降低到 ~200 词，低词汇量用户不再高估 |

---

## 11. 未来优化方向

1. **用真实用户数据重新训练**：当前参数基于合成数据训练，如果积累足够的真实用户测试记录，可以用实际数据微调模型
2. **二阶段自适应测试优化**：当前 Stage 2 的边界桶判定阈值可以基于用户实际表现动态调整
3. **更多的官方考试词表锚点**：扩充 TEM-4/8、IELTS、TOEFL 等词表，提升校准精度
4. **多级置信区间**：目前使用单一的 90% 置信区间，可以扩展到多个置信水平
5. **词汇学习建议生成**：根据用户的薄弱桶，生成针对性的词汇学习建议
6. **前端用户体验提升**：选择题界面优化、测试进度提示、历史趋势展示等
