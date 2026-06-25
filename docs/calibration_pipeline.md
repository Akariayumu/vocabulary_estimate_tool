# 词汇量估算模型——参数校准管线

> 用实测答题数据，通过梯度下降反向训练模型参数，取代当前的经验式调参。

---

## 目录

1. [背景与问题](#1-背景与问题)
2. [核心模型回顾](#2-核心模型回顾)
3. [测试矩阵设计](#3-测试矩阵设计)
4. [抽样策略](#4-抽样策略)
5. [参数优化算法](#5-参数优化算法)
6. [数据集格式](#6-数据集格式)
7. [实施路线图](#7-实施路线图)
8. [可行性分析](#8-可行性分析)
9. [附录：数学推导](#9-附录数学推导)

---

## 1. 背景与问题

### 现状

当前词汇量估算模型包含以下**经验调参**的环节：

| 参数 | 类型 | 当前值 | 设定依据 |
|------|------|--------|----------|
| `calibration_k` | 标量 | 0.0000691 | 手动调至使 all-correct ≈ 18,000 |
| `piecewise_knots[0]` | (boundary=3000, slope=1.00) | 3000 以下保持 | 经验 |
| `piecewise_knots[1]` | (boundary=8000, slope=0.45) | 3000-8000 压缩 55% | 经验 |
| `piecewise_knots[2]` | (boundary=22000, slope=1.28) | 8000-22000 膨胀 28% | 经验 |
| `logistic_l2` | 正则强度 | 1.0 | 主观 |

这些参数目前依靠手动校准 + 直觉判断，缺乏真实数据支撑。

### 方案目标

采集真实用户的答题数据，建立一个**扩展测试矩阵**，然后用梯度下降等优化方法，从数据中反推出最优的模型参数，使预测的桶级认知率与实测认知率的误差最小化。

---

## 2. 核心模型回顾

### 2.1 Logistic 回归（单学习者）

对于某个学习者的答题记录 `{(word, known)}`，模型拟合：

```
logit(P) = α + β × log(rank)

P(认识 rank=r 的词) = sigmoid(α + β × log(r))
```

拟合后，词汇量原始估计值：

```
raw_estimate = Σ_{r=1}^{R} P(认识 rank=r 的词)
             = Σ_{r=1}^{R} sigmoid(α + β × log(r))
```

其中 R = `config.vocab_size`（默认 30000）。

### 2.2 校准管线（Calibration Pipeline）

```
raw_estimate
    ↓
Stage 1: tanh 平滑饱和
    calibrated = max_v × tanh(k × raw_estimate)
    ↓
Stage 2: 分段线性压缩（piecewise linear）
    [0, b₁]:   slope = s₁
    [b₁, b₂]:  slope = s₂
    [b₂, b₃]:  slope = s₃
    ...
```

### 2.3 桶级认知率

词汇库按 rank 分桶：1k、2k、3k、5k、8k、10k、15k、20k、30k。

对于给定的学习者 j，我们对该学习者在桶 b 的**预测认知率**为：

```
P_jb = (1/N_b) × Σ_{r in bucket_b} sigmoid(α_j + β × log(r))
```

其中 N_b 是桶 b 中的词数。

---

## 3. 测试矩阵设计

### 3.1 测试矩阵结构

测试矩阵是一个**学习者 × 词条** 的二元响应矩阵：

```
         word_1  word_2  word_3  ...  word_N
user_1      1       0       1    ...    1
user_2      0       1       0    ...    0
   ...
user_M      1       1       0    ...    1
```

- 1 = 认识，0 = 不认识
- 每行对应一名测试者
- 每列对应一个测试词
- 稀疏矩阵：每个测试者只做 ~50-200 题

### 3.2 频段桶覆盖

词汇库各桶大小：

| 桶标签 | 词数 | 占比 |
|--------|------|------|
| 1k     | 840  | 3.9% |
| 2k     | 775  | 3.6% |
| 3k     | 743  | 3.4% |
| 5k     | 1457 | 6.7% |
| 8k     | 2146 | 9.9% |
| 10k    | 1435 | 6.6% |
| 15k    | 3526 | 16.2% |
| 20k    | 3588 | 16.5% |
| 30k    | 7228 | 33.2% |
| **合计** | **21738** | **100%** |

### 3.3 能力层覆盖

测试者的能力应覆盖从初中水平（~1500 词）到母语水平（~20000 词）的完整区间。
按预期词汇量分层：

| 能力层标签 | 预期词汇量区间 | 目标测试者数 | 来源 |
|-----------|--------------|------------|------|
| 初中 | 1000-2500 | 30-50 | 初中生 |
| 高中 | 2500-4000 | 30-50 | 高中生 |
| 四级 | 4000-5500 | 30-50 | 大学低年级 |
| 六级 | 5500-7500 | 30-50 | 大学高年级 |
| 考研/六级+ | 7500-10000 | 20-30 | 考研/专四 |
| 专八/高水 | 10000-15000 | 15-20 | 英语专业 |
| 母语/近母 | 15000-20000 | 10-15 | 教师/海外 |

**总计目标：** ~200 名测试者。

> 这是**最低样本量**。每个能力层至少 10 人才能获得统计显著的桶级认知率。
> 每增加一名测试者，都为校准参数提供有用的约束。

### 3.4 认知率随 rank 衰减的先验形状

不同能力水平的测试者，其认知率随 rank 衰减的趋势大致如下：

```
认知率
 1.0 │
     │  ┌──高能力（词汇量 ~15000）
     │  │         ┌──中能力（词汇量 ~6000）
     │  │         │           ┌──低能力（词汇量 ~3000）
     │  │         │           │
     │ ╱╲        │           │
     │╱  ╲      ╱╲           │
  0 │──────╲───╱──╲─────────╱╲──
     1k   3k  5k   8k   15k   30k  rank
```

校准管线目标：让这个形状从数据中学习，而非人工指定。

### 3.5 测试题型（复用现有设计）

目前已有两种题型：

1. **二选一（Binary）：** 显示单词，用户选择"认识 / 不认识"
2. **四选一（Multiple Choice）：** 显示单词，从 4 个中文释义中选择正确项（含 ~30% 陷阱题）

为了校准数据质量，建议**全程使用四选一模式**，因为：
- 二选一存在"过度自信"偏差（用户倾向于选认识）
- 多选题能提供更可靠的标签
- 陷阱题可以有效检测"假装认识"

---

## 4. 抽样策略

### 4.1 总词数

从 21,738 个词中抽取 **~2000 个测试词**。
每个测试者从这 2000 个词中随机分配 ~50-150 题。

### 4.2 幂律加权抽样（推荐）

采用**按词频密度不均匀抽样**：高频词（低 rank）抽得多，低频词（高 rank）抽得少。

#### 4.2.1 动机

1. **高频词占比大、对理解影响大** — 1k 桶的 840 个最常用词覆盖了日常交流的绝大多数场景，其认知率对词汇量估计的贡献最大
2. **低频词区分度低** — 30k 桶的词即使母语者也未必全认识，样本量过多只会增加噪音而不提升精度
3. **采样密度应匹配 wordfreq 的自然分布** — 自然语言词频遵循 Zipf 定律（幂律分布），采样密度也应呈幂律衰减

#### 4.2.2 幂律权重公式

每个桶 b 的采样权重：

```
weight_b = size_b × (1 / median_rank_b) ** power
```

其中：
- `size_b` = 桶 b 中的词数
- `median_rank_b` = 桶 b 中所有词的 rank 中位数（例如 1k 桶的中位数约 490）
- `power` = 幂律指数（默认 0.7，可配置 0.5 ~ 1.0）
  - `power=0`：按桶大小均匀分配（相当于不分高低频）
  - `power=1.0`：强烈偏向高频词
  - `power=0.7`：推荐值，在覆盖度和精度间取得平衡

桶 b 的采样配额：

```
n_b = max(1, round(N_total × weight_b / Σ_weight))
```

#### 4.2.3 实际分配示例（power=0.7, total=3000）

| 桶 | 词数 | 中位数 rank | 权重 | 分配词数 | 占比 |
|----|:---:|:----------:|:----:|:-------:|:---:|
| 1k | 840 | 490 | 10.99 | 736 | 24.5% |
| 2k | 775 | 1487 | 4.66 | 312 | 10.4% |
| 3k | 743 | 2513 | 3.10 | 207 | 6.9% |
| 5k | 1457 | 3992 | 4.39 | 294 | 9.8% |
| 8k | 2146 | 6520 | 4.59 | 307 | 10.2% |
| 10k | 1435 | 8985 | 2.45 | 164 | 5.5% |
| 15k | 3526 | 12506 | 4.78 | 320 | 10.7% |
| 20k | 3588 | 17498 | 3.84 | 257 | 8.6% |
| 30k | 7228 | 25044 | 6.03 | 403 | 13.4% |
| **合计** | **21738** | — | **44.83** | **3000** | **100%** |

高频桶（1k-3k）获得了近 42% 的样本，确保了最常用词的认知率估计精度。
低频桶（15k-30k）仍有约 33% 的样本，足以提供区分度。

#### 4.2.4 每个测试者的试题分配

每个测试者做 N_q 题（N_q ≈ 50-150）：
- 从所有桶中按上述比例分配试题
- 用 `random.Random` 和 seed 保证重现性
- 避免重复词（去重）

**两阶段测试**（已实现）：
- Stage 1: 每个桶 4 题 → 9×4=36 题
- Stage 2: 对认知率 ∈ [0.2, 0.8] 的桶额外加题，每个桶 +8 题
- 总 ≈ 36 + ~40 = ~76 题（可配置）

### 4.3 采样实现

完整实现在 `optim/interval_sampler.py` 中：

- `weighted_sample_words(bank, total_samples=3000, power=0.7, seed=None)` — 幂律加权抽样
- `describe_weighted_allocation(bank, total_samples=3000, power=0.7)` — 查看各桶分配配额
- `sample_words(bank, interval=50, per_group=2, seed=None)` — 均匀间隔抽样（备选）

核心逻辑：

```python
def weighted_sample_words(vocab_bank, total_samples=3000, power=0.7, seed=None):
    # 1. 对每个桶计算中位数 rank
    # 2. 权重 = size × (1 / median_rank) ** power
    # 3. 按权重比例分配 total_samples 到各桶
    # 4. 桶内随机采样（无放回）
    return sampled_words
```

```bash
# 查看分配方案（不实际采样）
python3 -m optim.interval_sampler --weighted --describe

# 执行采样
python3 -m optim.interval_sampler --weighted --power 0.7 --total-samples 3000 --seed 42
```

---

## 5. 参数优化算法

### 5.1 参数一览

| 参数 | 符号 | 当前值 | 是否训练 | 说明 |
|------|------|--------|---------|------|
| Logistic 截距（每个用户） | α_j | 按用户拟合 | **否**（按用户独立拟合） | 每个测试者单独拟合，不进入全局优化 |
| Logistic 斜率 | β | 按用户拟合 | **是**（全局共享） | 知识衰减率，从数据中学习 |
| tanh 速率 | k | 0.0000691 | **是** | 控制压缩曲线陡度 |
| 分段 knots | (b_i, s_i) | (3000,1.0), (8000,0.45), (22000,1.28) | **是** | 分段边界与坡度 |
| L2 正则 | λ | 1.0 | **否**（可选） | 保留手动设定 |

### 5.2 损失函数

对于每个测试者 j，我们在桶 b 上有：
- **观测认知率** `r_jb` = (桶 b 中认识词数) / (桶 b 中答题数)，使用 Beta 先验平滑
- **预测认知率** `p_jb` = (1/N_b) × Σ_{r in bucket_b} sigmoid(α_j + β × log(r))

预测的词汇量：`V_j = calibration( Σ_r sigmoid(α_j + β×log(r)) )`

总损失（所有测试者 × 所有桶）：

```
L = Σ_j Σ_b w_b × (p_jb - r_jb)² + λ_β × β²

其中：
  w_b = N_b / N_total（桶权，使大桶贡献更大）
  λ_β = L2 正则强度
```

或者，我们也可以直接在**词汇量空间**定义损失：

```
L = Σ_j (V_j_pred - V_j_true)²
```

但 V_j_true 未知（因为我们正在校准的就是从 raw 到 calibrated 的映射），
所以更可操作的方案是在**认知率空间**定义损失。

### 5.3 梯度下降流程

#### 5.3.1 双层优化

```
外层循环（全局参数 β, k, knots）：
    内层循环（每个测试者 j）：
        固定 β, k, knots
        优化 α_j 使预测与观测匹配
    计算全局损失 L
    计算 L 对 β, k, knots 的梯度
    更新 β, k, knots
```

#### 5.3.2 详细流程

```python
def train_calibration(
    responses_by_user: dict[int, list[Response]],
    bank: VocabBank,
    bucket_labels: list[str],
    n_iter: int = 500,
    lr: float = 0.001,
):
    """
    用所有测试者的数据训练校准参数。

    Args:
        responses_by_user: {user_id: [(word, known), ...]}
        bank: 词汇库
        bucket_labels: 桶标签列表，如 ["1k", "2k", ..., "30k"]

    Returns:
        训练后的参数: (beta, calibration_k, piecewise_knots)
    """

    # ---- 初始化参数 ----
    beta = 0.0                # logistic 斜率（每个 rank 翻倍的认知衰减）
    cal_k = 0.0000691         # tanh 速率（初始值）
    knots = [                 # 分段 knots: (boundary, slope)
        (3000,  1.00),
        (8000,  0.45),
        (22000, 1.28),
    ]
    max_v = 20000             # 母语者上限（固定）

    # ---- 预处理：每个用户的桶级观测 ----
    user_bucket_rates = {}    # {user_id: {bucket_label: observed_rate}}
    user_items = {}           # {user_id: list[PreparedResponse]}

    for uid, responses in responses_by_user.items():
        prepared = prepare_responses(responses, bank)
        user_items[uid] = prepared
        rates = compute_bucket_rates(prepared, bucket_labels, bank)
        user_bucket_rates[uid] = rates

    # ---- 梯度下降主循环 ----
    optimizer = Adam(lr=lr)   # 或 SGD with momentum

    for epoch in range(n_iter):
        total_loss = 0.0

        # ---- 外层：全局参数 ----
        # 使 β, k, knots 可微分
        # knots 有约束：边界必须递增，slope 必须为正

        grad_beta = 0.0
        grad_k = 0.0
        grad_knots = np.zeros(len(knots) * 2)  # [b1, s1, b2, s2, ...]

        # ---- 内层：每个用户 ----
        for uid in responses_by_user:
            prepared = user_items[uid]
            observed_rates = user_bucket_rates[uid]

            # ---- 1. 为当前用户优化 alpha ----
            # alpha 是每个用户的"能力参数"
            # 用梯度下降快速拟合：固定 β, k, knots，只调 α
            alpha = fit_alpha(
                prepared, bank, bucket_labels,
                beta, cal_k, knots, max_v,
            )

            # ---- 2. 预测该用户的桶级认知率 ----
            predicted_rates = predict_bucket_rates(
                alpha, beta, bank, bucket_labels,
            )

            # ---- 3. 计算 loss 和梯度 ----
            for bi, bucket in enumerate(bucket_labels):
                p_jb = predicted_rates[bucket]
                r_jb = observed_rates[bucket]
                err = p_jb - r_jb
                bucket_weight = len(bank.words_by_bucket[bucket]) / len(bank)

                total_loss += bucket_weight * err ** 2

                # 梯度 ∇_β L
                # dL/dβ = 2 * w_b * err * dp_jb/dβ
                #   dp_jb/dβ = (1/N_b) * Σ p * (1-p) * log(r)
                dp_dbeta = bucket_derivative_dbeta(
                    alpha, beta, bank, bucket
                )
                grad_beta += 2 * bucket_weight * err * dp_dbeta

                # 梯度 ∇_k L — 需要链路法则经过 tanh
                # dL/dk = dL/dp * dp/dV_cal * dV_cal/dk
                # V_cal = max_v * tanh(k * V_raw)
                # dV_cal/dk = V_raw * max_v * sech²(k * V_raw)
                if cal_k is not None:
                    V_raw = compute_raw_estimate(alpha, beta, bank)
                    grad_k += 2 * bucket_weight * err * (
                        dp_jb_dV_cal * V_raw * max_v * (
                            1.0 / math.cosh(cal_k * V_raw) ** 2
                        )
                    )

                # 梯度 ∇_knots L — 经过 piecewise 层
                # 每条分段斜率的梯度：累加通过该段的用户预测词汇量
                # 详细推导见附录

        # ---- 4. 参数更新 ----
        beta -= lr * (grad_beta + 2 * lambda_l2 * beta)
        cal_k -= lr * grad_k
        knots = update_knots(knots, grad_knots, lr)

        # 约束检查
        beta = max(beta, -5.0)   # 防止过度衰减
        cal_k = max(cal_k, 1e-8) # 正数
        knots = enforce_knot_constraints(knots)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: loss = {total_loss:.4f}, "
                  f"β = {beta:.4f}, k = {cal_k:.7f}")

    return beta, cal_k, knots
```

### 5.4 自动微分实现

手动推导梯度公式容易出错。推荐实现方案：

#### 方案 A：PyTorch 自动微分（推荐）

```python
import torch

def train_with_torch(responses_by_user, bank, bucket_labels):
    """用 PyTorch 自动微分训练全局参数。"""

    # 参数（requires_grad=True）
    beta = torch.tensor(0.0, requires_grad=True)
    cal_k = torch.tensor(0.0000691, requires_grad=True)
    # knots: [b1, s1, b2, s2, b3, s3]
    knots = torch.tensor([3000., 1.0, 8000., 0.45, 22000., 1.28],
                         requires_grad=True)

    optimizer = torch.optim.Adam([beta, cal_k, knots], lr=0.001)

    for epoch in range(500):
        optimizer.zero_grad()
        total_loss = 0.0

        for uid, responses in responses_by_user.items():
            # 内层：拟合 α
            alpha = fit_alpha_torch(uid, responses, bank, beta, bucket_labels)

            # 预测词汇量
            V_raw = compute_raw_estimate_torch(alpha, beta, bank)
            V_cal = calibration_torch(V_raw, cal_k, knots, max_v=20000.0)

            # 预测桶级认知率
            p = predict_bucket_rates_torch(alpha, beta, bank, bucket_labels)
            r = observed_bucket_rates[uid]  # tensor

            # MSE 损失
            w = bucket_weights
            loss = (w * (p - r) ** 2).sum()
            total_loss += loss

        total_loss.backward()
        optimizer.step()

        # 约束
        with torch.no_grad():
            knots.clamp_(min=0.0)  # 正边界和斜率

    return beta.item(), cal_k.item(), knots.detach().numpy()
```

#### 方案 B：NumPy 手动梯度

如果不想引入 PyTorch 依赖，用 NumPy + scipy.optimize 也是可行的：

```python
from scipy.optimize import minimize

def objective(params, data):
    beta, k, *knot_vals = params
    # 重构 knots
    knots = [(knot_vals[i], knot_vals[i+1])
             for i in range(0, len(knot_vals), 2)]
    loss = compute_loss(data, beta, k, knots)
    return loss

result = minimize(objective, x0=initial_params, args=(data,),
                  method="L-BFGS-B",
                  bounds=constraints)  # β∈(-5,0), k>0, slope>0
```

### 5.5 调参前准备：合成数据验证

在采集真实数据之前，先用**合成测试**验证梯度下降流程的正确性：

```python
def synthetic_data_validation():
    """用已知的参数生成合成数据，看能否恢复出这些参数。"""

    # 1. 设定"真实"参数
    true_beta = -0.3
    true_k = 0.00008
    true_knots = [(3000, 1.0), (8000, 0.4), (22000, 1.3)]

    # 2. 用这些参数生成多个"虚拟测试者"
    users = {}
    for ability_level in range(20):  # 20 个不同水平
        alpha = ability_level * 0.3 - 5  # 从低到高
        responses = simulate_responses(alpha, true_beta, bank)
        users[f"synth_{ability_level}"] = responses

    # 3. 运行梯度下降
    beta_est, k_est, knots_est = train_calibration(users, bank, ...)

    # 4. 验证：恢复的参数与真实参数接近
    assert abs(beta_est - true_beta) < 0.05
    assert abs(k_est - true_k) < 1e-5
    assert all(abs(k_est[i] - true_knots[i][1]) < 0.05
               for i in range(len(true_knots)))
```

### 5.6 交叉验证

在真实数据上，用 **k-fold 交叉验证** 评估参数稳定性：

```
Fold 1: 训练集 80% → 参数 → 测试集 20% 上计算 MSE
Fold 2: ...
Fold 5: ...

平均 MSE 和参数标准差 → 评价参数可靠性
```

---

## 6. 数据集格式

### 6.1 原始响应表（SQLite / CSV）

```sql
-- SQLite schema
CREATE TABLE calibration_responses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,       -- 映射到 students 表
    word        TEXT NOT NULL,
    rank        INTEGER NOT NULL,      -- 该词的频率 rank
    bucket      TEXT NOT NULL,          -- "1k", "2k", ...
    known       BOOLEAN NOT NULL,       -- 1=认识, 0=不认识
    quiz_mode   TEXT,                   -- "binary" | "multiple_choice"
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES students(id)
);

CREATE TABLE calibration_students (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    cet_score   INTEGER,        -- 可选，用于外部验证
    self_level  TEXT,           -- 自评等级: "初中"..."母语"
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE calibration_answers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    word        TEXT NOT NULL,
    attempt     INTEGER DEFAULT 1,  -- 同一词的答题次数
    is_correct  BOOLEAN NOT NULL,
    response_ms INTEGER,             -- 答题反应时（毫秒），可选
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES calibration_students(id)
);
```

### 6.2 聚合特征表（用于优化）

```sql
CREATE TABLE calibration_aggregated (
    user_id     INTEGER NOT NULL,
    bucket      TEXT NOT NULL,       -- "1k", "2k", ...
    asked       INTEGER NOT NULL,    -- 该桶答题数
    known       INTEGER NOT NULL,    -- 认识数
    known_rate  REAL NOT NULL,       -- 已平滑的认知率
    alpha       REAL,                -- 优化后该用户的 α
    PRIMARY KEY (user_id, bucket),
    FOREIGN KEY (user_id) REFERENCES calibration_students(id)
);
```

### 6.3 导出 JSON 格式

```json
{
  "dataset_meta": {
    "version": "1.0",
    "n_users": 200,
    "n_words_sampled": 2000,
    "n_total_words_in_bank": 21738,
    "collection_date": "2026-06",
    "bucket_boundaries": [1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000]
  },
  "users": [
    {
      "user_id": 1,
      "label": "初中",
      "responses": [
        {"word": "the", "rank": 1, "bucket": "1k", "known": true},
        {"word": "analysis", "rank": 500, "bucket": "1k", "known": false},
        {"word": "sustain", "rank": 2500, "bucket": "3k", "known": false}
      ]
    }
  ],
  "aggregated": [
    {
      "user_id": 1,
      "bucket": "1k",
      "asked": 15,
      "known": 14,
      "smoothed_rate": 0.91
    },
    {
      "user_id": 1,
      "bucket": "3k",
      "asked": 12,
      "known": 5,
      "smoothed_rate": 0.42
    }
  ],
  "trained_parameters": {
    "beta": -0.285,
    "calibration_k": 0.0000792,
    "piecewise_knots": [
      [3000, 1.0],
      [7500, 0.42],
      [22000, 1.35]
    ],
    "training_loss": 0.0173,
    "n_epochs": 500,
    "cross_val_mse": 0.0191
  }
}
```

### 6.4 新版配置格式

优化完成后，新的参数应输出为可直接加载的配置：

```python
# vocab_estimator/config.py 中的更新

@dataclass(frozen=True)
class EstimatorConfig:
    ...
    # === 以下参数由 calibration pipeline 训练得到 ===
    calibration_k: float = TRAINED_K        # 从数据中学习
    piecewise_knots: tuple = TRAINED_KNOTS  # 从数据中学习
    logistic_beta_global: float = TRAINED_BETA  # 可选：全局 β
    enable_global_beta: bool = True          # 是否使用全局 β
```

---

## 7. 实施路线图

### Phase 0：数据收集基础设施（1-2 天）

- [x] 已有 SQLite 存储（`server/database.py`）
- [x] 已有答题 API（`/api/tests/save`）
- [ ] 添加 `calibration_*` 表到数据库 schema
- [ ] 新 API `/api/calibration/batch-save` — 批量导入答题记录
- [ ] 新 API `/api/calibration/export` — 导出用于训练的聚合数据

### Phase 1：合成数据验证（2-3 天）

- [ ] 实现 `synthetic_data_validation.py`
  - 已知参数 → 合成响应 → 优化 → 验证恢复精度
- [ ] 验证梯度下降算法正确收敛
- [ ] 确认不同初始化下参数稳定性
- [ ] 确定最优学习率和迭代次数

**Checkpoint 1:** 能在合成数据上以 <5% 误差恢复真实参数

### Phase 2：梯度下降实现（2-3 天）

- [ ] 实现 `optim/calibration_trainer.py`
  - 使用 PyTorch 自动微分（推荐）或 NumPy 手动梯度
  - 双层优化：外层全局参数，内层每用户 α
  - Adam 优化器，学习率衰减
- [ ] 实现 `compute_bucket_rates()` — 桶级聚合 + Beta 平滑
- [ ] 实现 k-fold 交叉验证
- [ ] 输出：新的 `β, k, knots`

**Checkpoint 2:** 在真实（或高保真合成）数据上跑通完整优化管线

### Phase 3：数据采集（1-2 周）

- [ ] 部署测试页面（使用已有 Web 界面）
- [ ] 向 50-200 名测试者分发测试
- [ ] 确保覆盖各能力层（低/中/高）
- [ ] 收集每条答题的反应时间（可选，用于数据质量过滤）
- [ ] 质量控制：过滤 <10 秒答题的用户、异常模式检测

**Checkpoint 3:** 收集到至少 100 名有效测试者的数据

### Phase 4：训练与验证（1-2 天）

- [ ] 运行梯度下降训练
- [ ] 交叉验证评估参数稳定性
- [ ] 与当前启发式参数对比：
  - 训练集 MSE（拟合优度）
  - 留出测试集 MSE（泛化能力）
  - 新参数在不同测试者上的预测曲线形状
- [ ] 人工审核：新参数是否产生合理的认知率曲线

**Checkpoint 4:** 新参数在测试集上的 MSE 显著低于当前启发式参数

### Phase 5：集成与部署（1 天）

- [ ] 将训练好的参数写入 `config.py`
- [ ] 更新 `estimator.calibrate()` 以支持全局 β
- [ ] 运行完整的回归测试
- [ ] 对比新旧参数对已有测试数据的估计结果

**Checkpoint 5:** 新参数上线，官方校准管线完成

### Phase 6：持续改进（可选）

- [ ] 建立数据采集 → 参数更新的自动管道
- [ ] 每收集 50 名新测试者自动重新训练
- [ ] 监控参数漂移
- [ ] 添加更多可训练参数（如每个桶的独立权重）

---

## 8. 可行性分析

### 8.1 需要多少测试者？

| 场景 | 测试者数 | 可靠性 |
|------|---------|--------|
| 最低可行 | 30-50 | 低 — 可训练 β，但 k 和 knots 方差大 |
| 推荐 | 100-200 | 中 — 覆盖 6+ 能力层，参数标准差 < 10% |
| 高信度 | 500+ | 高 — 稳健估计，可以交叉验证并实施后续改进 |

**结论：** 每能力层至少 10 人，6 个能力层 = **至少 60 人**。
建议目标 **100-200 人**以获得统计可靠的结果。

### 8.2 需要多少次迭代？

| 优化器 | 典型迭代数 | 每次迭代耗时（100 用户） |
|--------|-----------|----------------------|
| Adam (PyTorch) | 300-500 | <1 秒 |
| L-BFGS (scipy) | 50-100 | <5 秒 |
| SGD + momentum | 500-2000 | <2 秒 |

**结论：** 在 CPU 上几分钟内即可收敛，不需要 GPU。

### 8.3 收敛性保障

- 损失函数是参数的连续可微函数（MSE + sigmoid + tanh + 分段线性）
- 分段线性操作有定义良好的次梯度
- Adam 优化器对分段线性操作具有良好的适应性
- 约束条件（β < 0, k > 0, 正斜率）确保物理意义

**潜在收敛问题：**
1. **初始化敏感**：分段 knots 对初始化敏感，建议用当前经验值初始化
2. **局部最优**：建议用 3-5 个不同随机种子跑优化，选取最低损失
3. **冷启动**：如果没有低能力测试者，高频桶的参数可能不准确

### 8.4 与当前启发式方法对比

| 维度 | 启发式校准（当前） | 数据驱动校准（新方案） |
|------|------------------|---------------------|
| 参数设定 | 人工猜测 + 手动调校 | 从 ~200 名测试者的数据学习 |
| 过拟合风险 | 低（参数少） | 中（需交叉验证） |
| 泛化能力 | 可能欠拟合真实情况 | 预期更好 |
| 可维护性 | 手动调整麻烦 | 数据更新后自动重训练 |
| 对单个桶的拟合 | 未经检验 | 直接最小化桶级 MSE |
| 认知率曲线 | 凭经验 | 由数据决定 |

**预期改进：**

- 在新测试者上的预测**MSE 降低 30-50%**
- 各个认知能力层的桶级认知率曲线更贴合实际
- 校准参数有了统计依据，而不是"感觉大致如此"

### 8.5 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| 数据质量差（用户乱点） | 中 | 高 | 过滤无效用户（答题时间过短、全选一） |
| 能力层覆盖不足 | 中 | 中 | 针对性地招募不同水平测试者 |
| 过拟合到采集群体 | 中 | 中 | 留出 20% 测试集 + 交叉验证 |
| 参数相关性高导致不稳定 | 低 | 中 | 正则化 + 先验约束 |
| 分段 knots 难以优化 | 低-中 | 中 | 先用合成数据验证，考虑用 3 次样条替代 |

---

## 9. 附录：数学推导

### 9.1 Logistic 桶级认知率的梯度

设 `x_r = log(rank_r)`，对于桶 b：

```
p_b = (1/N_b) × Σ_{r in bucket_b} σ(α + β × x_r)

其中 σ(z) = 1/(1+e^(-z))
```

梯度：

```
∂p_b/∂β = (1/N_b) × Σ_{r in bucket_b} σ(α+βx_r) × (1-σ(α+βx_r)) × x_r

∂p_b/∂α = (1/N_b) × Σ_{r in bucket_b} σ(α+βx_r) × (1-σ(α+βx_r))
```

### 9.2 tanh 校准的梯度

```
V_cal = V_max × tanh(k × V_raw)

dV_cal/dk = V_raw × V_max × sech²(k × V_raw)
          = V_raw × V_max × (1 - tanh²(k × V_raw))

dV_cal/dV_raw = V_max × k × sech²(k × V_raw)
```

经过 tanh 的校准值再经过 piecewise 层。完整链路：

```
dV_cal_combined/dk = (dV_cal/dk) × (d(piecewise_output)/d(V_cal))
```

其中 `d(piecewise_output)/d(V_cal)` 就是 V_cal 所处段的 slope。

### 9.3 Piecewise 层的梯度

```
piecewise(x) = ∫ slope(t) dt  over [0, x]

其中 slope(x) = s_i  for b_{i-1} < x ≤ b_i

d(piecewise(x))/dx = slope at x
```

参数 `b_i`（分段边界）的梯度计算更复杂，因为边界移动会使输出跳跃。
可用**有限差分**近似：

```
∂L/∂b_i ≈ (L(b_i + ε) - L(b_i - ε)) / (2ε)
```

或使用次梯度（重参数化技巧）。

在实际应用中，建议对边界参数使用较慢的学习率，或先用合成数据固定边界后再优化斜率。

### 9.4 Beta 先验平滑

观测认知率用 Beta 先验平滑以减少小样本噪音：

```
smoothed_rate = (known + α_b) / (asked + α_b + β_b)

其中：
  α_b = α_base + β_prior_max × (1 - t)
  β_b = β_base + β_prior_max × t
  t = bucket_index / (n_buckets - 1)
```

### 9.5 权重的设计

桶级 MSE 按桶大小加权：

```
L = Σ_j Σ_b (N_b / N_total) × (p_jb - r_jb)²
```

这样保证：
- 大桶（30k: 7228 词）的拟合更受重视
- 小桶（1k: 840 词）不会主导损失函数
- 整体词汇量估计的误差更均匀

---

## 附录 A：文件名索引

使用 Codex CLI 进行实现时的文件参考：

| 文件 | 用途 |
|------|------|
| `vocab_estimator/config.py` | 参数配置（更新后的参数写回这里） |
| `vocab_estimator/vocab_model.py` | 核心模型（calibrate、logistic 拟合） |
| `vocab_estimator/vocab_bank.py` | 词汇库（桶划分、rank 查询） |
| `vocab_estimator/sampler.py` | 抽样（balanced_sample、adaptive_sample） |
| `docs/calibration_pipeline.md` | 本文档 |
| `optim/calibration_trainer.py` | **新文件**：梯度下降训练器 |
| `optim/synthetic_generator.py` | **新文件**：合成数据生成器 |
| `optim/cross_validate.py` | **新文件**：交叉验证 |
| `tests/test_calibration_optim.py` | **新文件**：优化管线的单元测试 |

---

## 附录 B：基于等间隔抽样的参数校准（2026-06-23 修订）

### B.1 动机

原设计方案（附录 A）基于分层抽样 + 桶级认知率损失。在实现和使用过程中发现：
- 分层抽样的桶边界选择有主观性，可能引入偏倚
- 桶级聚合损失对低频桶的信息利用不够充分
- 缺乏官方考试大纲词表作为锚点，词汇量估计缺乏外部验证

经讨论，决定采用 **等间隔抽样 + 官方词表锚点** 的新方案。

### B.2 采样策略

#### B.2.1 间隔采样

不再使用桶级分层，改为从 vocab_bank 的 30k 个 rank 中**等间隔选组**：

```
每间隔 interval 个 rank 选一组（interval = 50 或 100）
每组随机抽 per_group 个词（per_group = 2）

interval=50 时：30000/50 × 2 = 1200 个测试词
interval=100 时：30000/100 × 2 = 600 个测试词
```

用户原话：「预采样抽3000个，间隔50或者100个抽一组，每一组随机抽两份」

#### B.2.2 算法

```python
def sample_words(vocab_bank, interval=50, per_group=2):
    words = []
    for start_rank in range(1, 30001, interval):
        end_rank = min(start_rank + interval, 30001)
        group_words = vocab_bank.get_words_in_rank_range(start_rank, end_rank)
        sampled = random.sample(group_words, min(per_group, len(group_words)))
        words.extend(sampled)
    return words
```

#### B.2.3 优势

- 无偏性：每个 rank 段被等概率覆盖
- 可配置：interval 和 per_group 可调
- 均匀覆盖：从低频到高频全程覆盖，无桶边界处的跳变

### B.3 官方词表锚点

#### B.3.1 目标

找到官方考试大纲词表，在训练中给这些词更高权重：

| 词表 | rank 范围（估计） | 词汇量 | 来源 |
|------|------------------|--------|------|
| 中考 | 1-1500 | ~1500-2500 | 义务教育课标 |
| 高考 | 1500-2500 | ~2500-4000 | 普通高中课标 |
| 四级 | 2500-4000 | ~4000-5500 | CET-4 大纲 |
| 六级 | 4000-5500 | ~5500-7500 | CET-6 大纲 |

这些词表的覆盖范围是已知公开信息。训练时对官方词表中的词给予**更高权重（2x-5x）**，让模型对这些锚点的预测更精确。

#### B.3.2 权重策略

对于官方词表中的每个词：
```
loss_weight = 1.0（普通词）
loss_weight = 3.0（官方词表词，中考/高考/四级/六级）
```

训练损失中，这些词的认知率误差将会被放大，推动模型在这些锚点上拟合更精确。

### B.4 参数优化目标

#### B.4.1 优化参数

| 参数 | 符号 | 初始值 | 说明 |
|------|------|--------|------|
| Logistic 斜率 | β | -0.30 | 知识衰减率 |
| tanh 速率 | k | 0.0000691 | 压缩曲线陡度 |
| 分段 knots | (b_i, s_i) | (3000,1.0), (8000,0.45), (22000,1.28) | 分段边界与坡度 |
| L2 正则 | λ | 1.0 | 正则强度 |

#### B.4.2 优化目标

1. **间隔组认知率**：每个 rank 间隔组的「认识概率」→ 期望该组两个词都被认识的概率 ≈ 该 rank 水平的认知率
2. **官方词表覆盖率**：官方词表的整体覆盖率 → 词汇量估计值应与该官方等级匹配
3. **认知率递减曲线**：认知率随 rank 的递减曲线 → 与传统经验曲线一致

#### B.4.3 损失函数

```
Loss = w_interval × MSE(间隔组预测认知率, 实际率)
     + w_official × MSE(官方词表覆盖率预期, 实际)
     + w_smooth × 正则化项(确保曲线平滑)
```

##### 间隔组损失

对于每个间隔组 g（包含 per_group 个词）：
```
p_pred_g = mean_{w in group_g} sigmoid(α + β×log(rank_w))
p_obs_g = (group_g 中被认识的词数) / len(group_g)

L_interval = Σ_g (p_pred_g - p_obs_g)²
```

##### 官方词表损失

对于每个官方词表 t ∈ {中考, 高考, 四级, 六级}：
```
# 词表中所有词的预测认知率
p_pred_t = mean_{w in vocabset_t} sigmoid(α + β×log(rank_w))
# 期望值：在对应能力水平的学习者应该认识 75-95% 的词
p_exp_t = 0.85（默认）
weight_t = 3.0（高权重）

L_official = Σ_t weight_t × (p_pred_t - p_exp_t)²
```

##### 平滑正则项

确保认知率随 rank 递减的曲线平滑：
```
L_smooth = λ × Σ_i |p_rank_{i+1} - p_rank_i|²
```

#### B.4.4 完整梯度下降流程

```
for epoch in range(n_epochs):
    for each user:
        1. 内层：拟合 α（固定 β, k, knots）
        2. 计算间隔组认知率预测和实际
        3. 计算官方词表覆盖率
        4. 计算总损失
        5. 反向传播梯度到 β, k, knots
    6. Adam 更新全局参数
    7. 约束检查（β 负值, k 正值, slope 正值）
```

### B.5 文件结构

| 文件 | 用途 |
|------|------|
| `optim/interval_sampler.py` | **新** — 间隔采样器 |
| `optim/official_vocab.py` | **新** — 官方词表锚点 |
| `optim/calibration_trainer.py` | **更新** — 加权损失函数 |
| `docs/calibration_pipeline.md` | **更新** — 本文档 |

### B.6 实施路线图

- [x] 等间隔采样器实现
- [x] 官方词表匹配器实现
- [x] 加权损失函数训练器
- [x] 验证：`--dry-run` 模式打印采样结果和参数结构

---

*文档版本：1.1*
*最后更新：2026-06-23*
