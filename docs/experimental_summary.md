# 词汇量估算项目实验总结

> 综合记录所有实验、模拟和设计决策，供论文/报告引用。

---

## 1. 项目概览

**目标**：构建一个基于 Rasch 模型的英语词汇量自适应测试系统，支持中文英语学习者。

**核心资产**：
- `stage_vocab.json`：11418 词，每词含 difficulty [0,1]、cluster_20 [0-19]、cluster_100 [0-99]、翻译、教育阶段标签
- VocabBank：21738 词（含 wordfreq rank）
- FastAPI 线上服务 (154.9.242.48:7860)
- 前端 web 页面 (web/index.html)

**关键代码模块**：

| 模块 | 功能 |
|------|------|
| `vocab_estimator/stratified_quiz.py` | StratifiedQuiz — 两阶段分层 Rasch 测验 |
| `vocab_estimator/vocab_model.py` | VocabEstimator — Beta 平滑 + 逻辑回归旧模型 |
| `vocab_estimator/sampler.py` | VocabularySampler — 旧版自适应采样 |
| `vocab_estimator/difficulty.py` | 难度评分器 — 教育阶段 + wordfreq rank 融合 |
| `vocab_estimator/article_estimator.py` | 文章估算模块 |
| `server/main.py` | FastAPI 服务 |
| `web/app.js` | 前端交互 |

---

## 2. 模型架构

### 2.1 Rasch 模型

单参数逻辑斯蒂模型（1PL IRT）：

```
P(word_j known | θ) = σ(θ - d_j)
σ(x) = 1 / (1 + exp(-x))
```

- `θ`：用户能力参数（logit 尺度）
- `d_j`：词 j 的难度参数（logit 尺度）
- `d_j = logit(difficulty)`，difficulty ∈ [0, 1] 归一化
- N(0, 2) 先验防极端

**词汇量估计**：

```
vocab_size = Σ_j σ(θ - logit(difficulty_j))    × 0.8 (经验校准系数)
```

对所有 11418+ 词求和，0.8 是经验校准（原始估算偏高约 25%）。

### 2.2 分层采样 (StratifiedQuiz)

两阶段设计：

**Phase 1** — 40 题，20 个 cluster_20 类 × 2 题
- 5 个 diagnostic spread 类（c20: 0, 5, 10, 15, 19）用"extremes"策略（选类内最易/最难题）
- 其余 15 类用"balanced"策略（难度均匀分布）
- 顺序随机打乱

**Phase 2** — 低置信类细化
- 对 Phase 1 每类统计答对数（0/2=确信不知，2/2=确信知道，1/2=不确定）
- 不确定的类别追加 n_per_class=8 题，选 Fisher 信息量最大的词
- 平均追加 ~44 题

**Rasch MLE 拟合**：
- Newton-Raphson 迭代，多起点（θ₀, 0, ±2, ±5）
- Fisher 信息量 SE = 1/√Σ(σ·(1-σ))

---

## 3. 主模拟评估 (2026-06-24)

### 3.1 方法

- 生成 2000 个合成用户（seed=42）
- 真词汇量 [1000, 15000] 均匀采样
- 每个用户真 θ 通过二分搜索反推（使 Σσ(θ-d) ≈ target）
- 响应生成：P(known) = σ(true_θ - d_j)
- 两阶段全流程 (Phase 1 + Phase 2)

### 3.2 结果

| 指标 | 值 |
|------|:---:|
| MAE | **363** |
| RMSE | 455 |
| R² | **0.977** |
| 相关系数 | 0.989 |
| 平均偏差 | -134 |
| Phase 1 平均题量 | 40.0 |
| Phase 2 平均题量 | 43.4 |

**分桶精度**：

| 词汇量区间 | 用户数 | MAE | R² | 偏差 |
|:---:|:---:|:---:|:---:|:---:|
| 低 (1k-3k) | 369 | 334 | 0.447 | +2 |
| 中 (3k-8k) | 965 | 374 | 0.896 | -91 |
| 高 (8k-15k) | 666 | 363 | 0.796 | -271 |

### 3.3 关键发现

1. **Phase 2 几乎无提升**：Phase 1 (40题) 和 Phase 1+2 (~84题) 的 MAE 基本相同
2. **系统性负偏差**：高中端持续低估（偏差 -91 至 -271），低端稍微高估
3. **CRLB 分析**：MAE=363 接近 Fisher 信息量理论下界（~300 词）

### 3.4 理论下界分析（CRLB）

基于 Fisher 信息量 I(θ) = Σ σ(θ-d_j)(1-σ(θ-d_j))：

- 40 题平均 Fisher 信息：I ≈ 40 × 0.25 = 10 (每题最大贡献 0.25)
- θ 标准误：SE ≈ 1/√10 ≈ 0.316
- 词汇量标准误：SE_vocab ≈ SE 将 θ 映射到词汇量的导数为 ~1000
- 理论最小 MAE ≈ 0.8 × 1000 × 0.316 ≈ 253（乘以 0.8 校准系数）
- 当前 MAE=363 > 253，仍有 ~30% 的优化空间

---

## 4. 混合二分搜索 + CAT 验证 (2026-06-24)

### 4.1 方法

**问题**：二分搜索式出题能否更快收敛到词汇量边界？

**方案 A**：Hybrid Bisection + CAT
- Phase 1 (6题)：quantile 二分搜索
  - 锚点：P5=0.43, P10=0.56, P25=0.74, P50=0.86, P75=0.94, P90=0.98, P95=0.99
  - 初始区间 [P5_idx=0, P95_idx=6]，每步取 midpoint
  - 答对→右移，答错→左移
- Phase 2 (34题)：逐题 Fisher 信息量 CAT
  - 每答一题后 MLE 拟合 θ，选 |θ - d_j| 最小的词

**方案 B**：现有分层 40 题 + Phase 2 细化

### 4.2 结果 (500 用户)

| 指标 | 方案A (40题) | 方案B (~84题) | 差值 |
|:---:|:---:|:---:|:---:|
| MAE | 443 | 365 | +79 |
| R² | 0.961 | 0.976 | -0.015 |
| 相关系数 | 0.982 | 0.989 | -0.007 |

**分桶误差**：

| 词汇量区间 | 方案A (MAE) | 方案B (MAE) | n |
|:---:|:---:|:---:|:---:|
| 低 (1k-3k) | 410 | 348 | 90 |
| 中 (3k-8k) | 490 | 401 | 231 |
| 高 (8k-15k) | 400 | 326 | 179 |

### 4.3 结论

1. **二分搜索不如分层采样**：6 题粗定位在右偏分布中效率低
2. **Difficulty 极度右偏**：P50=0.86 → 第一题就跳到高难度，低水平用户区间快速收缩失去信息
3. **Phase 1 对比不公平**：方案A Phase 1 只有 6 题，方案B 有 40 题
4. **方案A 在 40 题内**已达到 R²=0.961，比方案B的 40 题 Phase 1 略差（0.976 vs 0.961）

---

## 5. 最优题量探索 (2026-06-24)

### 5.1 方法

- 同一用户（seed=42）的 40 题 Phase 1 顺序
- 截取前缀：10, 15, 20, 25, 30, 35, 40
- 300 用户，用 streaming 顺序（先覆盖所有 20 类各一题，再补第二轮）
- 仅用 Phase 1，不做 Phase 2

### 5.2 结果

| 题量 | MAE | RMSE | R² | Corr | 平均偏差 | θ CI 宽 | 词汇 CI 宽 | P90 误差 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 1317 | 1572 | 0.720 | 0.900 | -851 | 3.32 | 5455 | 2413 |
| 15 | 1081 | 1286 | 0.813 | 0.927 | -606 | 2.72 | 4459 | 2036 |
| 20 | 928 | 1101 | 0.863 | 0.943 | -469 | 2.35 | 3797 | 1678 |
| 25 | 819 | 981 | 0.891 | 0.954 | -380 | 2.14 | 3452 | 1534 |
| 30 | 701 | 855 | 0.917 | 0.966 | -346 | 1.95 | 3133 | 1373 |
| 35 | 649 | 790 | 0.929 | 0.971 | -310 | 1.82 | 2888 | 1227 |
| 40 | 596 | 729 | 0.940 | 0.974 | -258 | 1.70 | 2678 | 1124 |

### 5.3 收益递减分析

| 区间 | MAE 下降 | 边际收益 (MAE/题) |
|:---:|:---:|:---:|
| 10→15 | -236 | -47.2 |
| 15→20 | -153 | -30.6 |
| 20→25 | -109 | -21.8 |
| 25→30 | -118 | -23.6 |
| 30→35 | -52 | -10.4 |
| 35→40 | -53 | -10.6 |

### 5.4 建议

- **25 题**为第一门槛：R²=0.89，CI~3450，适合快速初估
- **30 题**为最优折中：R²=0.917，CI~3133，边际收益开始骤降
- **40 题**为全精度：R²=0.94，CI~2678
- **streaming 模式**：按先覆盖所有类的顺序出题，保证前缀信息量

---

## 6. Streaming 渐进式细化设计 (方案)

### 6.1 交互流程

```
Step 1 (15题) ──→ 初估结果 + 置信区间
     │            提示 "继续答题可提高精度"
     ▼
Step 2 (25题) ──→ 更新结果 + 区间缩小
     │            ±xxx 词
     ▼
Step 3 (35题) ──→ 更精细估算
     │
     ▼
Step 4 (40题) ──→ 最终结果
```

### 6.2 后端 API 设计

```
POST /api/vocabulary/quiz-v2/stream
{
  "phase1_responses": [{"word": "abandon", "known": true}, ...]
}
→ {
    "theta": 0.83,
    "theta_ci": [0.37, 1.30],
    "vocab_raw": 3576,
    "vocab_ci": [3112, 4040],
    "n_questions": 15,
    "se": 0.239,
    "continue_available": true,
    "suggested": ["abdomen", "aberration", ...]  // 下一批 5 题
  }
```

### 6.3 前端状态机

```
setup → answering → result_shown → [refine → answering → result_shown]
                                → finish
```

每步显示：
- 当前词汇量估算 ± CI
- 确定性标签（高/中/低）
- 教育阶段映射
- "继续细化"按钮

---

## 7. 文章估算模块 (article_estimator)

### 7.1 功能

- 输入文章 → 输出词汇量估算
- 分词用 `[a-z]+(?:-[a-z]+)*`
- 541 停用词（59→541 扩充）
- lemma 归一化：规则型（-ing/-ed/-s/-es/-ies/-ly/-er/-est/-tion）+ 60+ 不规则词
- 覆盖率字段：`coverage_unique`（unique word 覆盖率）

### 7.2 API

```
POST /api/v2/estimate/article
{
  "article": "...英文文章..."
}
→ {
    "estimated_vocab": 6542,
    "level": "四级",
    "difficulty_median": 0.85,
    "coverage": {"stage_vocab": 0.88, "coverage_unique": 0.92},
    "article_stats": {"total_tokens": 342, "content_tokens": 201, "unique_content_words": 128}
  }
```

---

## 8. 困难度校正实验 (2026-06-24)

### 8.1 目标

用词向量（GloVe/word2vec）和形态学特征，校正基于 rank 的 difficulty 评分。

### 8.2 方法

- MLP 残差校正器：形态学特征（词长、音节数 R²=0.13）
- 训练目标：rank-based difficulty 与实际 difficulty 的残差

### 8.3 问题

- GloVe 840B（2GB）网络下载失败
- 纯形态学特征 R²=0.13，无统计学意义
- **Codex 发现关键 bug**：训练目标（绝对 diff）和 apply（残差叠加）不匹配
- 最终相关系数仅 0.67，异常词 796

### 8.4 下一步

- 用现有 word_embeddings_384d.npy 做 kNN 平滑
- 建人工 gold set（人工标注难度）

---

## 9. 模拟评估方法论

### 9.1 合成用户生成

```python
generate_synthetic_users(n_users=2000, true_min=1000, true_max=15000, seed=42)
```

每个用户：
1. 均匀采样 target_vocab ∈ [1000, 15000]
2. 二分搜索 θ：使 Σσ(θ - logit(d_j)) ≈ target_vocab
3. 合成回答：P(known) = σ(true_θ - logit(difficulty))
4. 每个用户独立 seed（user_seed = py_rng.randrange(0, 2³²)）

### 9.2 回答 RNG

```python
response_rng = random.Random(user.seed ^ 0x9E3779B9)
```

黄金比例常数 0x9E3779B9 确保回答分布与采样分布去相关。

### 9.3 评估指标

| 指标 | 定义 |
|------|------|
| MAE | mean(predicted - true) |
| RMSE | sqrt(mean((predicted - true)²)) |
| R² | 1 - SSE/SST |
| Correlation | Pearson r |
| Mean Bias | mean(predicted - true) |
| P90 Error | 90th percentile absolute error |
| θ CI Width | mean(θ_high - θ_low) |
| Vocab CI Width | mean(vocab_high - vocab_low) |

### 9.4 分桶

| 桶 | 范围 |
|:---:|:---:|
| low_1k_3k | 1000 ≤ true < 3000 |
| mid_3k_8k | 3000 ≤ true < 8000 |
| high_8k_15k | 8000 ≤ true ≤ 15000 |

---

## 10. 关键结果汇总

| 实验 | MAE | R² | 题量 | 备注 |
|:---:|:---:|:---:|:---:|:---|
| Phase 1 only (40题) | 596 | 0.940 | 40 | streaming 300用户 |
| Phase 2 full (~84题) | 363 | 0.977 | ~84 | 2000用户，基准结果 |
| Hybrid bisection + CAT | 443 | 0.961 | 40 | 6 bisection + 34 CAT |
| 25 题初估 | 819 | 0.891 | 25 | streaming 顺序 |
| 30 题最优折中 | 701 | 0.917 | 30 | streaming 顺序 |

---

## 11. 脚本与输出文件索引

| 脚本 | 功能 | 输出 |
|:---:|:---|:---|
| `tests/simulation_eval.py` | 主模拟评估（2000用户） | `outputs/simulation_results_v2.json` |
| `scripts/validate_hybrid_bisection.py` | 二分搜索 vs 分层对比 | `outputs/hybrid_bisection_validation.json` |
| `scripts/explore_question_count.py` | 最优题量探索 | `outputs/question_count_exploration.json` |
| `scripts/validate_stratified.py` | 3用户 profile 验证 | 终端输出 |
| `scripts/train_difficulty_corrector.py` | 难度校正 MLP | `docs/difficulty_correction_eval.md` |

**运行命令**：
```bash
cd ~/stu/vocab_estimator
# 主模拟
python3 tests/simulation_eval.py --n-users 2000

# 混合方案验证
python3 scripts/validate_hybrid_bisection.py

# 题量探索
python3 scripts/explore_question_count.py --n-users 300

# 自定义题量
python3 scripts/explore_question_count.py --n-users 300 --counts 10 20 30 40 --order-policy shuffled
```

---

## 12. 设计文档索引

| 文档 | 内容 |
|:---|:---|
| `docs/stratified_quiz_design.md` | 分层 Rasch 测试完整设计 + 数学推导 |
| `docs/stage_based_model_design.md` | 旧版分层模型设计 |
| `docs/calibration_pipeline.md` | 参数校准管线（梯度下降调参） |
| `docs/article_estimation.md` | 文章估算 API 口径说明 |
| `docs/difficulty_correction_eval.md` | 词向量难度校正实验报告（274行） |
| `docs/difficulty_scoring_design.md` | 难度评分算法设计 |
| `docs/exam_based_calibration.md` | 基于考试词汇表的校准 |
| `docs/experimental_summary.md` | **本文**—全部实验总结 |

---

## 13. 开放问题

1. **0.8 校准系数的合理性**：当前经验值，需要真实用户数据验证
2. **phase2 无用**：Phase 2 追加 44 题几乎不提升精度，应重新设计或删除
3. **低水平用户偏差**：低端（1k-3k）MAE 被动接受，是否需专用筛选题？
4. **难度分布重塑**：当前 distribution [0.05, 1.0] 右偏严重，easy 端缺少题目
5. **论文方向**：渐进式测试（streaming） + 置信区间动态可视化 可作为 HCI 方向
