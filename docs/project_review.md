# 项目阶段性总结（英语词汇量估算工具）

> **代码库**: `~/stu/vocab_estimator/`  
> **审查日期**: 2026-06-23  
> **总代码量**: ~33,389 行（核心 ~8,500 行，翻译表 ~27,402 行，前端 ~787 行）  
> **核心文件**: 22 个 Python 模块 + 3 个前端文件 + 6 套训练参数

---

## 1. 演进路线

```
v1 (2025-12)                        v2 (2026-06)
┌────────────────────────┐          ┌──────────────────────┐
│ wordfreq 词库 (90k)    │          │ stage_vocab.json     │
│ 9 个频段桶             │          │ 11 教育阶段           │
│ 每桶独立 θ + 用户 γ    │          │ 20/100 类聚类         │
│ 桶矩阵模型             │          │ Rasch 模型 (MLE)      │
│ 合成训练数据           │          │ CAT 自适应抽样        │
│ 两阶段自适应测试       │          │ 20 类分层抽样          │
└────────────────────────┘          └──────────────────────┘
         │                                   │
         └────────── 混合运行 ────────────────┘
               server/main.py 同时提供两组 API
               bucket_model.py / stratified_quiz.py 并行可用
```

**里程碑：**
1. **v1 桶矩阵模型** — 用 wordfreq 频率分 9 桶，合成数据训练桶参数，前端 10 热身 + 36 自适应 + Stage2 细化
2. **v2 Rasch 分层模型** — 用 `stage_vocab.json` 的困难度 / 聚类，Rasch MLE 估计 θ，20 类分层抽样，前端 60 题（当前实际返回 25 题）
3. **混合状态** — 两套模型和 API 端点同时存活，前端已切到 v2

---

## 2. 当前架构

### 2.1 数据流图

```
词库加载
  wordfreq 90k ──→ VocabBank (vocab_bank.py) ───→ pickle 缓存 /tmp/vocab_bank_cache.pkl
                                                     ↓
  stage_vocab.json ──→ StratifiedQuiz (stratified_quiz.py)
  11,950 词               ↑ 困难度 / 聚类预计算
  (11 教育阶段)

训练流程 (optim/)
  VocabBank + official_vocab.py ──→ train_bucket_matrix.py ──→ trained_params_bucket.json
  6 种合成数据方案                  PyTorch/NumPy 两种优化器    9 个 θ 参数 + 校准 k + 分段斜率

API 层 (server/main.py)
  v1: /api/estimate → VocabEstimator (vocab_model.py) / bucket_model.py
      /api/vocabulary/quiz → sampler.py (热身+自适应)
      /api/vocabulary/quiz-stage2 → sampler.stage2_refine_sample()
  v2: /api/vocabulary/quiz-v2 → StratifiedQuiz.phase1_sample()
      /api/vocabulary/quiz-v2-stage2 → StratifiedQuiz.phase2_sample()
      /api/vocabulary/quiz-v2/estimate → StratifiedQuiz.estimate_with_ci()

前端 (web/)
  app.js ──→ v2 端点 ──→ 答题汇总 → 提交估算 → 结果展示
                    └──→ stage2 细化 → 合并提交 → 更精确结果
```

### 2.2 模块依赖关系

```
web/index.html ──→ web/app.js ──→ server/main.py
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                     │
         vocab_estimator/       server/              optim/
          vocab_bank.py       database.py        train_bucket_matrix.py
          vocab_model.py      translations.py    official_vocab.py
          bucket_model.py                       calibration_trainer.py
          stratified_quiz.py                    interval_sampler.py
          sampler.py                            synthetic_generator.py
          difficulty.py
          config.py
          coverage.py
          lemmatizer.py
```

---

## 3. 各模块评估

### 3.1 数据层

#### `data/stage_vocab.json` ⭐⭐⭐⭐⭐ (5/5)
- **位置**: `~/stu/vocab_estimator/data/stage_vocab.json`
- **内容**: 11,950 词，覆盖 11 个教育阶段（小学 3/4/5/6, 初中 7/8/9, 高中, 四级, 六级, 雅思），每个词有 `difficulty` (0-1)、`cluster_20`、`cluster_100`、`sources`、`first_stage`
- **难度分布**: `cluster_20` 每个类 ~598 词，均匀分布
- **优点**: 自有标注，不依赖第三方词库；多源融合（PEP 教材、mahavivo、official_vocab）；聚类结构为分层抽样提供天然框架
- **不足**: 词量 11,950 偏少（完整英语词典约 20k-30k 词族）；`difficulty` 只是基于教育阶段的经验值，未用真实用户数据进行校准；重叠矩阵 (`overlap_matrix`) 未在代码中使用

#### `data/exam_vocab/` ⭐⭐⭐⭐ (4/5)
- **文件**: gaokao.txt (3,468), cet6.txt (8,027), coca20000.txt (20,199), gre.txt (6,676), toefl.txt (3,469)
- **用途**: 只在训练的考纲锚点中使用；未集成到推理路径
- **问题**: 文件格式需和 lemmatizer 对齐；coca20000 和 gre/toefl 的官方考试锚点未使用

#### `vocab_estimator/vocab_bank.py` ⭐⭐⭐⭐ (4/5)
- **位置**: `vocab_estimator/vocab_bank.py:1-296`
- **当前**: 加载 wordfreq 90k 词，构建 21,738 个词族的词库，按 9 个频段桶组织
- **性能**: 首次 ~0.57s，二次通过 pickle 缓存 ~0.15s（`server/main.py:65-91`）
- **问题**: wordfreq 词库对中文用户不够准确（中文母语者不需要的罕见词过多，高频生活词汇不足）；`used_fallback` 逻辑（vocab_bank.py 未显示在截取部分，但 server 中有检测）

#### `vocab_estimator/difficulty.py` ⭐⭐⭐ (3/5)
- **位置**: `vocab_estimator/difficulty.py:1-203`
- **内容**: 基于教育阶段的难度打分公式
- **问题**: 公式凭经验设计，未用实际用户答题数据验证；cluster_20/100 的聚类算法不可见（未在 repo 中）

### 3.2 模型层

#### 旧桶矩阵模型 `vocab_estimator/bucket_model.py` ⭐⭐⭐ (3/5)
- **位置**: `bucket_model.py:1-186`
- **核心参数** (`trained_params_bucket.json`):
  - θ 参数: `1k: +9.95, 2k: +7.13, 3k: +4.46, 5k: +1.62, 8k: +0.87, 10k: -0.23, 15k: -2.31, 20k: -4.88, 30k: -8.14`
  - 校准: `k≈0`, `ks=[1.25, 0.84, 1.03]`（几乎恒等映射）
- **优点**: 有训练参数，3 种训练数据变体（freq_only / cet6_hybrid / two_phase）
- **缺点**: 偏大 θ 值（1k 桶 θ=9.95, sigmoid(9.95)≈1.0）意味着高频桶几乎全对；桶间界限硬划分；网格搜索太粗糙（step=0.1）；置信区间用固定 ±15% 近似

#### 新 Rasch 模型 `vocab_estimator/stratified_quiz.py` ⭐⭐⭐⭐ (4/5)
- **位置**: `stratified_quiz.py:1-582`
- **设计**: 20 个 difficulty class, 每个类约 598 词；Rasch MLE 通过 Newton-Raphson（无 scipy 依赖）
- **实现亮点**:
  - `_sigmoid_scalar` 和 `np.clip(x, -40, 40)` 数值稳定
  - Newton-Raphson 内置（无 scipy 依赖，`stratified_quiz.py:226-270`）
  - 信息量感知的 Phase 2 抽样
  - Fisher 信息量计算标准误差
- **已知 bug**: `phase1_sample()` 意图生成 60 题，但实际只生成 ~25 题
  - **根因** (`stratified_quiz.py:100-140`): spread_classes 取 2*5=10 题，剩余每个 c20 类只取 1 题 × 15 个非 spread 类 = 15 题，合计 25 题
  - 代码注释说"3 per cluster"但实现逻辑是每个 c20 类只取 1 题
- **其他问题**: `_word_difficulties` 只包含 stage_vocab 的 11,950 词，不覆盖 wordfreq 的扩展词

#### `vocab_estimator/vocab_model.py` ⭐⭐⭐ (3/5)
- **位置**: `vocab_model.py:1-501`
- **功能**: 逻辑回归 + 贝叶斯平滑 + Bootstrap 置信区间
- **保留原因**: v1 的核心估计器，v2 的 Rasch 模型替代了它；目前 `/api/estimate` 还用它
- **问题**: 复杂度过高（两种回归算法、权重拟合、PAVA、Bootstrap 300 次迭代）；混合了太多策略

#### `optim/official_vocab.py` ⭐⭐⭐⭐ (4/5)
- **位置**: `optim/official_vocab.py:1-51909`（51k 行，大部分是词表数据）
- **内容**: 内置中考→高考→四级→六级嵌套词表，形成课程体系锚点
- **优点**: 完整嵌套（中考 ⊂ 高考 ⊂ 四级 ⊂ 六级），锚点权重 3.0
- **问题**: 词表硬编码在源码中（27k 行翻译 + 51k 行词表 = 78k 行数据在代码里），应移入 `data/` 目录

### 3.3 API 层

`server/main.py` ⭐⭐⭐ (3/5)

| 端点 | 版本 | 模型 | 状态 |
|------|------|------|------|
| `POST /api/estimate` | v1 | bucket_model / VocabEstimator | **已弃用** |
| `POST /api/estimate/groups` | v1 | VocabEstimator.estimate_groups | 已弃用 |
| `POST /api/estimate/article` | v1 | 加权 rank 公式 | 偶尔用 |
| `GET /api/vocabulary/warmup` | v1 | sampler.warmup_sample | 已弃用 |
| `POST /api/vocabulary/quiz-adaptive` | v1 | sampler.adaptive_normal_sample | 已弃用 |
| `GET /api/vocabulary/quiz` | v1 | sampler.balanced_sample | 已弃用 |
| `POST /api/vocabulary/quiz-stage2` | v1 | sampler.stage2_refine_sample | 已弃用 |
| `GET /api/vocabulary/quiz-v2` | v2 | StratifiedQuiz.phase1_sample | **当前主端点** |
| `POST /api/vocabulary/quiz-v2-stage2` | v2 | StratifiedQuiz.phase2_sample | **当前主端点** |
| `POST /api/vocabulary/quiz-v2/estimate` | v2 | StratifiedQuiz.estimate_with_ci | **当前主端点** |
| `GET /api/vocabulary/stats` | common | — | 活跃 |
| `POST /api/tests/save` | common | database.py | 活跃 |
| `GET /api/tests/records` | common | database.py | 活跃 |

- **问题**: 7 个已弃用端点仍暴露在外，增加了维护负担；v1 和 v2 模型混合在同一应用中，`get_stratified_quiz()` 缓存单例 (`server/main.py:113`) 但重建会加载全部 11,950 词

### 3.4 前端层

#### `web/app.js` ⭐⭐⭐⭐ (4/5)
- **位置**: `web/app.js:1-711`
- **UX 亮点**:
  - 答题完成后自动展示汇总页（`showSummary`, `line 246-282`）
  - 支持回退修改答案（goPrev, 键盘左右键）
  - 键盘快捷键 1-6 选择选项
  - 所有 60（实际 25）题汇总列表
  - Stage 2 细化测试无缝追加
- **问题**: "60 道单选题"的文案与实际不符（目前返回 25 题）；binary mode 在无中文选项时的降级策略

#### `web/index.html` + `web/styles.css` ⭐⭐⭐⭐ (4/5)
- 简洁现代 UI，响应式设计，暗色模式未实现但色系预留了变量
- CSS 动画流畅（`fadeInUp`, `pulse`）

### 3.5 训练层

#### `optim/train_bucket_matrix.py` ⭐⭐⭐ (3/5)
- **位置**: `optim/train_bucket_matrix.py:1-1939`
- **6 种数据生成方案**:

| 方案 | 用户数 | 说明 |
|------|--------|------|
| `freq_only` | 12 | 纯词频排序填充 |
| `cet6_simple` | 2000 | CET-6 词表纯随机抽样 |
| `cet6_hybrid` | 551+12 | CET-6 加权 + 词频边界 |
| `twostage` | 1301 | 词频打底 2500 + CET-6 加权 |
| `curriculum` | 1301 | 中考⊂高考⊂四级⊂六级课程体系 |
| `two_phase` | 1000 | 组 A two-phase + 组 B 纯词频 |

- **问题**: 6 套参数文件共存（`trained_params_bucket.json`, `_cet6.json`, `_anchor.json`, `_backup.json`, `_scheme1.json`, `.bak`）；训练数据全合成（无真实用户数据）；Phase 2 校准训练的效果微积分有限

---

## 4. 性能分析

### 4.1 首次加载

| 组件 | 耗时 | 说明 |
|------|------|------|
| `VocabBank` 构建 | ~0.57s | 加载 wordfreq 90k + lemmatizer + 桶分配 |
| `VocabBank` 缓存加载 | ~0.15s | pickle 反序列化 |
| `StratifiedQuiz` 构建 | ~0.80s | 加载 stage_vocab.json (214k 行) + 索引构建 |
| 服务器启动 | ~1.4s | 全部模块 + 两个模型就绪 |

### 4.2 用户请求

| 操作 | 耗时 | 说明 |
|------|------|------|
| v2 测验生成 (25 题) | ~0.02s | phase1_sample 纯 Python |
| v2 估算提交 | ~0.35s | fit_ability (Newton-Raphson) + estimate_vocab |
| v1 估算 (Bootstrap 300) | ~3-5s | bootstrap_interval 300 次逻辑回归 |
| 文章估算 | ~0.01s | 纯 rank 查找 + 加权公式 |

### 4.3 内存占用

| 组件 | 内存 |
|------|------|
| VocabBank | ~30 MB (wordfreq 90k + lemmatizer) |
| StratifiedQuiz | ~15 MB (stage_vocab 11,950 词 + 索引) |
| translations.py | ~8 MB (27,402 行中文释义) |
| **总计 (运行时)** | ~55 MB |

### 4.4 关键瓶颈

- **Bootstrap 300 次**: v1 的 `bootstrap_interval` (`vocab_model.py:272-293`) 每次做 300 次逻辑回归重采样 → 3-5s 响应，远高于 v2 的 ~0.35s
- **Newton-Raphson**: v2 的 `_mle_theta` (`stratified_quiz.py:226-270`) 归功于无 scipy 依赖，但迭代次数是硬编码的，极端数据可能不收敛

---

## 5. 模型效果评估

### 5.1 旧桶模型 vs 新 Rasch 模型

| 维度 | 桶模型 (v1) | Rasch 模型 (v2) |
|------|-------------|-----------------|
| 词库 | wordfreq 90k → 21,738 词族 | stage_vocab 11,950 词 |
| 词等级 | 9 个频段桶，硬边界 | 20 个 difficulty class，连续概率 |
| 模型公式 | P = sigmoid(θ_b + γ_u) | P = sigmoid(θ - logit(d)) |
| 用户参数 | 网格搜索 γ ∈ [-15, 15], step 0.1 | MLE Newton-Raphson |
| 置信区间 | 固定 ±15% | Fisher 信息量 → 95% CI |
| 响应时间 | 3-5s (Bootstrap) | ~0.35s |
| 题数 | 10 热身 + 36 自适应 + Stage2 | 25 (+Stage2 细化) |

**Rasch 模型的优势**：
- 连续概率曲线（无硬阈值），self-consistent
- 信息量最大化的 CAT 抽样
- 更快的推理速度（Bootstrap 300 → Newton-Raphson）
- 统计理论基础扎实（Fisher 信息量 → 标准误差）

**桶模型的优势**：
- 更大的词库（21,738 vs 11,950）
- 有训练参数（虽然合成数据可能不准确）
- 经得起更多迭代训练

### 5.2 两阶段训练 vs 纯词频 vs 纯 CET-6

| 方案 | 桶参数特点 | 评估 |
|------|-----------|------|
| `freq_only` | θ 梯度大（1k: +9.95 → 30k: -8.14） | 过于极化，实际用户不会这么极端 |
| `cet6_hybrid` | θ 梯度小（1k: +1.98 → 30k: -4.23） | 更接近真实学习路径 |
| `two_phase` | 混合方案，freq fill + cet6 | 折中，但 freq fill 和 cet6 的拼合比例是人为的 |

**结论**: `cet6_hybrid` 的 θ 分布更温和合理；`freq_only` 过于激进（sigmoid(9.95)≈1.0 的高频桶意味着所有高频词用户全都会——实际不现实）

### 5.3 难度打分公式的合理性

`difficulty.py` 基于教育阶段计算经验值：
- 小学 3-6 → difficulty ≈ 0.1-0.2
- 初中 7 → 0.3, 8 → 0.4, 9 → 0.5
- 高中 → 0.5-0.6
- CET-4 → 0.6-0.7
- CET-6 → 0.7-0.8
- IELTS → 0.8-0.9

**问题**: 这是纯教育专家的经验估计，缺乏数据验证。同一个教育阶段内的词被视为同等难度，但实际上同一教材内有简单的日常词汇和更抽象的学术词汇。logit 变换后和 Rasch θ 的尺度匹配性也未经验证。

---

## 6. 待优化项

### 🔴 高优先级（影响使用体验）

| # | 问题 | 影响 | 建议方案 |
|---|------|------|----------|
| 1 | **v2 测验数不足** (`stratified_quiz.py:100-140`): phase1_sample 返回 25 题而非 60 题 | UI 说"60 道"实际只有 25，用户困惑；测试信息量不足 | 修复抽题逻辑：每个 c20 类取 3 题（low/mid/high difficulty）= 60 题，或按代码注释实现"First 10 + remaining 50 CAT" |
| 2 | **阶段 2 细化未被使用** | server 的 quiz-v2-stage2 端点存在，但前端 submitEstimate() 不等待 stage2 | 修改 app.js 流程：用户点击"进一步细化"完成后 → 自动合并提交；或在 summary 页面增加"提交含细化的估算"按钮 |
| 3 | **7 个已弃用的 v1 端点** (`server/main.py:142-484`) | 无意义的维护负担，可能被误调用 | 在 v1 端点前加 `@app.middleware` 添加 deprecation warning header 或直接移除 |
| 4 | **词库覆盖不足** (`stage_vocab.json` 仅 11,950 词) | 高阶用户（IELTS 8k+）可测词汇有限 | 扩展 stage_vocab 至 20k+；或融合 wordfreq 的额外词作为 "oos" (out-of-sample) |

### 🟡 中优先级（提升准确度）

| # | 问题 | 影响 | 建议方案 |
|---|------|------|----------|
| 5 | **无真实用户数据验证**：所有训练数据是合成的 | 模型准确度未经实测校准 | 收集真实用户的 CET-4/6/考研/雅思成绩进行交叉验证；用真实答题数据做 A/B test |
| 6 | **难度打分无数据反馈** (`difficulty.py`) | 教育阶段经验公式可能偏离真实难度 | 收集用户答题正确率 → 反向校正 difficulty；实现 IRT 3PL 模型的 discrimination / guessing 参数 |
| 7 | **词库一致性**：stage_vocab 和 VocabBank 是两套不同的词 | 两个模型的词库不一致，比较不公平 | 统一到一套词库：以 stage_vocab 为主，wordfreq 做扩展 |
| 8 | **75k 行数据在代码里**：`translations.py` (27k) + `official_vocab.py` (51k) | 代码臃肿，维护困难 | 将翻译表和词表移入 `data/` 目录，运行时按需加载 |

### 🟢 低优先级（锦上添花）

| # | 问题 | 影响 | 建议方案 |
|---|------|------|----------|
| 9 | **前端说了 60 题但代码只有 25** (`web/index.html:28`) | 用户体验偏差 | 修复文案：改为"约 20-60 道适应性测试"或修复 60 题 |
| 10 | **无暗色模式** (`web/styles.css`) | 视觉效果单一 | 添加 CSS prefers-color-scheme 支持 |
| 11 | **6 套训练参数文件共存** | 选型混乱 | 清理参数文件，只保留最优方案（cet6_hybrid + 课程体系） |
| 12 | **Wordfreq 词库的适用性** | 词频数据基于通用英语语料库而非 EFL 场景 | 考虑使用 COCA 或 BNC 频率列表 |
| 13 | **无单元/集成测试** | 代码修改风险高 | 添加 pytest 测试套件覆盖核心推理路径 |
| 14 | **数据库用 SQLite 无迁移机制** | schema 变更困难 | 添加 Alembic 或手工迁移脚本 |

---

## 7. 下一步建议

### 短期（1-2 周）

1. **修复 v2 60 题** — 调整 `stratified_quiz.py:phase1_sample()` 的抽题逻辑，确保返回 60 题并按计划覆盖 20 个 difficulty class
2. **打通 stage2 前端流程** — 修改 `app.js` 的提交逻辑，让用户在完成细化后自动合并提交
3. **清理已弃用端点** — 在 v1 端点加 deprecation headers 或重定向到 v2
4. **统一文案** — 修复前端"60 道"提示为实际题数

### 中期（1-2 月）

5. **真实用户数据采集** — 在结果页添加可选的 CET-4/6 成绩录入，积累交叉验证数据
6. **难度分值校正** — 用累计答题数据反向更新 `difficulty` 矩阵（在线学习或定期批量重算）
7. **数据迁出代码** — 将 `translations.py` 和 `official_vocab.py` 的词表移至 `data/` JSON 文件
8. **统一词库** — 以 stage_vocab 为主词库，wordfreq 做 out-of-vocabulary 扩展评估

### 长期（3-6 月）

9. **IRT 3PL 模型** — 引入项目反应理论的 3 参数模型（难度 + 区分度 + 猜测参数），针对选择题（非纯 binary）提供更准确的估计
10. **主动学习** — 用贝叶斯信息量最大化策略动态生成下一题（真正的 CAT），而不是预先抽取固定 60 题
11. **学习轨迹可视化** — 跨会话追踪 θ 的变化曲线，展示用户的学习进步
12. **多语言评估** — 扩展到日语/韩语/其他语言的词汇评估框架

---

## 优先级表格汇总

| 问题 | 影响 | 建议方案 | 优先级 |
|------|------|----------|--------|
| v2 60 题只返回 25 | 测试信息量不足 | 修复 `_pick_from_class` 抽题循环 | 🔴 |
| Stage2 前端流程未打通 | 细化功能无法使用 | 修改 app.js 自动合并提交 | 🔴 |
| 7 个废弃端点未清理 | 维护负担大 | 标记废弃 + 逐步移除 | 🔴 |
| stage_vocab 仅 11,950 词 | 高段用户可测词汇少 | 扩展至 20k+ | 🔴 |
| 无真实用户验证数据 | 准确度存疑 | 收集 CET/雅思成绩交叉验证 | 🟡 |
| 难度分值未经数据校正 | 经验公式可能偏差 | 用答题数据反向校正 | 🟡 |
| 75k 行数据在 Python 文件里 | 代码臃肿 | 移入 data/ JSON | 🟡 |
| 词库不一致（两套词） | 模型比较不公平 | 统一到一套词库 | 🟡 |
| 前端说 60 题实际 25 | 文案不符 | 同步文案与实现 | 🟢 |
| 6 套参数文件共存 | 选型混乱 | 清理 + 保留最优方案 | 🟢 |
| 无测试套件 | 重构风险高 | 添加 pytest | 🟡 |
| 无暗色模式 | UI 单一 | CSS prefers-color-scheme | 🟢 |
| SQLite 无迁移机制 | 未来 schema 变更难 | 添加迁移脚本 | 🟢 |

---

## 评分总览

| 模块 | 文件路径 | 评分 |
|------|----------|:----:|
| `stage_vocab.json` | 数据层 | ⭐⭐⭐⭐⭐ |
| `exam_vocab/` 考纲词表 | 数据层 | ⭐⭐⭐⭐ |
| `vocab_bank.py` | 数据层 | ⭐⭐⭐⭐ |
| `difficulty.py` | 数据层 | ⭐⭐⭐ |
| `bucket_model.py` | 模型层 v1 | ⭐⭐⭐ |
| `stratified_quiz.py` | 模型层 v2 | ⭐⭐⭐⭐ |
| `vocab_model.py` | 模型层 v1 | ⭐⭐⭐ |
| `official_vocab.py` | 模型层 | ⭐⭐⭐⭐ |
| `server/main.py` | API 层 | ⭐⭐⭐ |
| `web/app.js` | 前端层 | ⭐⭐⭐⭐ |
| `web/styles.css` | 前端层 | ⭐⭐⭐⭐ |
| `train_bucket_matrix.py` | 训练层 | ⭐⭐⭐ |

**项目总体**: ⭐⭐⭐ (3.5/5) — 设计思路好（从桶模型→Rasch）、v2 架构合理，但在实现完整性（60题bug、stage2流程）和工程规范（数据在代码中、参数文件乱）上还有改进空间。

---

*本文档由 `docs/project_review.md` 自动审查生成，基于 `~/stu/vocab_estimator/` 截至 2026-06-23 的代码快照。*
