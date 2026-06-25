# 词向量增强 Difficulty 标定——实验方案

> 用词向量（word embeddings）改进当前 `0.6×stage + 0.4×rank` 的线性 difficulty 公式。

## 1. 动机

### 现状

当前 difficulty 只用了 2 个标量特征：

```python
difficulty = alpha * norm_stage(first_stage_priority) + beta * norm_rank(wordfreq_rank)
```

仅考虑**教材阶段**和**语料频率**，完全没有用到词的语义、形态或词族信息。

### 丢失的信号

词向量能提供的信息：
- **语义相关词应有相近难度**：run/running/runner 不应差异很大（当前因频率 rank 不同会拉开差距）
- **形态复杂度**：多音节/长词通常更难（当前完全不考虑词长）
- **词族关系**：know/knowledge/acknowledge 的难度应协同变化
- **语义场**：如医学词汇整体偏难，但不在教材覆盖范围内时会被低估

### 预期收益

即使 MAE 已接近理论下界（~360 词），改进 difficulty 标定可以：
- 让词间难度排序更合理（如 `abacus` 应比 `butterfly` 难）
- 提高出题质量（layer 20 内选择更具区分度的词）
- 让覆盖度报告（如文章估算）更准确
- 为真实用户数据校准提供更好的先验

## 2. 方案对比

### 方案 A：Embedding 聚类后难度平滑（推荐优先实验）

**思路**：在 cluster_20/100 内，用词向量相似性做二次平滑

```
对于每个 cluster_20 内的 n 个词：
  1. 获取每个词的 300d GloVe 向量
  2. 计算 pairwise cosine similarity
  3. 每个词的平滑后 difficulty = 原始 diff × (1-α) + α × 邻居平均 diff
  4. 邻居定义为 embedding 空间中距离最近的 k=5 个词
```

**优点**：
- 完全不改变现有模型结构
- 可快速验证（1 天就能跑完）
- 可解释性强（能看到哪些词被拉近了）
- 1GB 服务器可运行

**缺点**：
- 只做平滑，不提升预测力
- 需要调 α（平滑强度）和 k（邻居数）

**验证**：跑 simulation_eval_v2 对比平滑前后的 MAE/出题区分度

---

### 方案 B：MLP 残差校正（轻量语义特征）

**思路**：用词向量 + 形态特征训练一个小 MLP，输出 difficulty 偏移量（残差）

```
输入特征（每词 310d）：
  - 300d GloVe embedding
  - 1d 词长（log normalized to [0,1]）
  - 1d 音节数估算（元音组数 / max_syllables）
  - 1d 当前 difficulty（保证 baseline 始终存在）
  - 1d 词频 rank（log normalized）
  
模型：MLP 1-2 层, 64→16→1, ReLU, Dropout=0.2
目标：residual = ground_truth_diff - baseline_diff
      （用现有 11418 词的 difficulty 作为伪标签）
      
最终 difficulty = baseline_diff + MLP_residual
```

**优点**：
- 保留 baseline 可解释性，只修正系统性偏差
- 轻量（~5K 参数，CPU 推理 <1ms/11418 词）
- 直观：能看到哪些特征增加了难度（如长词+0.1）

**缺点**：
- 伪标签来自现有 difficulty，噪声会传递
- 如果没有真实用户数据，提升有限

**数据要求**：仅需 11418 词 + 预训练词向量，无需额外标注

**验证**：
1. 训练/验证集划分（80%/20%），比较 test MAE
2. 用修正后的 difficulty 跑 simulation_eval_v2
3. 检查：同词族的 difficulty 是否变得更合理

---

### 方案 C：端到端 Rasch + 语义难度（远期探索）

**思路**：用 MLP(word_embedding) 直接输出 logit_difficulty，替代当前 hand-crafted 公式

```
P(known | θ, word) = sigmoid(θ - MLP(glove_embedding + rank + length))
```

**优点**：
- 语义特征自然融入 Rasch 框架
- 可端到端训练（需要答题数据）
- 理论上限最高

**缺点**：
- ❌ 需要大量真实答题数据（至少 200+ 用户的完整答题记录）
- ❌ 需要 GPU 训练
- ❌ 复杂度高，难以在一周内完成
- ❌ 可解释性差

**建议推迟**，直到收集到足够的真实用户数据。

## 3. 数据需求

| 方案 | 所需数据 | 额外标注 | 数据准备 |
|:---:|:---------|:--------|:---------|
| A | 无 | 无 | 下载 GloVe 300d（5 分钟）|
| B | 现有 11418 词 + difficulty | 无 | 同上 + 算词长/音节数 |
| C | 200+ 用户答题数据 | 需要 | 需上线收集数据 |

### 预训练词向量选择

| 模型 | 维度 | 大小 | 推荐理由 |
|:----|:---:|:----|:---------|
| **GloVe 300d** (glove.840B.300d) | 300 | 2GB | 覆盖广、轻量、稳定 |
| fastText (crawl-300d-2M) | 300 | 4GB | 支持 OOV（子词）|
| word2vec (GoogleNews) | 300 | 3.5GB | 语义质量好 |
| BERT (all-mpnet-base-v2) | 768 | 1GB | 上下文敏感，但太重 |

**推荐**：先用 GloVe 300d（最轻量、覆盖最广、OOV 率低）。如果 OOV 词多（>5%），换 fastText。

## 4. 验证方案

### 4.1 直接指标
- **Difficulty 排序一致性**：修正后与人工判断的 Spearman 相关系数
- **词族内方差**：run/running/runner 三词的 difficulty 标准差应减小
- **极端值检查**：明显简单/难的词是否被合理标定

### 4.2 模拟评估
用 `tests/simulation_eval.py` V2 管道，对比新旧 difficulty：

```bash
# 需要模拟器支持外部 difficulty（当前需修改 simulation_eval.py）
# 1. 读取修正后的 stage_vocab.json
# 2. 初始化 StratifiedQuiz 时传入新数据
# 3. 跑 2000 用户，对比 MAE / R² / bucket_errors
```

### 4.3 出题质量
- cluster_20 内 difficulty 分布更均匀 → 出题区分度更高
- cluster_100 内 coherence 更好 → 同类词难度更一致

## 5. 实施路线图（一共约 5 天）

### Phase 1：数据准备（第 1 天）
- [ ] 下载 GloVe 300d 词向量
- [ ] 提取 11418 词的 embedding（保存为 `.npy`）
- [ ] 统计 OOV 率，判断是否需要换 fastText
- [ ] 计算每词的词长、音节数（元音组规则）
- [ ] 脚本在 `scripts/extract_embeddings.py`

### Phase 2：方案 A 实现 + 验证（第 2 天）
- [ ] 实现 cluster_20 内 embedding 平滑
- [ ] 参数搜索：α∈[0.1, 0.5], k∈[3, 10]
- [ ] 跑 200 用户 simulation_eval 快速验证
- [ ] 输出最佳参数和 MAE 变化

### Phase 3：方案 B 实现 + 验证（第 3-4 天）
- [ ] 实现 MLP 残差校正模型
- [ ] 训练/验证/测试划分（80/10/10）
- [ ] 训练并保存模型（`data/difficulty_corrector.pkl`）
- [ ] 跑 2000 用户完整评估
- [ ] 对比 A/B 两方案结果

### Phase 4：消融 + 报告（第 5 天）
- [ ] 分析哪些特征贡献最大（词长？embedding？）
- [ ] 错误分析：哪些词修正最多/最少
- [ ] 写实验结论到 `docs/difficulty_enhancement_results.md`
- [ ] 如有效果提升，更新 stage_vocab.json 并重新部署

## 6. 可行性评估

### 环境
- **1GB/2核 服务器**：方案 A/B 完全可跑（仅 CPU，无需 GPU）
- **本地开发机**（V100 Linux）：方案 B 训练也很快
- **训练时间**：方案 A < 1 秒，方案 B < 1 分钟

### 与现有管线配合
- 不影响 `stratified_quiz.py`（只改 `difficulty.py` 的计算逻辑或直接更新 `stage_vocab.json`）
- 不影响 `calibration_trainer.py`（校准是后处理，独立于 difficulty）
- 不影响 `article_estimator.py`（只读 difficulty 值）

### 风险
- GloVe 可能 OOV 率 >10%（一些罕见学术词或缩写）
  - 缓解：换 fastText（子词级别 OOV 率低）
  - 缓解：用 word2vec + 随机初始化未登录词向量
- **预期效果有限**：模拟 MAE 已接近理论下界，difficulty 改进可能不体现在 MAE 上
  - 但出题质量和词排序会改善

## 7. 推荐

**先做方案 A（1 天）→ 如果效果好再做方案 B（2 天）。**
