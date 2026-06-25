# Difficulty Corrector 训练结果评估

## 结论摘要

本次 `scripts/train_difficulty_corrector.py` 训练结果**不应采纳到线上词库**。

表面指标如 Test MAE = 0.008、cluster_20 内 std 从 0.0117 降到 0.0082 看起来很好，但它们没有证明模型学到了有效难度信号。相反，796 个异常词、简单高频词被推到 0.99 difficulty、Top corrections 大量为 +0.49，说明模型输出已经破坏了 difficulty 排序。

最关键的问题不是 V100 训练是否成功，也不只是形态学特征太弱，而是当前训练目标和应用方式存在错配：

- 训练时模型学习的是**绝对 difficulty**：`y = current_difficulty`
- 应用时却把模型输出当成**残差 residual** 加回 baseline：

```python
corrected = baseline_difficulty + model_output
```

如果模型成功复现当前 difficulty，最终结果会近似变成 `2 * baseline_difficulty`，再被 clip 到 0.999。这正好解释了 `someone`、`chinese`、`store` 等原本约 0.51 的词被整体推到 0.99+ 的现象。

因此，本次实验结果不能被解释为“形态学模型发现了大量词应该变难”，更接近于一个目标函数/后处理错误导致的系统性上推。

## 当前实验的主要质量问题

### 1. 指标存在标签泄漏或自我复现

当前方案用现有 `stage_vocab.json` 的 difficulty 作为伪标签。这个标签本身来自 baseline 规则，而不是独立人工标注或用户答题数据。

在这种设置下，Test MAE = 0.008 并不表示模型接近真实英语难度，只表示它在随机划分的测试集上复现了现有 difficulty。尤其当输入特征中包含 baseline difficulty 时，模型可以直接学习 identity mapping。

这类指标不能回答真正关心的问题：

- `a`、`about` 是否比 `abandon` 简单？
- 同一个 cluster 内抽题是否更有区分度？
- 修正后的 difficulty 是否能降低真实用户测评误差？

### 2. 残差校正实现和训练目标不一致

如果要做 residual corrector，训练目标应是：

```text
target_residual = target_difficulty - baseline_difficulty
corrected = baseline_difficulty + predicted_residual
```

但当前结果表现更像：

```text
model_output ≈ baseline_difficulty
corrected = baseline_difficulty + model_output
```

这会导致中高难度词大面积触顶，cluster 内方差被压平，相关系数下降。`cluster_20` 内 std 降低并不是有效平滑，而是 clip 到上限造成的假象。

### 3. 形态学特征预测力确实很弱

我用当前 `data/stage_vocab.json` 粗略验证了纯形态学特征的上限。用 `log(word_len)`、`log(syllables)`、`vowel_ratio`、`long_word` 做线性回归预测当前 difficulty：

| 指标 | 数值 |
|---|---:|
| MAE | 0.1217 |
| RMSE | 0.1566 |
| R² | 0.1339 |
| `log(word_len)` 与 difficulty 相关 | 0.3581 |
| `log(syllables)` 与 difficulty 相关 | 0.3383 |
| `vowel_ratio` 与 difficulty 相关 | 0.0508 |
| `long_word >= 8` 与 difficulty 相关 | 0.2775 |

这说明形态特征只能解释很小一部分难度差异。它们可以作为辅助特征，但不能作为主信号。

典型反例：

| word | 当前 difficulty | 长度 | 音节估计 |
|---|---:|---:|---:|
| someone | 0.5082 | 7 | 3 |
| abandon | 0.7377 | 7 | 3 |
| chinese | 0.5098 | 7 | 3 |
| store | 0.5092 | 5 | 2 |
| about | 0.3868 | 5 | 2 |

`someone` 和 `abandon` 形态几乎一样，但学习难度明显不同。差异来自频率、教材阶段、语义抽象度、词义熟悉度和使用场景，不是词长或音节数能稳定捕捉的。

### 4. “压平”不是有效校准

cluster 内 std 下降需要结合排序和异常词检查一起看。理想的校正应该让相似难度词更稳定，同时保留简单词和难词之间的分离。

本次结果中：

- baseline 相关系数只有 0.67，说明排序被大幅打乱
- 大量简单词被推到 0.99，违反常识
- Top corrections 集中在 +0.49，像是上限 clip，而不是语义校正
- residual std 仅 0.0102，说明模型本身几乎没有学到有效可解释修正

所以 `cluster_20` std 从 0.0117 到 0.0082 只能说明数值更接近，不能说明出题质量更好。

## 为什么形态学特征不够

形态学特征的有效信息主要是：

- 长词平均更难
- 多音节词平均更难
- 派生词、学术词常常更长

但英语词汇难度的主要来源不是形态，而是：

- 词频：`about` 和 `abandon` 的日常暴露频率完全不同
- 习得阶段：教材早期词和考试词表词的学习路径不同
- 语义抽象度：`justice`、`assume`、`criterion` 这类词不是靠长度变难
- 领域性：医学、法律、学术、技术词汇常因语义场变难
- 多义性和搭配：常见词的学习难度还受短语和语境影响
- 词族关系：`know`、`knowledge`、`acknowledge` 应该有相关但不相同的难度

因此，形态特征适合当作弱特征或正则项，不适合单独训练 difficulty corrector。

## 下载 GloVe 的替代方案评估

### 方案 A：HuggingFace sentence embeddings

推荐优先级：**高**

`all-MiniLM-L6-v2` 约 80MB，维度通常为 384，支持子词分词，对 OOV 友好。相比 GloVe，它更容易下载，语义质量也足够用于 11418 个词的相似性、聚类和平滑。

仓库当前已经存在：

- `data/word_embeddings_384d.npy`
- `data/word_embeddings_index.json`

并且覆盖 11418 个词。这意味着下一步很可能不需要继续卡在 GloVe 下载上，而应优先确认这份 384d embedding 的来源、质量和生成脚本，然后直接用它做受控实验。

注意：句子模型对单词的 embedding 未必总是优于专门的 word embedding，但它的覆盖率和工程可用性更好，适合当前阶段。

### 方案 B：fastText Python 包

推荐优先级：**低**

fastText 的子词能力很好，理论上适合词汇难度任务。但英文模型体积太大，压缩模型也可能达到数 GB。当前网络环境已经无法稳定下载 2GB GloVe，继续尝试 6GB fastText 不现实。

除非后续有稳定服务器缓存或已有本地模型，否则不建议作为近期路径。

### 方案 C：gensim 下载 `glove-wiki-gigaword-100`

推荐优先级：**中**

100d GloVe 约 300MB，下载难度小于 2GB GloVe，适合作为轻量 fallback。缺点是：

- 词表覆盖和语义质量弱于大 GloVe
- 仍然依赖外网
- 不支持子词，OOV 词需要特殊处理

如果现有 384d embedding 无法确认质量，可以试这个方案作为对照组，而不是作为主线。

### 方案 D：从 PolyMarket 云服务器下载后传回

推荐优先级：**中低**

如果必须拿到 GloVe 6B 或 840B，这是可行的工程绕路。但它解决的是下载问题，不解决实验设计问题。

在当前阶段，先修正训练目标和验证方案更重要。否则即使下载了 GloVe，仍可能得到同样不可用的“复现 baseline + 错误残差叠加”结果。

### 方案 E：用项目现有数据生成词向量

推荐优先级：**低**

仅用 11418 个孤立词从头训练 word2vec 或 sentencepiece embedding，几乎没有上下文共现信号。one-hot + SVD 也只能编码词表索引或已有元数据，不会自然产生语义结构。

如果要用项目现有数据，比较合理的方式不是“从词本身训练词向量”，而是构造结构化特征：

- stage one-hot / stage priority
- wordfreq rank
- 来源数量、来源类型、source confidence
- 是否出现在 CET4/CET6/TOEFL/GRE/IELTS
- cluster id 或历史抽题表现

这些可以作为 tabular features，但不能替代预训练语义 embedding。

## 实验设计修正建议

### 1. 先修正任务定义

当前没有独立真实标签，因此不要把“拟合当前 difficulty 的 MAE”当作核心成功指标。

短期可以做两类实验：

#### 实验 A：受限平滑，而不是自由校正

目标不是重新预测 difficulty，而是在局部范围内做小幅平滑：

```text
corrected = baseline * (1 - alpha) + neighbor_mean_difficulty * alpha
```

约束：

- 只在相同或相邻 `cluster_20` 内找邻居
- `alpha <= 0.15`
- 单词最大修正幅度 `abs(delta) <= 0.05`
- 不允许简单高频词被推到 0.9+
- 输出前做人工 sanity check

这比 MLP residual 更稳，因为它不会自由生成大幅 correction。

#### 实验 B：真正的 residual model

如果继续 MLP，必须明确：

```text
baseline = current difficulty
target = pseudo_target - baseline
model predicts residual
corrected = baseline + residual
```

但在没有真实 pseudo_target 的情况下，`target` 不能继续等于当前 difficulty。可以考虑构造一个弱目标，例如：

- exam/stage 规则重新计算出的 alternative difficulty
- embedding 邻居平滑后的 difficulty
- 人工审核的小规模 gold set
- 用户答题数据拟合出的 item difficulty

否则 residual target 全是 0，模型没有可学内容。

### 2. 加入硬性质量门禁

每次生成 `stage_vocab_enhanced.json` 前，至少检查：

- 与 baseline 的 Spearman 相关应高于 0.95，除非有真实标签证明可以重排
- `abs(delta) > 0.10` 的词必须输出人工审核列表
- `a`、`the`、`about`、`someone`、`store` 等简单词不得高于合理阈值
- `cluster_20` std 下降不能作为单独成功指标
- 修正后 difficulty 的分布不能大面积堆在 0.999
- Top positive/negative corrections 必须语义合理

### 3. 引入真实用户数据作为最终目标

根本方案是从答题记录中估计 item difficulty。可用 Rasch/IRT 形式：

```text
P(known | user_theta, word_difficulty) = sigmoid(user_theta - word_difficulty)
```

当某个词在大量不同水平用户上都有回答记录后，可以估计它的真实 `word_difficulty`。embedding、stage、rank、形态特征都应作为先验或正则，而不是最终标签。

在没有真实答题数据前，embedding 只能帮助做排序先验和平滑，不能证明“真实难度更准”。

## 推荐下一步

建议选择：**D) 其他：先修正实验设计，直接利用现有 384d embedding 做受控平滑实验；暂时不要继续把 GloVe 下载作为阻塞项。**

具体顺序：

1. **废弃本次生成的 `data/difficulty_corrector.pt` 和 `data/stage_vocab_enhanced.json`**
   - 本次结果存在系统性上推和 clip，不适合进入产品。

2. **修正 `train_difficulty_corrector.py` 的语义**
   - 如果模型输出绝对 difficulty，应用时就不要加 baseline。
   - 如果模型输出 residual，训练目标必须是 residual。
   - 同步修正文档和 `meta["difficulty_enhancement"]["features"]`，当前脚本实际特征与描述不一致。

3. **先做 embedding 邻居平滑 baseline**
   - 使用现有 `data/word_embeddings_384d.npy`。
   - 在 cluster 内做 kNN 平滑。
   - 设置 `alpha <= 0.15`、`max_delta <= 0.05`。
   - 输出 correction report，而不是直接覆盖主词库。

4. **建立小型人工 gold set**
   - 选 200-500 个词，覆盖简单词、考试词、抽象词、领域词、异常词。
   - 人工给出相对排序或粗分桶。
   - 用 Spearman、bucket accuracy、异常词召回评估，而不是只看 pseudo-label MAE。

5. **并行准备真实答题数据闭环**
   - 后续用用户答题记录拟合 item difficulty。
   - embedding 模型只作为冷启动先验和低样本平滑。

GloVe 下载可以作为对照实验继续尝试，但不应作为当前主线。当前项目已经有 384d embedding，先把目标函数、应用逻辑和质量门禁修好，收益会比换一个更大的 embedding 文件更直接。

