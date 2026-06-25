# 文章词汇量估算优化方案

## 1. 问题诊断

当前 `/api/v2/estimate/article` 的关键入口是
`vocab_estimator/article_estimator.py::estimate_article()`。实际流程是：

1. 使用正则 `[a-z]+(?:-[a-z]+)*` 分词，连字符词会被保留为一个 token。
2. 转小写、去内置停用词。
3. 使用 `article_estimator.py` 内联规则 `lemmatize_word()` 做轻量 lemma 候选，并优先返回能命中 `stage_vocab` 的候选。
4. 只对命中的 token 取 `difficulty`。
5. 用命中 token 的 difficulty 中位数，在 `stage_vocab` 全量 difficulty 分布中找累计位置，得到 `estimated_vocab`。
6. 返回 token 覆盖率、unique 覆盖率和未匹配词样例。

这解释了四篇测试语料整体偏低的核心原因：未命中词不会参与难度统计；而高难文本中未命中的往往正是最有区分度的学术词、技术词和新词。C.txt 虽然排序高于 F/P/K，但大量高难词被排除后，中位数只反映了剩余可命中词的难度。

### 低覆盖率原因

`data/stage_vocab.json` 当前有 11,418 个词条。它对国内考试词表覆盖很高，但对开放域高频、学术和 GRE/TOEFL 扩展词覆盖不足：

| 词表 | unique | 已在 stage_vocab | 缺失 | 覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| cet6.txt | 8,013 | 8,009 | 4 | 99.95% |
| gaokao.txt | 3,449 | 3,447 | 2 | 99.94% |
| toefl.txt | 3,470 | 2,428 | 1,042 | 70.0% |
| gre.txt | 6,677 | 2,598 | 4,079 | 38.9% |
| coca20000.txt | 17,634 | 9,785 | 7,849 | 55.5% |

对 C.txt 提到的未匹配词，词表状态如下：

| 词 | stage_vocab | exam_vocab 来源 | 判断 |
| --- | --- | --- | --- |
| algorithmic | 缺失 | 无 | 应加入；可由 `algorithm` 派生，AI/技术文本高频 |
| societal | 缺失 | coca20000 | 应加入；`society` 的常见学术形容词 |
| geopolitical | 缺失 | coca20000 | 应加入；新闻、国际关系和气候文本常见 |
| blockchain | 缺失 | 无 | 应加入；现代技术领域常见词 |
| hegemony | 缺失 | coca20000, gre | 应加入；GRE/学术人文词 |
| epistemological | 缺失 | coca20000 | 应加入；学术哲学/文学批评词 |
| epistemology | 缺失 | coca20000, gre | 应加入；可作为 `epistemological` 的词族核心 |
| governance | 缺失 | coca20000, gre | 应加入；政策/商业/气候文本常见 |
| mitigation | 缺失 | coca20000 | 应加入；气候变化核心词 |
| biodiversity | 缺失 | coca20000 | 应加入；环境文本核心词 |
| neoliberal | 缺失 | coca20000 | 可加入；人文社科常见但更偏专题 |

缺失词可以分为四类：

- **学术词和抽象名词**：`hegemony`, `epistemology`, `epistemological`, `governance`, `mitigation`, `biodiversity`, `neoliberal`。这些词往往是高难文章的难度信号，缺失会显著拉低 C.txt。
- **现代技术词和新词**：`algorithmic`, `blockchain`, `AI-driven`。这类词未必出现在传统考试词表，但在 AI、商业、科技文章中高频。
- **复合词/连字符词**：`AI-driven`, `sea-level`, `self-improvement`, `cross-cultural`, `twentieth-century` 等。当前正则会把连字符整体保留，若词库没有完整词条就不命中，也不会拆成基础词参与估计。
- **派生形态**：`societal`、`geopolitical`、`epistemological`、`algorithmically` 这类 `-al/-ic/-ical/-ly` 派生不能稳定还原到词库已有词。COCA 缺失词中有大量透明派生或屈折形式，例如 `significantly`, `politically`, `typically`, `increased`, `growing`。

另一个次要问题是专有名词和缩写。当前 `article_estimator.py` 在分词前统一 lower-case，无法保留大小写信息，因此 `AI`, `Hollywood`, 国家/族群名、人名、机构名会和普通词混在一起。专有名词是否应该计入词汇量需要单独规则，否则体育、好莱坞、商业文本中的实体词可能带来噪声。

### 当前 lemma 规则是否有效

需要区分两个模块：

- `vocab_estimator/article_estimator.py` 实际使用内联 `lemmatize_word()`，返回 `method.lemmatizer = inline_rule_based`。
- `vocab_estimator/lemmatizer.py` 提供 spaCy 优先、规则回退的 `Lemmatizer`，但当前文章估算没有调用它。

内联规则对屈折变化有效，尤其是词库命中优先策略能处理：

- `algorithms -> algorithm`
- `studies -> study`
- `increased -> increase`
- `growing -> grow`
- `significantly -> significant`
- `blockchains -> blockchain`，前提是未来加入 `blockchain`

但它对派生词和复合词不足：

- `algorithmic -> algorithmic`，不能还原到 `algorithm`
- `societal -> societal`，不能还原到 `society`
- `geopolitical -> geopolitical`，不能拆解或还原到 `political`
- `epistemological -> epistemological`，不能还原到 `epistemology`
- `sea-level -> sea-level`，不会拆成 `sea` 和 `level`
- `decarbonization -> decarbonizat`，`institutionalization -> institutionalizat`，`-tion/-ization` 规则会生成不自然词干，只有在候选恰好命中词库时才有价值

独立 `lemmatizer.py` 的 spaCy 版本更适合标准屈折 lemma，但也不会自动把 `algorithmic` 还原成 `algorithm`，不会把 `epistemological` 还原成 `epistemology`。因此本问题不能只靠 lemma 解决，词库扩展仍是优先项。

### difficulty 中位数法的局限性

当前估算公式：

```text
estimated_vocab = count(stage_vocab.difficulty <= article_median_difficulty)
```

它有三个局限：

1. **中位数忽略高难长尾**：只要高难词比例不到 50%，它们即使大量出现，也不会明显推高中位数。C.txt 的学术词、专有术语被过滤后，高难尾部影响更小。
2. **未匹配词没有难度贡献**：低覆盖率时，估计值不是“带不确定性的偏保守估计”，而是直接按剩余命中词计算，系统性偏低。
3. **stage_vocab difficulty 分布高度非线性**：11,418 个词的 difficulty 中位数约为 0.8572；`difficulty <= 0.7000` 只对应约 2,104 词，`<= 0.7378` 对应约 2,984 词。中位数小幅变化会导致估算区间变化，但高难文章如果中位数停在 0.70-0.74，就很难进入 3,000-4,000 档。

现有 `docs/article_estimation.md` 还写着“不做 lemma 还原”，这已经和代码不一致；后续实施时应同步更新该文档。

## 2. 优化方案

### 方案 A：词库扩展，优先级最高

目标：先提高高难文本的有效覆盖率，让文章中的关键难词进入 difficulty 分布。

建议分三层扩展：

1. **直接补全考试词表缺口**
   - 优先补 `toefl.txt` 和 `gre.txt` 中缺失词，尤其是同时出现在 `coca20000.txt` 的词。
   - CET6/高考已接近全覆盖，不是主要增量来源。
   - GRE 全量 4,079 个缺失词不应一次性无筛选导入，可按 COCA 是否出现、词长、是否专有名词、是否古僻词分层。

2. **导入 COCA 高频缺失词**
   - 从 `coca20000.txt` 中选择缺失的前 5,000-10,000 排名词，过滤噪声：缩写、口语碎片、明显人名地名、拼写异常、短词。
   - 重点保留文章估算有用的开放域内容词：新闻、科技、环境、商业、社科、人文、学术写作词。
   - 对 `societal`, `geopolitical`, `epistemological`, `governance`, `mitigation`, `biodiversity`, `neoliberal` 这类词应直接加入。

3. **补常见派生形态和词族成员**
   - 对 stage_vocab 已有核心词生成候选派生：`-tion`, `-sion`, `-ation`, `-ity`, `-ness`, `-ment`, `-al`, `-ial`, `-ic`, `-ical`, `-ive`, `-ous`, `-ize/-ise`, `-ism`, `-ist`, `-ly`。
   - 只加入能被外部词表验证的派生词，避免机械生成伪词。
   - 例：`algorithm -> algorithmic`, `society -> societal`, `epistemology -> epistemological`, `political -> geopolitical`，其中 `geopolitical` 更适合通过 COCA 直接加入，不建议只靠规则生成。

每个缺失词是否加入词库，建议使用以下判定：

| 条件 | 加入建议 |
| --- | --- |
| 出现在 COCA 且非明显专有名词/噪声 | 加入 |
| 出现在 GRE/TOEFL 且有现代文章使用价值 | 加入 |
| 是已有词的透明派生，且出现在 COCA/GRE/TOEFL 任一词表 | 加入 |
| 是 AI、气候、商业等现代高频领域词，但不在旧考试词表 | 建立人工白名单加入 |
| 仅为人名、地名、球队名、影视角色名 | 默认不加入，或标为 proper noun 并低权处理 |
| 古僻 GRE 词、罕见拼写、口语碎片 | 暂缓 |

difficulty 赋值建议：

- 有考试来源的词，按来源阶段初始化：TOEFL/GRE 通常应高于 CET6，GRE 生僻词可落在 `cluster_20` 高档。
- 有 COCA rank 的词，用 rank 映射到 difficulty，再和同词族核心词做平滑；例如 `societal` 应高于 `society`，但不应极端高于同类学术形容词。
- 现代白名单词用同类词锚定：`algorithmic` 参考 `algorithm`，`blockchain` 参考 `cryptocurrency/software/database/network` 一类技术词，`mitigation` 参考 `adaptation/sustainability/climate`。
- 保留 `sources` 和 `source_confidence`，新增词建议标注 `exam_vocab`, `coca20000`, `derived`, `manual_domain` 等来源。

预期效果：

- C.txt 的 coverage 应从 86.4% 提升到 92%-95%。
- C.txt 的高难词会进入 p75/p90/max 分布，即使中位数不大幅变化，也能为后续方案 B/D 提供有效信号。
- F.txt 中商业、娱乐、体育常见词也会受益，但提升幅度应低于 C.txt，从而保持 C 明显高于 F。

### 方案 B：估算公式改进，次优先级

词库扩展后，再调整估算公式。否则公式会在缺词数据上过拟合。

当前只使用 p50。建议改为保留 p50 的稳定性，同时引入高分位：

```text
base = vocab_count_at(p50)
tail = vocab_count_at(p90) - vocab_count_at(p50)
extreme = vocab_count_at(max) - vocab_count_at(p95)
estimated = base + 0.7 * tail + 0.15 * extreme
```

或等价地直接在 difficulty 上组合：

```text
effective_difficulty = p50 + 0.7 * (p90 - p50) + 0.15 * (max - p95)
estimated = vocab_count_at(effective_difficulty)
```

优点：

- 对学术文本中的少量高难术语更敏感。
- C.txt 这类“基础连接词 + 高难概念词”的结构不会被中位数过度压低。
- 对 K.txt 影响较小，因为它的 p90/max 不应显著高。

风险：

- 专有名词或噪声词如果被误加入词库，会推高 p90/max。
- 短文本的 p90/max 方差大，需要最小 token 数或 winsorize。

备选公式：

```text
estimated = vocab_count_at(weighted_mean_difficulty)
weighted_mean = 0.55 * p50 + 0.30 * mean + 0.15 * p75
```

或者引入置信度：

```text
effective = coverage * observed_estimate + (1 - coverage) * prior_adjusted_estimate
```

建议先在线下比较 p50、p75、p90、mean、加权公式对 C/F/P/K 的排序和目标区间影响，再决定是否改 API 口径。

### 方案 C：lemma 归一化增强

目标：降低透明形态变化造成的未命中，但不要把不同词义粗暴合并。

建议改进点：

1. **复合词拆分**
   - 对未命中的连字符词，尝试拆成多个词：`sea-level -> sea + level`，`AI-driven -> ai + driven`，`cross-cultural -> cross + cultural`。
   - 如果整体词命中，优先整体；整体不命中时才拆分。
   - 拆分后可按平均 difficulty 或最高 difficulty 计入，建议使用 `max(component_difficulty)` 或 `mean + tail bonus`，避免 `sea-level` 被过度低估。

2. **派生后缀候选**
   - 增加只在词库命中时生效的候选规则：`-ness -> adjective/root`，`-ity -> adjective/root`，`-al/-ial -> noun`，`-ic/-ical -> noun`，`-ize/-ise -> noun/adjective`。
   - 例：`algorithmically -> algorithmic -> algorithm`，`epistemological -> epistemology`，`politically -> political`。
   - 这类规则必须基于候选命中或词族映射表，不建议无验证截断。

3. **统一 lemmatizer 入口**
   - 目前 `vocab_estimator/lemmatizer.py` 没有被文章估算使用，`docs/article_estimation.md` 也与实际代码不一致。
   - 建议后续把内联规则抽到共享模块，或让文章估算明确依赖统一的 `Lemmatizer`，避免两套规则分叉。

4. **专有名词和缩写策略**
   - 分词前保留原始 token 大小写信息。
   - 全大写缩写、标题大小写专有名词默认不进入 difficulty，或进入单独 `proper_noun_count`。
   - 对已成为普通词的技术缩写可白名单，例如 `AI`, `DNA`, `CEO`。

方案 C 的优先级低于 A，因为 `algorithmic/societal/hegemony/blockchain` 即使有更好 lemma，也仍需要合理 difficulty 才能表达文章难度。

### 方案 D：coverage 加权校正

目标：当覆盖率低时，显式承认未匹配词带来的偏低风险。

建议两种实现方向：

1. **coverage bonus**

```text
missing_rate = 1 - coverage_unique
bonus = alpha * missing_rate * total_vocab
estimated = base_estimate + bonus
```

这个方案简单，但风险大。未匹配词可能是专有名词、拼写噪声或低频实体，不一定代表读者词汇量要求更高。建议只作为短期校正，并设置上限，例如最多增加 10%-25%。

2. **未匹配词插值估计**

对未匹配词先分类，再按类别补 difficulty：

| 未匹配类别 | 估计方式 |
| --- | --- |
| 出现在 COCA/GRE/TOEFL | 按来源和 rank 映射 difficulty |
| 透明派生词 | 使用词族核心词 difficulty + 派生增量 |
| 连字符复合词 | 使用组件 difficulty 聚合 |
| 专有名词/缩写 | 默认不加或低权 |
| 无来源未知词 | 使用文章已命中词的 p75/p90 插值，低权计入 |

更推荐第二种，因为它能区分 C.txt 的学术未命中词和 F.txt/K.txt 中可能出现的实体词或噪声。

## 3. 实验验证建议

建议建立一个固定评估集，至少包含：

- 当前四篇长文：C.txt、F.txt、P.txt、K.txt。
- 每类再扩充 5-10 篇相似主题文章，避免对四篇样本过拟合。
- 人工标注目标区间：K 约 800-1000，P 约 2000，F 约 3000-4000，C 应明显高于 F。

核心指标：

| 指标 | 用途 |
| --- | --- |
| `estimated_vocab` | 是否进入目标区间 |
| 排序准确率 | 是否满足 C > F > P > K |
| token coverage / unique coverage | 词库覆盖是否改善 |
| unmatched_unique_words top-N | 检查新增词和 lemma 是否解决真实缺口 |
| p50/p75/p90/max difficulty | 判断公式是否捕捉高难尾部 |
| 目标区间误差 | 用区间外距离衡量偏低/偏高 |
| 稳定性 | 同类文章估算方差不应异常扩大 |

现有 `scripts/evaluate_difficulty.py` 可用于比较 `stage_vocab.json` 和未来 `stage_vocab_enhanced.json`：

```bash
python3 scripts/evaluate_difficulty.py \
  --baseline data/stage_vocab.json \
  --enhanced data/stage_vocab_enhanced.json
```

它能检查 difficulty 分布、相关系数、cluster 内一致性和异常词。这个脚本主要评估词库质量，不直接评估文章估算效果。建议新增一个文章级评估脚本，例如：

```bash
python3 scripts/evaluate_article_estimation.py \
  --vocab data/stage_vocab_enhanced.json \
  --cases data/article_eval_cases.json
```

`article_eval_cases.json` 建议字段：

```json
[
  {
    "id": "C",
    "path": "path/to/C.txt",
    "expected_min": 4000,
    "expected_max": null,
    "rank": 4,
    "tags": ["academic", "ai", "climate", "literature"]
  }
]
```

脚本输出：

- 每篇文章的 estimate、level、coverage、unique coverage。
- p50/p75/p90/max、命中/未命中 top 词。
- 是否满足目标区间。
- 排序 Kendall tau 或 Spearman rho。
- 相比 baseline 的 delta。

上线前建议做三轮实验：

1. **A-only**：只扩词库，不改公式，观察 coverage 和估算是否自然上升。
2. **A+B**：在扩词库后测试 p50、高分位加权、mean 加权。
3. **A+B+C/D ablation**：分别打开复合词拆分、派生规则、coverage 插值，确认每项收益和副作用。

## 4. 实施路线图

### 第 1 阶段：词库简单扩展，1-2 天

优先做方案 A 的低风险部分。

- 从 `coca20000.txt`、`toefl.txt`、`gre.txt` 生成缺失候选。
- 过滤明显噪声、短词、短语、专有名词。
- 人工审核 C/F/P/K 未匹配 top 词，先加入高价值词 300-800 个。
- 为新增词生成 difficulty、`cluster_20`、`cluster_100`、来源和置信度。
- 生成 `data/stage_vocab_enhanced.json`，不要直接覆盖原文件。
- 用 `scripts/evaluate_difficulty.py` 检查分布和异常。
- 用四篇文章跑 baseline/enhanced 对比。

成功标准：

- C.txt unique coverage 提升到 92% 以上。
- C.txt 估算明显上升，且仍保持 C > F > P > K。
- K.txt 不应被大量推高，最好仍落在 800-1000 附近或只小幅上升。

### 第 2 阶段：文章级评估脚本，0.5-1 天

- 新增固定 case 配置和文章评估脚本。
- 输出 baseline/enhanced 对比表。
- 把 `unmatched_unique_words` 扩展为完整列表或单独调试输出，避免只看前 50 个。

成功标准：

- 每次改词库或公式都能复现实验结果。
- 能快速看到新增词是否命中真实文章。

### 第 3 阶段：公式实验，1 天

- 在不改 API 的情况下离线实现多种估算公式：p50、p75、p90、mean、加权 tail。
- 用固定评估集比较目标区间误差和排序。
- 选择最稳的公式后再更新 `estimate_article()`。

建议默认候选：

```text
effective_difficulty = p50 + 0.7 * (p90 - p50) + 0.15 * (max - p95)
```

短文本或命中 token 少于 50 时降低 tail 权重。

### 第 4 阶段：lemma 和复合词增强，1-2 天

- 对未命中连字符词增加整体优先、拆分回退。
- 增加词库命中约束下的派生规则。
- 统一 `article_estimator.py` 和 `lemmatizer.py` 的职责。
- 更新 `docs/article_estimation.md`，修正文档与代码不一致的问题。

成功标准：

- 透明派生和连字符词覆盖提升。
- 不出现明显错误归一化，例如把不同词义强行合并。

### 第 5 阶段：coverage 插值校正，1 天，可选

仅在 A/B/C 后仍明显偏低时实施。

- 按未匹配词类别估计 difficulty。
- 对未知词低权计入。
- 输出 `coverage_adjustment` 到 `method` 或 debug 字段，方便解释。

成功标准：

- 低覆盖高难文章上升。
- 专有名词多的普通文章不被误推高。

## 建议实施路径

推荐顺序：

1. 先做方案 A 的简单扩展，生成 `stage_vocab_enhanced.json`，这是收益最大且解释性最强的一步。
2. 同时补一个文章级评估脚本，固定 C/F/P/K 的目标区间和排序指标。
3. 如果 A 后 C 仍低于预期，再做方案 B 的高分位加权。
4. 方案 C 作为覆盖率和健壮性增强，优先做连字符拆分，再做派生词族。
5. 方案 D 只作为最后的校正层，不建议在词库扩展前使用。

总体判断：本轮偏低的主因是词库覆盖，不是单纯公式问题。先把 C.txt 中 `algorithmic`, `societal`, `geopolitical`, `blockchain`, `hegemony`, `epistemological` 这类高价值缺失词补进词库，再评估是否需要改中位数公式，风险最低、收益最可控。
