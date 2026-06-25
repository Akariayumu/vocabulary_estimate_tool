# 文章词汇量估算

本文档说明 `/api/v2/estimate/article` 的实现口径。新版本只使用
`data/stage_vocab.json`，不依赖 `wordfreq`、`VocabBank` 或旧文章估算函数。

## 数据源

- `word_to_stage`：词条索引。
- `difficulty`：0-1 难度分。
- `cluster_20`：20 档难度聚类，用于返回文章难度分布。
- `first_stage` / `stages`：用于把估算词汇量映射到教育阶段。
- `translation`：随词条保留，但文章估算当前不需要返回逐词翻译。

当前清洗后的 `stage_vocab.json` 包含 11418 个可用词条。

## 分词

实现位置：`vocab_estimator/article_estimator.py`

分词使用 Python 标准库 `re`：

```python
[a-z]+(?:-[a-z]+)*
```

处理步骤：

1. 将文章转为小写。
2. 用正则抽取英文单词，标点自动丢弃。
3. 去掉内置停用词，例如 `a`, `the`, `is`, `are`, `of`, `in`, `to`, `it`,
   `that`, `this` 等。
4. 用剩余内容词到 `stage_vocab.word_to_stage` 中查找 difficulty。

不使用 NLP 包，不做词频 rank 查询。

## 估算方法

对匹配到词库的文章内容词 token 统计 difficulty 分布：

- `p25`
- `median`
- `p75`
- `min`
- `max`
- `mean`
- `histogram`
- `cluster_20`

词汇量估算采用累计分布法：

```text
estimated_vocab = count(stage_vocab.difficulty <= article_median_difficulty)
```

这等价于：

```text
estimated_vocab = cumulative_percentile * 11418
```

其中 `cumulative_percentile` 使用 `stage_vocab.json` 的真实 difficulty 分布计算。
因为 difficulty 在词库中不是均匀分布，同一个 difficulty 不一定对应线性词汇量。

## Level 映射

`level` 使用 `first_stage` 的累计词数映射到教育阶段：

| 阶段 | 累计词数含义 |
| --- | --- |
| 小学三年级到九年级 | 教材阶段累计首次出现词 |
| 高中 | 高中首次出现词累计后的位置 |
| 大学四级 | 四级首次出现词累计后的位置 |
| 大学六级 | 六级首次出现词累计后的位置 |
| 雅思 | 词库最高阶段 |

因此返回的 level 表示“该文章的中位难度大约落在词库课程阶段的哪个累计位置”。

## API

请求：

```http
POST /api/v2/estimate/article
Content-Type: application/json
```

```json
{
  "article": "Students analyze evidence and compare information..."
}
```

响应示例：

```json
{
  "difficulty_median": 0.65,
  "estimated_vocab": 1744,
  "level": "九年级",
  "coverage": {
    "stage_vocab": 0.85,
    "difficulty_distribution": {
      "p25": 0.52,
      "median": 0.65,
      "p75": 0.76,
      "min": 0.28,
      "max": 0.93,
      "mean": 0.64,
      "histogram": {
        "0.0-0.2": 0,
        "0.2-0.4": 2,
        "0.4-0.6": 8,
        "0.6-0.8": 12,
        "0.8-1.0": 3
      },
      "cluster_20": {
        "4": 3,
        "5": 7,
        "6": 4
      }
    }
  }
}
```

实际数值取决于文章内容和当前 `stage_vocab.json` 的 difficulty 分布。

## 限制

- 只做简单英文分词，不做 lemma 还原。
- 复数、时态变化如果不在 `stage_vocab.json` 中，会计入未匹配词。
- `coverage.stage_vocab` 是内容词 token 覆盖率，不是 unique word 覆盖率。
- 文章过短或没有任何词命中 stage vocab 时，API 返回 HTTP 400。
