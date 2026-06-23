# 英语词汇量估算工具

实现混合模型：

- 词频分层抽样
- Logistic 回归平滑估计
- 文档覆盖率校验
- 等级映射
- C/F/P/K 排序一致性校验

## 安装

```bash
pip install -r requirements.txt
```

如果 `wordfreq` 或 `en_core_web_sm` 下载失败，程序仍可运行：词库会退回内置小词表，lemmatizer 会退回规则归一化。

## 运行

```bash
python main.py --input examples/sample_input.json
```

输出为 JSON。每个班级包含：

- `vocabulary_range`: bootstrap 90% 区间
- `point_estimate`: Logistic 平滑点估计
- `level`: 初中/高中/四级/六级/专业或过渡等级
- `confidence`: 高/中/低
- `document_coverage`: 文档 top-N 覆盖率与 95%/98% 门槛
- `order_adjusted_estimate`: 按 C>F>P>K 修正后的估计

## 输入格式

```json
{
  "responses": {
    "C": [["the", true], ["analysis", false]],
    "F": [["school", true], ["sustain", false]],
    "P": [["water", true], ["paradigm", false]],
    "K": [["cat", true], ["ubiquitous", false]]
  },
  "documents": {
    "C": ["examples/doc_c.txt"],
    "F": ["examples/doc_f.txt"],
    "P": ["examples/doc_p.txt"],
    "K": ["examples/doc_k.txt"]
  }
}
```

## 模块

- `vocab_estimator/config.py`: 参数集中配置
- `vocab_estimator/vocab_bank.py`: 词库、rank、分桶
- `vocab_estimator/lemmatizer.py`: lemma 归一化
- `vocab_estimator/sampler.py`: 分层与自适应抽样
- `vocab_estimator/vocab_model.py`: 基线估计、Logistic 平滑、bootstrap、排序约束
- `vocab_estimator/coverage.py`: 文档覆盖率分析
- `main.py`: 命令行入口
