# 英语词汇量估算工具：完整项目实现说明

## 1. 项目概述

本项目是在已有 `vocab_estimator` 核心预测模型之上扩展的课程设计完整系统。目标是让学生可以在网页端完成词汇认识测试，后端调用 Logistic 回归、Bootstrap 置信区间、Isotonic 排序约束等模型输出词汇量估计，并把测试记录保存到 SQLite。

技术栈：

- 后端：FastAPI
- 前端：纯 HTML/CSS/JavaScript
- 数据库：SQLite
- 核心模型：已有 `vocab_estimator` 包
- 批处理验证：Python CLI
- 浏览器自动化：Playwright

架构图：

```text
┌──────────────────────┐
│      web/index.html   │
│      web/app.js       │
└──────────┬───────────┘
           │ HTTP JSON
┌──────────▼───────────┐
│   server/main.py      │
│   FastAPI routes      │
└──────┬────────┬──────┘
       │        │
       │        ▼
       │   ┌─────────────────┐
       │   │ server/database │
       │   │ SQLite CRUD     │
       │   └─────────────────┘
       ▼
┌─────────────────────────────┐
│ vocab_estimator core package │
│ VocabBank / Sampler / Model │
└──────────┬──────────────────┘
           │
┌──────────▼───────────┐
│ verification modules │
│ batch + browser      │
└──────────────────────┘
```

## 2. 目录结构

```text
projects/vocab_estimator/
├── IMPLEMENTATION.md
├── README.md
├── main.py
├── requirements.txt
├── run.sh
├── data/
│   └── vocab_estimator.sqlite3
├── reports/
│   ├── batch_verification_report.json
│   └── browser_verification_report.json
├── server/
│   ├── __init__.py
│   ├── database.py
│   └── main.py
├── verification/
│   ├── __init__.py
│   ├── batch_verifier.py
│   └── browser_verifier.py
├── web/
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── vocab_estimator/
    ├── config.py
    ├── coverage.py
    ├── lemmatizer.py
    ├── sampler.py
    ├── vocab_bank.py
    └── vocab_model.py
```

## 3. 模块详解

### `vocab_estimator/`

职责：核心预测模型，不在本次扩展中重写。

关键调用：

- `VocabBank()`：构建词频词库、rank 查询、bucket 分层。
- `VocabularySampler.balanced_sample()`：生成测试词表。
- `VocabEstimator.estimate_single()`：单个学生词汇量估计。
- `VocabEstimator.estimate_groups()`：C/F/P/K 四类学员估计，并检查排序一致性。

输出字段包括：

- `point_estimate`：词汇量点估计。
- `vocabulary_range`：Bootstrap 90% 区间。
- `level`：初中/高中/四级/六级/专业级。
- `confidence`：高/中/低。

### `server/main.py`

职责：FastAPI HTTP API 层。

输入输出：

- 输入：前端或批处理提交的 JSON。
- 输出：估算结果、词库统计、历史记录。

关键设计：

- 使用 `@lru_cache` 缓存 `VocabBank` 和 `VocabEstimator`，避免每次请求重建词库。
- `parse_response_payload()` 同时支持 `[["word", true]]` 和 `[{"word":"word","known":true}]`。
- 额外提供 `GET /api/vocabulary/sample` 给前端生成测试题。

### `server/database.py`

职责：SQLite 数据库初始化和 CRUD。

数据表：

```sql
students(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  cet_score INTEGER,
  created_at TEXT
)

test_records(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id INTEGER NOT NULL,
  estimate INTEGER NOT NULL,
  level TEXT NOT NULL,
  confidence TEXT NOT NULL,
  range_low INTEGER NOT NULL,
  range_high INTEGER NOT NULL,
  responses_json TEXT NOT NULL,
  created_at TEXT
)
```

调用关系：

```text
POST /api/tests/save
  -> get_or_create_student()
  -> save_test_record()
  -> get_test_record()
```

### `web/index.html`, `web/app.js`, `web/styles.css`

职责：简单单页测试界面。

流程：

1. 读取词库统计。
2. 点击开始测试，请求 `/api/vocabulary/sample`。
3. 分页显示单词。
4. 学生选择“认识/不认识”。
5. 提交到 `/api/estimate`。
6. 展示词汇量、等级、置信区间。
7. 可调用 `/api/tests/save` 保存记录。

### `verification/batch_verifier.py`

职责：自动化批处理验证。

实验设计：

- 3 个抽样规模：每层 4、8、12 个词。
- 3 个噪声水平：0%、5%、10%。
- 每种组合 100 次。
- 总计 `3 x 3 x 100 = 900` 次。

输出指标：

- 每组均值。
- 方差。
- 平均绝对误差。
- C/F/P/K 排序一致性比例。
- Isotonic 调整后均值。

默认报告路径：

```text
reports/batch_verification_report.json
```

### `verification/browser_verifier.py`

职责：用 Playwright 打开 `testyourvocab.com`，自动勾选已知词，并尝试抓取网站结果，与本系统算法输出对比。

注意：

- `testyourvocab.com` 可能重定向到 Preply。
- 如果检测到 URL 中包含 `preply`，程序会输出 `redirected_to_preply` 状态。
- 网站结构可能变化，所以勾选和结果抓取是 best-effort，不影响本地算法结果输出。

## 4. 数据流

完整数据流：

```text
用户点击开始
  -> GET /api/vocabulary/sample
  -> VocabularySampler 从 VocabBank 分层抽样
  -> 前端分页展示单词
  -> 用户标记 known/unknown
  -> POST /api/estimate
  -> VocabEstimator.prepare_responses
  -> Logistic 回归估计
  -> Bootstrap 置信区间
  -> 等级映射
  -> 前端展示结果
  -> POST /api/tests/save
  -> SQLite 写入 students/test_records
  -> GET /api/tests/records 查询历史
```

四组学员数据流：

```text
POST /api/estimate/groups
  -> {C: responses, F: responses, P: responses, K: responses}
  -> estimate_single x 4
  -> 检查 C > F > P > K
  -> IsotonicRegression 或 PAVA 修正
  -> 返回原始估计和排序修正估计
```

## 5. API 文档

### `POST /api/estimate`

请求：

```json
{
  "responses": [
    {"word": "school", "known": true},
    {"word": "paradigm", "known": false}
  ]
}
```

也支持：

```json
{
  "responses": [
    ["school", true],
    ["paradigm", false]
  ]
}
```

响应：

```json
{
  "result": {
    "vocabulary_range": [3300, 6100],
    "point_estimate": 4700,
    "level": "四级",
    "confidence": "中",
    "confidence_interval_90": [3300, 6100],
    "baseline_estimate": 4500,
    "logistic_estimate": 4700,
    "sample_size": 36,
    "ignored_responses": 0
  },
  "input": {"response_count": 36},
  "vocab_bank": {
    "size": 30000,
    "used_fallback": false,
    "bucket_sizes": {"1k": 1000}
  }
}
```

### `POST /api/estimate/groups`

请求：

```json
{
  "groups": {
    "C": [["analysis", true], ["ubiquitous", true]],
    "F": [["analysis", true], ["ubiquitous", false]],
    "P": [["analysis", false], ["ubiquitous", false]],
    "K": [["school", true], ["ubiquitous", false]]
  }
}
```

响应核心字段：

```json
{
  "result": {
    "classes": {
      "C": {"point_estimate": 7600, "order_adjusted_estimate": 7600},
      "F": {"point_estimate": 6100, "order_adjusted_estimate": 6100}
    },
    "ordering_consistency": {
      "expected_order": "C>F>P>K",
      "was_consistent": true,
      "original_estimates": {"C": 7600, "F": 6100, "P": 4300, "K": 2700},
      "isotonic_estimates": {"C": 7600, "F": 6100, "P": 4300, "K": 2700}
    }
  }
}
```

### `GET /api/vocabulary/stats`

响应：

```json
{
  "size": 30000,
  "used_fallback": false,
  "bucket_sizes": {"1k": 1000, "2k": 1000},
  "bucket_boundaries": [1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000],
  "levels": [
    {"name": "初中", "low": 1500, "high": 2500}
  ]
}
```

### `GET /api/vocabulary/sample`

请求：

```text
GET /api/vocabulary/sample?per_bucket=4&seed=42
```

响应：

```json
{
  "items": [
    {"word": "school", "rank": 1234, "bucket": "2k"}
  ],
  "count": 36,
  "per_bucket": 4
}
```

### `POST /api/tests/save`

请求：

```json
{
  "student": {"name": "张三", "cet_score": 520},
  "responses": [{"word": "school", "known": true}],
  "result": {
    "point_estimate": 4700,
    "level": "四级",
    "confidence": "中",
    "vocabulary_range": [3300, 6100]
  }
}
```

响应：

```json
{
  "record": {
    "id": 1,
    "student_id": 1,
    "student_name": "张三",
    "estimate": 4700,
    "level": "四级",
    "confidence": "中",
    "range": [3300, 6100],
    "created_at": "2026-06-22 18:30:00"
  }
}
```

### `GET /api/tests/records`

请求：

```text
GET /api/tests/records?limit=20&offset=0
```

响应：

```json
{
  "records": [
    {
      "id": 1,
      "student_name": "张三",
      "estimate": 4700,
      "level": "四级",
      "confidence": "中",
      "range_low": 3300,
      "range_high": 6100
    }
  ],
  "total": 1,
  "limit": 20,
  "offset": 0
}
```

## 6. 部署与运行

安装依赖：

```bash
cd /home/akuai/.openclaw/workspace/projects/vocab_estimator
python3 -m pip install -r requirements.txt
```

如果 spaCy 英文模型或 `wordfreq` 安装失败，系统仍可运行，会回退到内置小词表和规则 lemmatizer。

启动后端和前端：

```bash
./run.sh
```

访问：

```text
http://127.0.0.1:8000
```

API 文档：

```text
http://127.0.0.1:8000/docs
```

批处理验证：

```bash
python3 -m verification.batch_verifier
```

快速 smoke 验证：

```bash
python3 -m verification.batch_verifier --runs 2 --bootstrap-iterations 5 --output reports/quick_batch.json
```

浏览器验证：

```bash
python3 -m playwright install chromium
python3 -m verification.browser_verifier --headed
```

## 7. 扩展指南

### 增加新算法

推荐在 `vocab_estimator/vocab_model.py` 中新增方法，例如：

```python
def bayesian_estimate(self, responses):
    ...
```

然后在 `estimate_single()` 中加入结果字段，或在 `server/main.py` 增加请求参数 `method` 来选择算法。API 层不应直接写算法逻辑。

### 连接真实数据库

当前数据库访问集中在 `server/database.py`。迁移到 MySQL/PostgreSQL 时：

1. 保留 `create_student()`、`save_test_record()` 等函数签名。
2. 替换 `sqlite3` 为 SQLAlchemy 或数据库驱动。
3. `server/main.py` 不需要大改。

### 部署到服务器

开发运行：

```bash
HOST=0.0.0.0 PORT=8000 ./run.sh
```

生产建议：

```bash
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 2
```

可在 Nginx 前面做反向代理，把 `/` 和 `/api/*` 都转发到 Uvicorn。

### 扩展前端

前端逻辑集中在 `web/app.js`：

- 调整每页题数：修改 `state.pageSize`。
- 增加二阶段自适应抽样：新增调用 `/api/vocabulary/sample` 或扩展 API。
- 增加图表：在 `renderResult()` 中展示 bucket 贡献。

## 8. 团队分工建议

3 人方案：

- 成员 A：后端与数据库。负责 `server/main.py`、`server/database.py`、接口联调。
- 成员 B：核心模型与批处理验证。负责解释 `vocab_estimator` 算法、运行 `batch_verifier.py`、整理统计表。
- 成员 C：前端与文档。负责 `web/` 页面、用户测试流程、课程设计报告整合。

4 人方案：

- 成员 A：FastAPI API 层。
- 成员 B：SQLite 数据库与测试记录。
- 成员 C：前端单页应用。
- 成员 D：批处理验证、Playwright 对比和实验报告。

建议工作量比例：

```text
后端 API：25%
数据库：15%
前端：20%
核心模型解释与集成：20%
验证与报告：20%
```
