# 课程设计报告（软件工程）

## 封面信息

| 项目 | 内容 |
| --- | --- |
| 学校 | 广州大学 |
| 学院 | 计算机科学与网络工程学院 |
| 专业年级 | 软件工程 |
| 成员 | 待填（单人完成） |

## 一、课程设计目的

本课程设计以"英语词汇量估计工具"为对象，按照软件工程思路完成需求分析、系统设计、编码实现、测试验证与部署维护的完整流程，训练工程综合能力。

词汇量估计不是简单计数，它涉及词库构建、难度标注、用户答题建模、前后端交互、数据库持久化和模拟评估，适合作为软件工程综合训练题目。项目需要支持两类场景：(1)用户通过选择题或二元判断题完成词汇量测试，系统返回估计值、等级和置信区间；(2)用户输入英文文章，系统估计理解该文章所需词汇量。项目经历了从 V1 词频桶模型到 V2 Rasch 1PL IRT 模型的演进，体现了设计→实现→测试→修正的迭代过程。

## 二、课程设计内容

### 2.1 设计题目

英语词汇量估计工具。

### 2.2 提交内容

**数据层**：VocabBank 词族词库 21,738 词；stage_vocab.json 共 11,418 个可用词，含 difficulty 分数、cluster_20/cluster_100 聚类标签、教育阶段标注和翻译字段；英中翻译字典 22,887 条，覆盖率 100%；考试词表（高考/CET-6/TOEFL/GRE/COCA20000 等多套）。

**算法设计**：核心模型为 Rasch 1PL IRT，辅助有 V1 分桶矩阵参数模型、分层自适应采样（StratifiedQuiz）、文章词汇量估算（累计百分位法）、PAVA 保序回归群组对比。

**前后端与数据库**：后端 FastAPI + uvicorn，前端 HTML + JS + CSS，SQLite 数据库存储学生记录和测试记录。

### 2.3 功能列表

**基本功能：**
1. 词汇量选择测试（四选一中文释义 / 二选一认识不认识）
2. 文章词汇量估算（输入英文文章，输出词汇量估计 + 教育阶段 + 难度分布）
3. 两阶段自适应测试（Phase 1 分层抽样 40 题 + Phase 2 低置信类边界细化）
4. 词汇量估算置信区间（Fisher 信息量 → 95% CI）
5. 后台批处理测试（tests/simulation_eval.py 批量评估）

**扩展功能：**
1. 群组对比（C/F/P/K 四类，PAVA 保序回归约束 C ≥ F ≥ P ≥ K）
2. Streaming 渐进式测试细化（边答题边更新估计结果，支持 15/25/30/40 题渐进）
3. 考试词表锚点校准（中考→高考→CET-4→CET-6 嵌套词表）
4. 参数训练管线（合成数据生成 + 桶矩阵训练 + 校准参数训练）

## 三、项目环境要求

硬件方面，普通 PC 即可完成推理和演示，V100 GPU 用于参数训练和模拟。软件环境为 Linux、Python 3.12+、FastAPI、uvicorn、NumPy/PyTorch、wordfreq、spaCy en_core_web_sm。开发工具为 VS Code 和 Git，部署使用 nginx 反向代理 + systemd 服务。

## 四、总体设计

### 4.1 系统架构

系统按数据层、模型层、API 层和前端层分层组织：

```text
[用户浏览器] → nginx (154.9.242.48) → systemd: vocab-estimator
    → uvicorn FastAPI (localhost:7860)
      ├── /api/vocabulary/quiz-v2        (分层Rasch测试)
      ├── /api/vocabulary/quiz-v2/estimate (MLE估算)
      ├── /api/v2/estimate/article       (文章估算)
      └── /api/tests/save               (保存记录)
```

### 4.2 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | Python FastAPI + uvicorn |
| 前端 | 静态 HTML + JS + CSS |
| 数据持久化 | SQLite |
| 词频数据 | wordfreq + 自建 vocab_bank |
| 词形还原 | spaCy en_core_web_sm + 规则 fallback |
| 机器翻译 | 自建英中字典 22,887 条 |
| 参数训练 | PyTorch / NumPy (V100) |
| 部署 | nginx + systemd |

### 4.3 模型演进

**V1（退役）**：2 参数 Logistic 回归 + 分桶矩阵参数模型，使用 21,738 词族和 9 个频段桶。公式为 P(known|bucket_b) = sigmoid(θ_b + γ_u)，vocab_size = Σ bucket_size_b × sigmoid(θ_b + γ_u)。问题：低词汇量用户严重高估 ~4000 词，桶间硬边界划分不够平滑。

**V2（当前）**：Rasch 1PL IRT 模型，使用 11,418 词的 difficulty 标注和 20 个 cluster_20 类。公式为 P(word_j known | θ) = σ(θ - d_j)，d_j = logit(difficulty_j)。用户参数 θ 通过 Newton-Raphson MLE 拟合，置信区间基于 Fisher 信息量。模拟评估 MAE=363，R²=0.977。

## 五、算法设计

### 5.1 Rasch 模型（IRT 1PL）

核心模型为单参数逻辑斯蒂模型：

```
P(word_j known | θ) = σ(θ - d_j),  σ(x) = 1/(1 + exp(-x))
d_j = logit(difficulty_j) = ln(difficulty_j / (1 - difficulty_j))
```

N(0,2) 先验防极端。Newton-Raphson 迭代求解 θ：

```
θ_{t+1} = θ_t + (Σ(y_j - σ(θ_t - d_j))) / (Σσ(θ_t - d_j)(1 - σ(θ_t - d_j)))
```

词汇量估计：vocab_size = Σ_j σ(θ - logit(difficulty_j)) × 0.8

置信区间：SE(θ) = 1 / √Σσ(θ-d_j)(1-σ(θ-d_j))，95% CI = θ ± 1.96 × SE(θ)

### 5.2 分层采样（StratifiedQuiz）

Phase 1：40 题，20 个 cluster_20 类 × 2 题。5 个 diagnostic spread 类（c20: 0,5,10,15,19）用 extremes 策略（最易+最难），其余 15 类用 balanced 策略（中间难度）。

Phase 2：低置信类（答对 1/2 题）追加 8 题/类，选 Fisher 信息量最大的词（|θ - d_j| 最小），平均追加 ~44 题。

### 5.3 文章词汇量估算

输入英文文章，分词（正则）+ 去停用词（541 个）+ lemma 归一化（规则型 + 60+ 不规则词）→ 匹配 stage_vocab → 计算 difficulty 分布 → 累计百分位估算：

```
estimated_vocab = count(stage_vocab.difficulty ≤ article_median_difficulty)
```

### 5.4 难度评分

```
difficulty = 0.60 × norm_stage + 0.40 × norm_rank
norm_stage = (priority - 1) / 10,  norm_rank = log(rank + 1) / log(30001)
```

教育阶段权重 0.60 确保跨阶段区分度，词频 rank 权重 0.40 保证同阶段内区分度（spread ~0.37）。

## 六、验证方法

### 6.1 合成用户模拟评估

tests/simulation_eval.py 生成 2000 个虚拟用户（true vocab ∈ [1000, 15000]），二分搜索反推真 θ，P(known) = σ(true_θ - d_j) 生成回答，执行完整 Phase 1+2 测试流程后评估指标。

### 6.2 主要实验结果

| 实验配置 | MAE | R² | 题量 |
|---------|:---:|:--:|:----:|
| Phase 1 only (40题) | 596 | 0.940 | 40 |
| Phase 1+2 (~84题) | **363** | **0.977** | ~84 |
| Hybrid bisection+CAT (40题) | 443 | 0.961 | 40 |
| 25题初估 | 819 | 0.891 | 25 |
| 30题最优折中 | 701 | 0.917 | 30 |

主模拟评估：MAE=363，RMSE=455，R²=0.977，相关系数=0.989，平均偏差=-134。

题量消融（300 用户，streaming 顺序）：10 题 MAE=1317, R²=0.720；15 题 MAE=1081, R²=0.813；20 题 MAE=928, R²=0.863；25 题 MAE=819, R²=0.891；30 题 MAE=701, R²=0.917；35 题 MAE=649, R²=0.929；40 题 MAE=596, R²=0.940。30 题后边际收益骤降。

### 6.3 分桶精度

| 词汇量区间 | 用户数 | MAE | R² | 偏差 |
|:---------:|:-----:|:---:|:--:|:----:|
| 低 (1k-3k) | 369 | 334 | 0.447 | +2 |
| 中 (3k-8k) | 965 | 374 | 0.896 | -91 |
| 高 (8k-15k) | 666 | 363 | 0.796 | -271 |

### 6.4 算法稳定性测试（批处理验证）

为评估估算算法的稳定性，设计了基于随机抽样的后台批处理验证方法：

1. 从词库中选取一个公认词汇列表 A（例如 COCA20000 词表，该词表在本算法设计中未使用过）
2. 对列表 A 进行多次随机采样，生成不同长度（200/300/400 词）和不同认识比例（10%/20%/30% 认识）的测试样本
3. 共 3×3=9 种组合，每种组合重复测试 100 次，统计估算结果的平均值和方差
4. 分析：(1)相同比例下不同长度的结果是否一致；(2)相同长度下不同比例是否呈线性关系；(3)方差是否随样本量增大而减小

此方法可有效评估算法在已知词表上的系统偏差和随机误差，为置信区间的设定提供实证依据。本系统的 StreamQuiz API 天然支持这类批处理测试，可直接通过 /api/vocabulary/quiz-v2/estimate 批量处理。

### 6.5 与行业产品 testyourvocab.com 的对比验证

设计对比验证方案（作为后续验证构想）：

1. **批量数据获取**：通过浏览器自动化模拟用户操作 testyourvocab.com，每次获取认识的词列表 R_i、不认识的词列表 U_i、网站估计值 C_i，模拟 100 次得到 {(R_i, U_i, C_i)}
2. **本算法估计**：对同组 {(R_i, U_i)} 运行本系统估算算法，输出 D_i
3. **对比分析**：比较 {D_i} 和 {C_i} 的 MAE、相关系数、误差分布

## 七、演示测试

### 7.1 GUI 演示测试

Web 界面支持：(1)选择题模式（四选一中文释义，含 ~30% 陷阱题）；(2)Binary 模式（认识/不认识，当翻译不可用时自动降级）；(3)文章估算模式（输入文章全文）。功能包括键盘快捷键、进度展示、汇总页、回退修改和 Stage 2 细化。结果页展示词汇量估算值、教育阶段映射、置信区间。

### 7.2 后台批处理测试

tests/simulation_eval.py 自动生成合成用户、执行完整测试流程、汇总评估指标。scripts/explore_question_count.py 探索最优题量。scripts/validate_hybrid_bisection.py 对比混合方案。结果保存到 outputs/ 目录。

### 7.3 实际估计：测试语料词汇量估计

对四类学员文档（C.txt > F.txt > P.txt > K.txt）使用文章估算接口得到结果：

| 文档 | 估算词汇量 | 教育阶段 | difficulty 中位数 | Token覆盖率 | Unique覆盖率 | 内容词 | 唯一词 |
|:---:|:---------:|:--------:|:---------------:|:----------:|:------------:|:-----:|:-----:|
| **C.txt** | **2,663** | **高中** | 0.7363 | 86.4% | 85.0% | 508 | 399 |
| **F.txt** | **2,160** | **高中** | 0.7057 | 89.8% | 89.7% | 402 | 329 |
| **P.txt** | **1,028** | **七年级** | 0.5452 | 83.1% | 85.1% | 402 | 248 |
| **K.txt** | **506** | **小学五年级** | 0.4162 | 91.9% | 95.3% | 148 | 107 |

**分析**：
- 排序完全一致：C(2663) > F(2160) > P(1028) > K(506)
- 内容覆盖合理：C 为学术文本（AI伦理、气候变化），F 为通用内容（体育、好莱坞），P 为故事阅读，K 为基础英语
- C.txt 未匹配词含 AI-driven, algorithmic, blockchain 等专业词（13.6% 未覆盖），说明词库需扩展
- K.txt 覆盖率最高（95.3%），基础词基本全覆盖
- 建议确信度区间：C ±400、F ±400、P ±350、K ±300

## 八、完成情况

已实现基本功能：词汇量选择测试（Rasch MLE + Fisher 置信区间）、文章词汇量估算、两阶段自适应测试、后台批处理测试。

额外实现功能：群组对比 (PAVA)、Streaming 渐进式细化、考试词表锚点校准、参数训练管线。

**自评主要亮点**：
1. 基于 IRT 的严谨统计基础（Rasch 1PL），可解释性强
2. 合成数据模拟验证 MAE=363、R²=0.977，接近理论下界
3. 完整工程闭环：从数据收集到部署上线
4. 实际语料估计结果 C > F > P > K 排序完全一致

**自评主要缺陷**：
1. 缺乏真实用户交叉验证（需对接 CET-4/6 成绩）
2. stage_vocab 11,418 词偏少，高阶用户分辨率不足
3. Phase 2 细化提升有限（追加 44 题几乎不增加精度）
4. 难度标注基于经验公式，未经答题数据校准

## 九、小组分工

单人完成 100%。本人负责：需求分析、词库构建、难度评分设计、Rasch 模型实现、分层采样设计、FastAPI 后端开发、Web 前端交互、SQLite 数据库设计、合成数据模拟评估、测试语料实际估算、nginx 部署上线、文档撰写。

## 十、参考文献

1. Rasch, G. (1960). Probabilistic Models for Some Intelligence and Attainment Tests.
2. Lord, F. M. (1980). Applications of Item Response Theory to Practical Testing Problems.
3. Baker, F. B., & Kim, S.-H. (2004). Item Response Theory (2nd ed.). CRC Press.
4. testyourvocab.com - English Vocabulary Size Test.
5. wordfreq - A database of word frequencies in natural language (Speer & Chin).
6. 义务教育英语课程标准 (2022), 中华人民共和国教育部.
7. 全国大学英语四、六级考试大纲 (2016).
8. 项目内部文档：docs/TECHNICAL.md, docs/experimental_summary.md, docs/project_review.md.
