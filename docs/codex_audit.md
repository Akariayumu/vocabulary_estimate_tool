# vocab_estimator 代码审计与优化建议

审计时间：2026-06-23  
目标文件：`~/stu/vocab_estimator/docs/codex_audit.md`  
说明：本会话沙箱拒绝写入 `/home/akuai/stu/vocab_estimator`，因此本文档先落盘在当前可写工作区：`/home/akuai/.openclaw/workspace/codex_audit_vocab_estimator.md`。

## 总体结论

项目当前能支撑 v2 测试主流程，但整体处在“多代方案叠加”的状态：旧 bucket 模型、旧 sampler、v2 Rasch 引擎、文章估算、训练脚本和前端契约同时存在。最大风险不是语法错误，而是模型口径、题量口径、缓存口径和数据口径不一致，导致线上结果难解释、实验难复现。

快速检查：

- `web/app.js` 通过 `node --check`。
- Python 文件通过 `ast.parse` 语法检查。
- `python3 -m py_compile ...` 未完成，原因是当前沙箱对目标项目 `__pycache__` 写入返回只读文件系统错误，不是语法错误。
- `StratifiedQuiz.phase1_sample(adaptive=True/False)` 实测均返回 40 题，不是多处注释/API 文档写的 60 题。
- `parse_response_payload({"responses": [{"word": "x", "known": "false"}]})` 实测返回 `[("x", True)]`。

## 文件评分汇总

| 文件 | 设计评分 | 主要理由 |
|---|---:|---|
| `server/main.py` | 2.5/5 | API 功能完整，但混合多代模型入口；大量 `Any` payload；全局可变单例和 pickle 缓存有并发/陈旧风险。 |
| `vocab_estimator/stratified_quiz.py` | 3/5 | v2 核心思路清晰；但文档与实现偏离，CAT 未真正实现，CI/校准边界问题明显。 |
| `vocab_estimator/difficulty.py` | 3.5/5 | 难度公式可读、依赖少；但硬编码经验参数多，缺版本化和数据质量校验。 |
| `vocab_estimator/vocab_bank.py` | 3/5 | 词库封装简洁；但缓存失效策略不完整，异常吞掉过多，过滤规则会丢合法词。 |
| `vocab_estimator/sampler.py` | 2.5/5 | 旧流程功能还在；但与 v2 并行存在，重复逻辑多，性能是 O(词库) 扫描。 |
| `web/app.js` + `web/index.html` | 3/5 | 主流程可用，渲染有 escape；但 UI 文案、题量、状态机和错误处理与 v2 API 不完全一致。 |
| `optim/train_bucket_matrix.py` | 2/5 | 实验覆盖多，但脚本堆积严重；训练/生成/校准/输出耦合，重复代码多。 |
| `data/stage_vocab.json` | 3/5 | JSON 有效，字段完整，cluster 分布均衡；但缺 schema/version，重复保存原始表和派生统计。 |

## 1. `server/main.py` - API 层

### 设计评分：2.5/5

端点覆盖完整，前端所需 v2 测试、估算、保存、文章估算都有入口。主要扣分点是 API 层同时承担缓存、采样、题目生成、模型路由、记录保存和静态资源服务，且新旧模型入口并存。

### Bug 或不良实践

- `server/main.py:62-88`：`/tmp/vocab_bank_cache.pkl` 用 pickle 缓存完整 `VocabBank`，只比较 `vocab_bank.py` mtime。`DEFAULT_CONFIG`、`wordfreq` 数据、lemmatizer、bucket 边界变化都不会触发失效。
- `server/main.py:93-105`：`get_estimator()`、`get_coverage_analyzer()` 有 `lru_cache`，但底层 `get_vocab_bank()` 没有；不同路径可能构造多个 bank 实例。
- `server/main.py:112-114`、`server/main.py:500-502`、`server/main.py:550-553`：`StratifiedQuiz` 是全局单例，但端点会 `reseed()` 修改内部 RNG，并发请求会互相覆盖随机状态。
- `server/main.py:142-153` 与 `server/main.py:584-597`：普通 `/api/estimate` 走 bucket model，v2 `/api/vocabulary/quiz-v2/estimate` 走 Rasch；同一批 responses 因入口不同可能得到不同结果。
- `server/main.py:302-355`、`server/main.py:419-478`：旧 warmup/adaptive/stage2 端点仍暴露，且与 v2 命名接近，容易误用。
- `server/main.py:486-495`：文档写 60 stratified questions，实际返回 40。
- `server/main.py:497`、`server/main.py:550`、`server/main.py:555`：保留“Bug 2/Bug 3”补丁式注释，不适合长期生产代码。
- `server/main.py:533-534`：`quiz-v2-stage2` 假定 payload 是 dict；误传 list 会 500，而不是 400。
- `server/main.py:646-658`：注释写 trap 约 30%，实际 `TRAP_PROBABILITY = 0.1`。
- `server/main.py:750-752`：v2 distractor 每道题都 `list(TRANSLATIONS)` 并 shuffle 全量翻译词，22k 词量下重复分配和洗牌。
- `server/main.py:794-815`：`known` 未做 bool 类型校验，使用 `bool(known)`。缺失字段变 False，字符串 `"false"` 变 True。
- `server/main.py:123-129`：CORS `allow_origins=["*"]` 适合 demo，不适合公网生产。

### 性能瓶颈

- v2 题目生成热点是 distractor：每题全量 shuffle 翻译字典。
- 首次启动加载 wordfreq、构造 VocabBank、pickle 序列化成本高，且缓存失效不可靠。
- `vocabulary_summary()` 每次隐式调用 `get_vocab_bank()`，统计查询可能触发重构。

### 优化建议

- 用 Pydantic request schema 严格校验 `known: bool`。
- 生产前端只保留 v2；旧端点迁移到 `/api/legacy/*` 或下线。
- `StratifiedQuiz` 拆成只读数据缓存和每请求独立 RNG sampler。
- distractor 候选池预计算为 list，并按 difficulty cluster/source 建索引。
- 缓存 key 包含 config、依赖包版本、数据文件 mtime/hash；避免裸 pickle 长期缓存。

## 2. `vocab_estimator/stratified_quiz.py` - v2 Rasch 测试引擎

### 设计评分：3/5

核心结构合理：加载 `stage_vocab.json`，按 `cluster_20/100` 建索引，用 Rasch θ 拟合，再对词表求和。但实现与描述不一致，且极端答题模式的统计输出不可解释。

### Bug 或不良实践

- `stratified_quiz.py:1-9`、`stratified_quiz.py:131-139`、`server/main.py:491-495`：40/60 题描述冲突；实现实际是 40 题。
- `stratified_quiz.py:131-167`：`adaptive=True` 没有根据回答动态 refit θ，也没有 CAT 逐题选择，只是固定分层抽样。
- `stratified_quiz.py:169-199`：全对/全错时 MLE 被 clip 到极端，Fisher 信息很小，CI 极宽。实测全对输出 `point_estimate=9815`、区间 `[1890, 9826]`；全错输出 `point_estimate=0`、区间 `[0, 9826]`。
- `stratified_quiz.py:281-318`：`estimate_with_ci()` 未显式保证词汇区间上下界排序。
- `stratified_quiz.py:298-301` 与 `stratified_quiz.py:535-542`：先乘经验 0.8，再 `_calibrate()`，但没有训练版本或来源记录。
- `stratified_quiz.py:320-322`：`reseed()` 修改内部状态，配合 API 全局单例会造成并发串扰。
- `stratified_quiz.py:324-332`：`bank_words` 返回 `len(self._candidates)`，实际只有 11950 stage words，字段名误导。
- `stratified_quiz.py:477-480`：`strategy == "informative"` 只是返回按 difficulty 排序的前 n 个，不是按 information 排序。
- `stratified_quiz.py:493-517`：低置信 class 只在 `len(vals) == 2` 时识别，题量变化或重复提交会静默跳过。
- `stratified_quiz.py:519-533`：只对 11950 stage words 求和；文件头声称 sum over all 21738 bank words，已不一致。

### 性能瓶颈

- 初始化完整读取 21 万行 JSON；单例可接受，多进程冷启动成本明显。
- 每次估算循环 11950 个词并逐个 `_logit`，可向量化。

### 优化建议

- 先统一题量口径。
- 极端答案加入 MAP 先验/正则化 θ，或单独定义全对/全错 CI 规则。
- `_word_difficulties` 改成 numpy difficulty logits 数组。
- 数据索引与随机采样解耦，避免 mutable singleton。

## 3. `vocab_estimator/difficulty.py` - 难度打分

### 设计评分：3.5/5

职责单一，公式可读。问题是参数经验化且缺版本管理，输出写回 JSON 时也没有原子性保障。

### Bug 或不良实践

- `difficulty.py:20`：`_VOCAB_SIZE = 30_000` 与 `stage_vocab.json` 的 11950 词、VocabBank 约 21738 词规模不一致，应来自 config。
- `difficulty.py:40-52`：`_STAGE_MEDIAN_RANK` 是硬编码历史值，缺版本说明。
- `difficulty.py:57-68`：rank 到 priority 的 knots 完全经验化，缺来源和评估指标。
- `difficulty.py:103-110`：`alpha + beta` 没校验，传入错误权重会改变尺度。
- `difficulty.py:124-129`：假定 JSON 一定含 `stages` 和 `word_to_stage`，无 schema 错误提示。
- `difficulty.py:148-161`：未知 `first_stage` 强行当 IELTS，会掩盖数据错误。
- `difficulty.py:164-175`：`bank_words_total` 只递增不使用，是死变量。
- `difficulty.py:179-203`：原地重写大 JSON，没有临时文件 + rename；中断可能留下半文件。

### 性能瓶颈

- 每次完整加载并遍历 JSON，离线可接受，线上不应调用。
- `bank.get_rank()` 对几万词重复 lemmatize，可先缓存 rank map。

### 优化建议

- 增加 `DifficultyConfig`，记录 `alpha/beta/vocab_size/knots/version/generated_at`。
- 原子写文件，并写入 `meta.difficulty_config`。
- 输入 JSON 做 schema 校验，未知 stage 直接报错。

## 4. `vocab_estimator/vocab_bank.py` - 词库与缓存

### 设计评分：3/5

抽象清楚，`VocabItem` 轻量。主要风险在缓存、异常处理和词形过滤，且 API 层又套了一层 pickle 缓存。

### Bug 或不良实践

- `vocab_bank.py:143-155`：`/tmp/vocab_bank_words.pkl` 无版本 key；`config.vocab_size`、`wordfreq` 版本、过滤规则变化后仍会复用旧缓存。
- `vocab_bank.py:151`：从缓存加载后直接设 `_wordfreq_available=True`，缓存可能来自旧逻辑或损坏数据。
- `vocab_bank.py:154-155`、`vocab_bank.py:171-176`：大范围 `except Exception` 静默降级，掩盖权限、损坏缓存、依赖错误。
- `vocab_bank.py:176-177`：任何 wordfreq 加载异常都会回退到 `FALLBACK_WORDS`，线上可能悄悄退化为小词表。
- `vocab_bank.py:223-234`：`word.isalpha()` 会过滤连字符、撇号、多词短语；`COMMON_PROPER_NAMES` 过滤 `china`、`america`，与 stage vocab 策略冲突。
- `vocab_bank.py:276-277`：normalize 时没复用 lower，依赖 lemmatizer 内部大小写处理。
- `vocab_bank.py:236-240`：bucket labels 从 config 派生，但训练脚本另硬编码一份，存在漂移风险。

### 性能瓶颈

- 首次 `top_n_list("en", vocab_size * 3)` 是主要启动成本。
- `_build_items()` normalize/filter 全量候选词，只能靠可靠缓存缓解。

### 优化建议

- 缓存文件名包含 config hash 和 wordfreq version。
- 生产环境缺 wordfreq 应 fail fast 或健康检查告警，不要静默 fallback。
- bucket labels 全项目统一从 config 派生。

## 5. `vocab_estimator/sampler.py` - 旧采样器

### 设计评分：2.5/5

旧 sampler 对 v1 bucket 测试仍可用，但 v2 已切到 `StratifiedQuiz` 后继续保留大量旧路径，会增加维护成本。

### Bug 或不良实践

- `sampler.py:43-49`：`WARMUP_LEVELS` 注释和类型标注不一致；第三项实际是 level index，第四项才是 exam set。
- `sampler.py:52-80`：每次调用都读 exam vocab 文件，没有缓存。
- `sampler.py:96-103`：`balanced_sample()` 不 shuffle 总结果，bucket 顺序固定。
- `sampler.py:122-123`：如果 `words_by_bucket` 为空会除以 0。
- `sampler.py:134-138`：更新 seen 时重复遍历整个 selected。
- `sampler.py:198-244`：旧 Stage 2 用 bucket rate；v2 Stage 2 用 difficulty class。两个概念并存。
- `sampler.py:246-271`：`sources` 类型标注为 `dict[str, int]`，实际存字符串。
- `sampler.py:255-262`：warmup 每个难度层都扫描全量 `vocab_bank.items`。
- `sampler.py:301-303`：注释说按 warmup level 使用不同 exam set，实际总是 `cet6`。

### 性能瓶颈

- warmup 和 adaptive 都有全量扫描词库行为。
- exam vocab 文件反复读取。

### 优化建议

- 如果线上只用 v2，旧 sampler 标记 legacy，端点下线后删除。
- 为 rank range、exam set 建缓存索引。
- 修正类型注解和注释。

## 6. `web/app.js` + `web/index.html` - 前端

### 设计评分：3/5

前端状态机能跑完 v2 主流程，渲染有 `escapeHtml()`。主要问题是 UI 文案、题量和 API 语义没有完全跟随后端 v2。

### Bug 或不良实践

- `web/app.js:58-60`、`web/app.js:410`：注释仍写 60 题，实际 40。
- `web/app.js:65`：强制 `balanced=true`，因此永远不用后端默认 adaptive 采样。
- `web/app.js:164-181`：主测试完成显示汇总，Stage2 完成自动 submit，两阶段体验不一致。
- `web/app.js:218`、`web/app.js:300`：v2 题目 rank 固定为 0，却仍显示 `rank 0 · cluster_x`。
- `web/app.js:346-354`：MC 选择只记录 known bool，不记录正确答案、选项快照，无法完整复盘。
- `web/app.js:597-648`：文章估算错误处理依赖外层 click handler，函数自身没有 catch。
- `web/app.js:654-673`：`loadRecords()` 内部没有 try/catch，失败时视图已切到 records，可能空白。
- `web/index.html:26`：写 40 题是当前实现正确值，但与 JS 注释和后端文档冲突。
- `web/index.html:43-47`：初始标题/progress 仍是 60。
- `web/index.html:74`：JS 版本号硬编码，需要手工改。

### 性能瓶颈

- 前端 40 题渲染成本不高。
- 题目生成慢主要在后端 distractor。

### 优化建议

- 题量完全由 API `count` 驱动，移除硬编码 40/60。
- v2 不显示 `rank 0`，改显示“难度类 X/20”或隐藏元信息。
- 保存完整题目快照：`word/known/choice/answer/options/phase/cluster_20`。
- 静态资源版本改为构建 hash 或服务端自动注入。

## 7. `optim/train_bucket_matrix.py` - 训练脚本

### 设计评分：2/5

这是实验脚本累计形态：生成器、训练器、校准器、评估打印、参数保存全部在一个 1900 行文件里。研究探索可接受，作为可复现训练管线偏脆弱。

### Bug 或不良实践

- `train_bucket_matrix.py:89-91`：`BUCKET_LABELS`、`CALIB_BOUNDARIES`、`MAX_V` 硬编码，可能和 `DEFAULT_CONFIG` 漂移。
- `train_bucket_matrix.py:190-206` 与 `train_bucket_matrix.py:1430-1519`：校准函数支持 tanh + piecewise，注释又说 identity；实际仍会训练 slopes。
- `train_bucket_matrix.py:213-1182`：多个 synthetic user generator 重复构造 bucket_words、n_per_bucket、responses、bucket rates。
- `train_bucket_matrix.py:397-410`、`train_bucket_matrix.py:537-548`、`train_bucket_matrix.py:783-797`、`train_bucket_matrix.py:1109-1115`：多处手写带权无放回抽样，大样本时效率差。
- `train_bucket_matrix.py:927-1016`：假定 `official_vocab` 列表“本身就是累加的”，但没有校验集合包含关系。
- `train_bucket_matrix.py:1074-1082`：如果课程集合不严格嵌套，`known_base` 与实际 `known` 大小可能不一致。
- `train_bucket_matrix.py:1189-1297` 和 `train_bucket_matrix.py:1300-1423`：PyTorch 与 NumPy 两套训练实现并行维护，梯度容易漂移。
- `train_bucket_matrix.py:1290-1291`：变量 `al` 未使用。
- `train_bucket_matrix.py:1389-1395`：NumPy anchor gradient 多乘了 2，与 loss 梯度不一致。
- `train_bucket_matrix.py:1584-1935`：`main()` 过长，参数解析、数据生成、训练、保存、打印耦合。
- `train_bucket_matrix.py:1655-1663`：无论最终 `gen_mode`，先生成 12 个 `freq_users`；某些模式后面又覆盖，造成不必要启动成本。
- `train_bucket_matrix.py:1887-1917`：输出 `accuracy` 用 `str(user['vocab_size'])` 做 key，重复 vocab size 会覆盖。
- `train_bucket_matrix.py:1918-1920`：写输出文件没有确保父目录存在，也没有原子写入。

### 性能瓶颈

- 大量 Python set/list 循环和重复 bucket 统计。
- `random.choices` 去重补抽在大样本时退化严重。
- 打印所有用户覆盖率和 accuracy 时输出量可能非常大。

### 优化建议

- 拆成 `data_generation.py`、`model.py`、`train.py`、`evaluate.py`。
- 所有 bucket label 和 calibration boundary 从 config 读取。
- 抽公共函数：`build_bucket_words()`、`sample_responses()`、`compute_bucket_rates()`、`weighted_sample_without_replacement()`。
- 输出 metadata：git commit、data hash、config hash、seed、生成模式、依赖版本。
- 给 NumPy/PyTorch 梯度加数值梯度测试，尤其 anchor loss。

## 8. `data/stage_vocab.json` - 词表数据

### 设计评分：3/5

JSON 语法有效，`word_to_stage` 共 11950 词，`difficulty`、`cluster_20`、`cluster_100` 字段覆盖完整；`cluster_20` 分布均衡，每类约 597-598 个词。数据能支撑当前 v2 采样。

### Bug 或不良实践

- `stage_vocab.json:2-20`：`meta` 只记录 sources，没有记录生成脚本、生成时间、difficulty 算法版本、cluster 算法版本、输入文件 hash。
- `stage_vocab.json:23-30703`：`stages.*.words` 总词条 30615，unique 11950；跨 stage 重复 7555 个词，文件体积被放大。
- `stage_vocab.json:27`、`stage_vocab.json:905`、`stage_vocab.json:3383`、`stage_vocab.json:7946`、`stage_vocab.json:13322`、`stage_vocab.json:21671`：`and` 在多个 stage 重复出现；消费方若不走 `word_to_stage` 会重复计数。
- `stage_vocab.json:54` 与 `stage_vocab.json:38189` 附近：`children` 保留为词条，未 lemma 到 `child`，需确认是否符合测试词形策略。
- `stage_vocab.json:55` 等多处：`china` 在 stage 中存在，但 `VocabBank.COMMON_PROPER_NAMES` 会过滤 `china`，stage v2 与旧 bank 策略不一致。
- `stage_vocab.json:30705`：`word_to_stage` 从 3 万行后开始，文件同时保存原始 stages 和派生索引，人工 review 困难。
- `stage_vocab.json` 尾部 `overlap_matrix`：属于可派生统计，占用大量行，容易与 stages/word_to_stage 不一致。

### 性能瓶颈

- `StratifiedQuiz` 冷启动完整加载 21 万行 JSON。
- JSON 同时包含重复 stages、word_to_stage、overlap_matrix，不利于快速加载。

### 优化建议

- 拆分为 `stage_vocab.words.json`、`stage_vocab.index.json`、`stage_vocab.stats.json`，线上只加载 index。
- 增加 `schema_version`、`generated_at`、`generator`、`difficulty_config`、`cluster_config`、`input_hashes`。
- 增加数据校验脚本：字段完整性、difficulty 范围、cluster 边界、stage 去重、bank 策略一致性。

## 优先级表

| 优先级 | 问题 | 影响 | 建议方案 |
|---|---|---|---|
| P0 | v2 题量口径不一致：后端文档/前端注释写 60，实际返回 40。 | 用户体验和实验报告口径不可信，测试数据难复现。 | 统一定义题量；若保留 40，更新所有 60 文案。 |
| P0 | `parse_response_payload()` 使用 `bool(known)`，字符串 `"false"` 会变 True。 | 外部 API 或错误前端提交会直接污染估计结果。 | 用 Pydantic schema 严格校验 bool；缺失或非 bool 返回 400。 |
| P0 | 全局 `StratifiedQuiz` 单例被 `reseed()` 修改。 | 并发请求随机状态串扰。 | 数据只读缓存，采样 RNG 每请求独立传入。 |
| P0 | Rasch 极端答案 CI 失真，全错可给出 `[0, 9826]`。 | 结果解释严重不稳定。 | 引入 MAP 先验/正则化 θ，极端模式单独处理 CI，并加单元测试。 |
| P1 | v1 bucket、旧 sampler、v2 Rasch 多套模型入口并存。 | 保存、估算、历史回放可能混用不同模型。 | 明确生产模型为 v2；旧端点迁移到 legacy 或下线。 |
| P1 | pickle 缓存失效只看单文件 mtime。 | 配置或数据更新后仍使用旧词库。 | 缓存 key 加 config/data/dependency hash。 |
| P1 | distractor 生成每题全量 shuffle `TRANSLATIONS`。 | 题目生成延迟随词表线性增长。 | 预构建翻译候选池，按 cluster/source 建索引。 |
| P1 | `stratified_quiz.py` 声称 CAT，但实现不是逐题自适应。 | 产品/文档/算法描述不一致。 | 改名为 stratified sampling，或实现真正 CAT。 |
| P1 | `stage_vocab.json` 缺 schema/version/hash。 | 难以追踪 difficulty/cluster 来源。 | 在 meta 写入生成配置和 input hashes；新增校验脚本。 |
| P1 | 训练脚本 NumPy anchor gradient 多乘 2。 | fallback 训练结果与 PyTorch 不一致。 | 做数值梯度测试并修正 NumPy 实现。 |
| P1 | 训练输出 accuracy 用 vocab_size 字符串做 key。 | 重复 vocab size 用户会被覆盖。 | 输出 list，或 key 使用 user index + vocab_size。 |
| P2 | API 大量使用 `Any = Body(...)`。 | 错误响应不稳定，OpenAPI 文档价值低。 | 为端点定义 request/response models。 |
| P2 | `VocabBank` 静默 fallback 到小词表。 | 依赖缺失时服务看似可用但结果质量崩溃。 | 生产环境 fail fast；健康检查暴露 fallback 状态。 |
| P2 | 前端保存记录只存 `word/known`。 | 无法复盘选项、trap、正确答案和 phase。 | 保存完整题目快照与用户选择。 |
| P2 | `sampler.py` 旧 warmup/adaptive 逻辑仍扫描全词库。 | 旧端点保留时请求成本高。 | 下线旧端点或为 rank range/exam vocab 建缓存索引。 |
| P2 | `train_bucket_matrix.py` 单文件 1900 行且重复生成逻辑。 | 后续实验难维护、难测试。 | 拆模块并抽公共采样/统计函数。 |
