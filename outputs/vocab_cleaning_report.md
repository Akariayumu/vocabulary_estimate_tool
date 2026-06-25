# 词库清洗报告

- 生成时间：2026-06-25T16:15:41+08:00
- 允许保留的词形：`^[A-Za-z][A-Za-z'-]*$`（英文字母，允许连字符 `-` 和撇号 `'`）
- 清洗脚本：`scripts/clean_vocab.py`

## 汇总

| 词库 | 输入文件 | 总词数 | 脏词数 | 缺失翻译数 | 缺 translation 字段 | 空翻译字符串 | 清洗后词数 | 输出文件 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v1 | `data/stage_vocab.json` | 11418 | 124 | 0 | 0 | 0 | 11294 | `data/stage_vocab_clean_v1.json` |
| v2 | `data/stage_vocab_v2_clusterv1.json` | 19801 | 124 | 8383 | 0 | 8383 | 19677 | `data/stage_vocab_clean_v2.json` |

## 跨词库脏词来源

- 去重脏词数：124
- 示例（前20个）：
  - `a.m`：v1, v2
  - `a.m.`：v1, v2
  - `a.m.(=a.m`：v1, v2
  - `arise(arose,arisen`：v1, v2
  - `ask…for`：v1, v2
  - `autumn/fall`：v1, v2
  - `awake(awoke,awoken`：v1, v2
  - `b/l`：v1, v2
  - `backward(s`：v1, v2
  - `bad(worse,worst`：v1, v2
  - `ballpoint“＝”ballpointpen`：v1, v2
  - `beat(beaten`：v1, v2
  - `become(became,become`：v1, v2
  - `begin(began,begun`：v1, v2
  - `bend(bent`：v1, v2
  - `between…and`：v1, v2
  - `bike“＝”bicycle`：v1, v2
  - `bite(bit,bitten`：v1, v2
  - `blow(blew,blown`：v1, v2
  - `boss/bqs`：v1, v2

## v1

- 输入：`data/stage_vocab.json`
- 输出：`data/stage_vocab_clean_v1.json`
- 总词数：11418
- 脏词数：124
- 缺失翻译数：0
- 缺 translation 字段：0
- 空翻译字符串：0
- 清洗后词库大小：11294
- 非法字符统计：`(`: 75, `,`: 49, `.`: 36, `＝`: 13, `/`: 12, `“`: 12, `”`: 12, `…`: 9, `)`: 5, `美`: 5, `=`: 4, `é`: 4, `复`: 3, `或`: 3, `缩`: 3, ``: 1, `（`: 1

### 脏词示例（前20个）

- `a.m`
- `a.m.`
- `a.m.(=a.m`
- `arise(arose,arisen`
- `ask…for`
- `autumn/fall`
- `awake(awoke,awoken`
- `b/l`
- `backward(s`
- `bad(worse,worst`
- `ballpoint“＝”ballpointpen`
- `beat(beaten`
- `become(became,become`
- `begin(began,begun`
- `bend(bent`
- `between…and`
- `bike“＝”bicycle`
- `bite(bit,bitten`
- `blow(blew,blown`
- `boss/bqs`

### 缺失翻译示例（前20个）

- 无


## v2

- 输入：`data/stage_vocab_v2_clusterv1.json`
- 输出：`data/stage_vocab_clean_v2.json`
- 总词数：19801
- 脏词数：124
- 缺失翻译数：8383
- 缺 translation 字段：0
- 空翻译字符串：8383
- 清洗后词库大小：19677
- 非法字符统计：`(`: 75, `,`: 49, `.`: 36, `＝`: 13, `/`: 12, `“`: 12, `”`: 12, `…`: 9, `)`: 5, `美`: 5, `=`: 4, `é`: 4, `复`: 3, `或`: 3, `缩`: 3, ``: 1, `（`: 1

### 脏词示例（前20个）

- `a.m`
- `a.m.`
- `a.m.(=a.m`
- `arise(arose,arisen`
- `ask…for`
- `autumn/fall`
- `awake(awoke,awoken`
- `b/l`
- `backward(s`
- `bad(worse,worst`
- `ballpoint“＝”ballpointpen`
- `beat(beaten`
- `become(became,become`
- `begin(began,begun`
- `bend(bent`
- `between…and`
- `bike“＝”bicycle`
- `bite(bit,bitten`
- `blow(blew,blown`
- `boss/bqs`

### 缺失翻译示例（前20个）

- `aback`
- `abandoned`
- `abase`
- `abash`
- `abate`
- `abatement`
- `abbreviated`
- `abdicate`
- `abdomen`
- `abdominal`
- `abduct`
- `abduction`
- `aberration`
- `abhor`
- `abhorrent`
- `abiding`
- `abject`
- `ablaze`
- `able-bodied`
- `ablution`
