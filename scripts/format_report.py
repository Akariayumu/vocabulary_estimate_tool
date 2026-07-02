#!/usr/bin/env python3
"""将 course_design_report.md 转换为标准 Markdown 展示版。"""
import re

src = "docs/course_design_report.md"
dst = "docs/course_design_report_presentation.md"

with open(src, "r", encoding="utf-8") as f:
    raw = f.read()

# 1. 替换图片路径 figures/ -> docs/figures/
raw = re.sub(r'\(figures/(.+?)\)', r'(docs/figures/\1)', raw)

# 2. 合并连续 3 行以上空白为 2 行
raw = re.sub(r'\n{4,}', '\n\n\n', raw)

# 3. 表格对齐标记规范化: 确保表头分隔行有对齐标记
#    --- -> | --- | 过渡，但标准 markdown 允许裸 ---
#    确保表头下一行正确: | --- | ---: | 等
lines = raw.split('\n')
out_lines = []
table_header_fix = False

for i, line in enumerate(lines):
    # 检测表格分隔行: 只包含 | - : 空格
    if re.match(r'^\|[\s\-:|]+\|$', line) and not re.search(r'[a-zA-Z0-9\u4e00-\u9fff]', line):
        # 已经在分隔位置有 : 就不用动
        # 确保每个单元格有 --- 或 :--- 或 ---:
        def fix_sep_cell(m):
            content = m.group(1).strip()
            if content and ':' not in content and '---' not in content:
                return '| --- '
            return f'| {content} '
        line = re.sub(r'\|([^|]+)', fix_sep_cell, line)
        out_lines.append(line)
        continue
    # 表格行，如果上一行是分隔行，确保没有多余空格
    out_lines.append(line)

raw = '\n'.join(out_lines)

# 4. 标准 markdown 标题确保 # 后有空格
raw = re.sub(r'^(#+)([^#\s])', r'\1 \2', raw, flags=re.MULTILINE)

# 5. 修复封面双标题：保持原始格式
# 原封面已有正确格式

with open(dst, "w", encoding="utf-8") as f:
    f.write(raw)

print(f"✅ 已输出到 {dst}")
print(f"总行数: {len(raw.split(chr(10)))}")
