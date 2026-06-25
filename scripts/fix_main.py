#!/usr/bin/env python3
"""修复损坏的 server/main.py。"""
import sys
sys.path.insert(0, '.')
with open('server/main.py', 'r') as f:
    content = f.read()

# 移除位置错误的 get_bucket_fn
import re
# 移除位置错误的代码块（插入到了错误缩进层级）
old = """@lru_cache(maxsize=1)
def get_coverage_analyzer() -> DocumentCoverageAnalyzer:


def get_bucket_fn():
    \"\"\"Return bucket-name function for bucket matrix model.\"\"\"
    return get_vocab_bank().get_bucket

    return DocumentCoverageAnalyzer(get_vocab_bank(), config=DEFAULT_CONFIG)"""

new = """@lru_cache(maxsize=1)
def get_coverage_analyzer() -> DocumentCoverageAnalyzer:
    return DocumentCoverageAnalyzer(get_vocab_bank(), config=DEFAULT_CONFIG)"""

content = content.replace(old, new)

# 查找 get_bucket_fn 插入位置（get_estimator 之后，get_coverage_analyzer 之前）
insertion_point = """@lru_cache(maxsize=1)
def get_estimator() -> VocabEstimator:
    return VocabEstimator(get_vocab_bank(), DEFAULT_CONFIG)"""

replacement = insertion_point + """


def get_bucket_fn():
    \"\"\"Return bucket-name function for bucket matrix model.\"\"\"
    return get_vocab_bank().get_bucket"""
content = content.replace(insertion_point, replacement)

with open('server/main.py', 'w') as f:
    f.write(content)
print('Fixed.')
