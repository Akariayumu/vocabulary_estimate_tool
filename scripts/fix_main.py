#!/usr/bin/env python3
"""Fix the corrupted server/main.py."""
import sys
sys.path.insert(0, '.')
with open('server/main.py', 'r') as f:
    content = f.read()

# Remove the misplaced get_bucket_fn
import re
# Remove the misplaced block (inserted at wrong indent level)
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

# Find where to insert get_bucket_fn (after get_estimator, before get_coverage_analyzer)
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
