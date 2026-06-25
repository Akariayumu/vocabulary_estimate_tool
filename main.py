"""词汇量估算器的命令行入口。

输入 JSON 示例：

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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vocab_estimator.config import DEFAULT_CONFIG
from vocab_estimator.coverage import DocumentCoverageAnalyzer
from vocab_estimator.vocab_bank import VocabBank
from vocab_estimator.vocab_model import VocabEstimator


def normalize_responses(raw: Any) -> dict[str, list[tuple[str, bool]]]:
    """校验并归一化 response JSON 为 ``(word, known)`` 对。"""

    if not isinstance(raw, dict):
        raise ValueError("responses must be a dict: {class: [[word, known_bool], ...]}")

    output: dict[str, list[tuple[str, bool]]] = {}
    for class_name, records in raw.items():
        if not isinstance(records, list):
            raise ValueError(f"responses[{class_name!r}] must be a list")
        normalized = []
        for idx, record in enumerate(records):
            if not isinstance(record, (list, tuple)) or len(record) < 2:
                raise ValueError(f"responses[{class_name!r}][{idx}] must be [word, known_bool]")
            word, known = record[0], record[1]
            if not isinstance(word, str):
                raise ValueError(f"word at responses[{class_name!r}][{idx}] must be a string")
            normalized.append((word, bool(known)))
        output[str(class_name)] = normalized
    return output


def normalize_documents(raw: Any) -> dict[str, list[str]]:
    """将文档路径 JSON 归一化为 ``{class: [paths...]}``。"""

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("documents must be a dict: {class: path_or_paths}")

    output: dict[str, list[str]] = {}
    for class_name, value in raw.items():
        if isinstance(value, str):
            output[str(class_name)] = [value]
        elif isinstance(value, list):
            output[str(class_name)] = [str(path) for path in value]
        else:
            raise ValueError(f"documents[{class_name!r}] must be a string or list")
    return output


def run_estimation(payload: dict[str, Any]) -> dict:
    """为所有学习者类别运行完整 hybrid model。"""

    responses = normalize_responses(payload.get("responses", payload))
    documents = normalize_documents(payload.get("documents"))

    vocab_bank = VocabBank(DEFAULT_CONFIG)
    estimator = VocabEstimator(vocab_bank, DEFAULT_CONFIG)
    coverage = DocumentCoverageAnalyzer(vocab_bank)

    estimated = estimator.estimate_groups(responses)
    classes = estimated["classes"]

    for class_name, paths in documents.items():
        if class_name not in classes:
            continue
        classes[class_name]["document_coverage"] = coverage.analyze_paths(paths)

    for class_name in classes:
        classes[class_name].setdefault("document_coverage", None)
        classes[class_name]["vocab_bank_size"] = len(vocab_bank)
        classes[class_name]["vocab_bank_used_fallback"] = vocab_bank.used_fallback

    return {
        "classes": classes,
        "ordering_consistency": estimated["ordering_consistency"],
        "config": {
            "bucket_boundaries": DEFAULT_CONFIG.bucket_boundaries,
            "levels": DEFAULT_CONFIG.levels,
            "bootstrap_iterations": DEFAULT_CONFIG.bootstrap_iterations,
        },
    }


def load_payload(path: str | Path) -> dict[str, Any]:
    """读取输入 JSON 文件。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate English vocabulary size from test responses.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="JSON file containing responses and optional documents mapping.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional output JSON path. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    result = run_estimation(load_payload(args.input))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
