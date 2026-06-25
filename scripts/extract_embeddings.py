#!/usr/bin/env python3
"""Extract GloVe 300d embeddings for all words in stage_vocab.json."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOVE_URL = (
    "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip"
)
DEFAULT_GLOVE_PATH = PROJECT_ROOT / "data" / "glove.840B.300d.txt"
DEFAULT_STAGE_VOCAB = PROJECT_ROOT / "data" / "stage_vocab.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract GloVe embeddings for stage_vocab words."
    )
    parser.add_argument(
        "--glove-path",
        type=Path,
        default=DEFAULT_GLOVE_PATH,
        help="Path to glove.840B.300d.txt (downloads if missing).",
    )
    parser.add_argument(
        "--stage-vocab",
        type=Path,
        default=DEFAULT_STAGE_VOCAB,
        help="Path to stage_vocab.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for .npy and index files.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=300,
        help="Embedding dimension (default 300).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download GloVe even if file exists.",
    )
    return parser.parse_args(argv)


def download_glove(glove_path: Path, force: bool = False) -> None:
    """Download and extract GloVe 840B 300d if not present."""
    if glove_path.exists() and not force:
        print(f"GloVe file exists at {glove_path}, skipping download.")
        return

    zip_path = glove_path.with_suffix(".zip")
    glove_path.parent.mkdir(parents=True, exist_ok=True)

    import urllib.request
    import zipfile

    url = DEFAULT_GLOVE_URL
    print(f"Downloading GloVe from {url} ...")
    t0 = time.time()
    urllib.request.urlretrieve(url, zip_path)
    print(f"Downloaded {zip_path} ({time.time() - t0:.0f}s)")

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(glove_path.parent)
    print(f"Extracted to {glove_path}")

    # Clean up zip
    zip_path.unlink()


def load_glove_vocab(glove_path: Path, dim: int = 300) -> dict[str, int]:
    """Return {word: index} from the GloVe file (reading header only)."""
    vocab: dict[str, int] = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            word = line.split(" ", 1)[0]
            vocab[word] = idx
    return vocab


def extract_embeddings(
    glove_path: Path,
    words: list[str],
    dim: int = 300,
) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """Extract embeddings for the given word list from GloVe file.

    Returns:
        (embedding_matrix, word_to_idx, oov_words)
    """
    print(f"Loading GloVe from {glove_path} ...")
    t0 = time.time()

    # Build GloVe embedding dict
    glove_emb: dict[str, np.ndarray] = {}
    line_count = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            word = parts[0]
            vec = np.array([float(v) for v in parts[1:]], dtype=np.float32)
            if len(vec) == dim:
                glove_emb[word] = vec
            line_count += 1

    print(f"Loaded {len(glove_emb)} embeddings from {line_count} lines ({time.time() - t0:.0f}s)")

    # Match our words
    word_to_idx: dict[str, int] = {}
    embeddings: list[np.ndarray] = []
    oov_words: list[str] = []
    oov_count = 0

    for word in words:
        w = word.lower().strip()
        if w in glove_emb:
            idx = len(embeddings)
            word_to_idx[word] = idx
            embeddings.append(glove_emb[w])
        else:
            oov_words.append(word)
            oov_count += 1

    matrix = np.stack(embeddings, axis=0) if embeddings else np.zeros((0, dim), dtype=np.float32)
    oov_rate = oov_count / len(words) * 100 if words else 0

    print(f"Matched {len(embeddings)} / {len(words)} words (OOV rate: {oov_rate:.1f}%)")
    print(f"Embedding matrix shape: {matrix.shape}")

    return matrix, word_to_idx, oov_words


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download if needed
    download_glove(args.glove_path, force=args.force_download)

    if not args.glove_path.exists():
        print(f"Error: GloVe file not found at {args.glove_path}", file=sys.stderr)
        return 1

    # Load stage vocab words
    with open(args.stage_vocab, encoding="utf-8") as f:
        data = json.load(f)

    word_to_stage: dict = data["word_to_stage"]
    words = list(word_to_stage.keys())
    print(f"Stage vocab: {len(words)} words")

    # Extract embeddings
    matrix, word_to_idx, oov = extract_embeddings(args.glove_path, words, dim=args.dim)

    # Save outputs
    npy_path = args.output_dir / "word_embeddings_300d.npy"
    index_path = args.output_dir / "word_embeddings_index.json"
    oov_path = args.output_dir / "word_embeddings_oov.json"

    np.save(npy_path, matrix)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(word_to_idx, f, ensure_ascii=False, indent=2)
    with open(oov_path, "w", encoding="utf-8") as f:
        json.dump(oov, f, ensure_ascii=False, indent=2)

    print(f"Saved embedding matrix: {npy_path} ({matrix.nbytes / 1024 / 1024:.1f} MB)")
    print(f"Saved word index: {index_path}")
    print(f"Saved OOV list: {oov_path}")

    if oov:
        print(f"\nOOV words (first 20): {oov[:20]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
