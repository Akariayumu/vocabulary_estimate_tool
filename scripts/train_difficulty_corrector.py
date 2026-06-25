#!/usr/bin/env python3
"""训练词汇 difficulty 的 MLP residual corrector（方案 B）。

支持在 V100 上进行 GPU（CUDA）训练；不可用时平滑回退到 CPU。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDING_NPY = PROJECT_ROOT / "data" / "word_embeddings_300d.npy"
DEFAULT_EMBEDDING_INDEX = PROJECT_ROOT / "data" / "word_embeddings_index.json"
DEFAULT_STAGE_VOCAB = PROJECT_ROOT / "data" / "stage_vocab.json"
DEFAULT_OUTPUT_MODEL = PROJECT_ROOT / "data" / "difficulty_corrector.pt"
DEFAULT_OUTPUT_VOCAB = PROJECT_ROOT / "data" / "stage_vocab_enhanced.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MLP residual corrector for word difficulty."
    )
    parser.add_argument("--embedding-npy", type=Path, default=DEFAULT_EMBEDDING_NPY)
    parser.add_argument("--embedding-index", type=Path, default=DEFAULT_EMBEDDING_INDEX)
    parser.add_argument("--stage-vocab", type=Path, default=DEFAULT_STAGE_VOCAB)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--output-vocab", type=Path, default=DEFAULT_OUTPUT_VOCAB)
    parser.add_argument("--hidden1", type=int, default=128, help="First hidden layer size.")
    parser.add_argument("--hidden2", type=int, default=32, help="Second hidden layer size.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--epochs", type=int, default=500, help="Max epochs.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="Device override (e.g. 'cuda:0', 'cpu'). Auto-detect if not set.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and print stats, no training.")
    return parser.parse_args(argv)


def _count_syllables(word: str) -> int:
    """基于元音组启发式粗略统计音节数。"""
    word = word.lower().strip()
    if not word:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    return max(1, count)


def _log_norm(value: float, max_val: float) -> float:
    return math.log(value + 1) / math.log(max_val + 1)


def build_features(
    words: list[str],
    embeddings: np.ndarray,
    word_to_idx: dict[str, int],
    difficulties: list[float],
    max_syllables: int = 12,
    max_word_len: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """构建特征矩阵和目标向量。

    每个词的特征：预计算 embeddings（来自 npy/index）+ baseline difficulty。
    """
    n = len(words)
    features_list: list[np.ndarray] = []

    for i, word in enumerate(words):
        w = word.lower().strip()
        if w in word_to_idx:
            emb = embeddings[word_to_idx[w]]
        else:
            n_dims = embeddings.shape[1] if embeddings.ndim > 1 else 300
            emb = np.zeros(n_dims, dtype=np.float32)

        feat = np.concatenate([emb, [difficulties[i]]])
        features_list.append(feat)

    X = np.stack(features_list, axis=0).astype(np.float32)
    y = np.array(difficulties, dtype=np.float32)
    return X, y  # 模型学习 embedding-based prediction 与 baseline 之间的 residual


def train(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    """训练 MLP corrector。

    Returns:
        (model, stats)
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("PyTorch not installed. Install with:")
        print("  pip install torch torchvision torchaudio")
        print("Or run on V100 with the yolov5 conda env.")
        sys.exit(1)

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 划分
    n = len(X)
    indices = np.random.RandomState(args.seed).permutation(n)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train = torch.tensor(X[train_idx]).to(device)
    y_train = torch.tensor(y[train_idx]).to(device)
    X_val = torch.tensor(X[val_idx]).to(device)
    y_val = torch.tensor(y[val_idx]).to(device)
    X_test = torch.tensor(X[test_idx]).to(device)
    y_test = torch.tensor(y[test_idx]).to(device)

    input_dim = X.shape[1]

    # 模型
    class DifficultyCorrector(nn.Module):
        def __init__(self, input_dim: int, h1: int, h2: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, h1),
                nn.BatchNorm1d(h1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h1, h2),
                nn.BatchNorm1d(h2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.66),
                nn.Linear(h2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = DifficultyCorrector(input_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # 训练
    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict | None = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for i in range(0, len(X_train), args.batch_size):
            batch_x = X_train[i:i + args.batch_size]
            batch_y = y_train[i:i + args.batch_size]
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            train_loss = criterion(model(X_train), y_train).item()

        scheduler.step(val_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0

    # 恢复最佳模型
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # 测试
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test).item()
        test_mae = nn.L1Loss()(test_pred, y_test).item()

        # 计算 residuals
        residuals = (test_pred - y_test).cpu().numpy()
        residual_std = float(np.std(residuals))
        residual_mean = float(np.mean(residuals))

    stats = {
        "train_loss": round(train_loss, 6),
        "val_loss": round(best_val_loss, 6),
        "test_loss": round(test_loss, 6),
        "test_mae": round(test_mae, 6),
        "residual_mean": round(residual_mean, 6),
        "residual_std": round(residual_std, 6),
        "epochs_trained": epoch,
        "training_seconds": round(elapsed, 1),
        "device": str(device),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "input_dim": input_dim,
        "params": sum(p.numel() for p in model.parameters()),
    }

    print(f"\n=== Training complete ===")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Test MSE:      {test_loss:.6f}")
    print(f"  Test MAE:      {test_mae:.6f}")
    print(f"  Residual std:  {residual_std:.4f}")
    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Device:        {device}")

    return model, stats


def apply_correction(
    model: Any,
    X_all: np.ndarray,
    difficulties: list[float],
    args: argparse.Namespace,
) -> np.ndarray:
    """对所有词应用 MLP residual correction。"""
    import torch
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_all).to(device)
        residuals = model(X_t).cpu().numpy()

    corrected = np.array(difficulties, dtype=np.float32) + residuals.flatten()
    corrected = np.clip(corrected, 0.001, 0.999)
    return corrected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 加载数据
    print("Loading data ...")
    with open(args.stage_vocab, encoding="utf-8") as f:
        sv_data = json.load(f)

    word_to_stage: dict = sv_data["word_to_stage"]
    words = list(word_to_stage.keys())
    difficulties = [word_to_stage[w]["difficulty"] for w in words]

    # 加载 embeddings
    embeddings = np.load(args.embedding_npy)
    with open(args.embedding_index) as f:
        word_to_idx = json.load(f)

    n_found = sum(1 for w in words if w.lower().strip() in word_to_idx)
    print(f"  Words: {len(words)} (embedding match: {n_found})")
    print(f"  Embedding matrix: {embeddings.shape}")

    # 构建 features
    X, y = build_features(words, embeddings, word_to_idx, difficulties)

    if args.dry_run:
        print(f"\n=== Dry run ===")
        print(f"  Feature matrix: {X.shape}")
        print(f"  Feature dim: {X.shape[1]} (300 emb + word_len + syllables + difficulty)")
        print(f"  Target: {y.shape}")
        print(f"  Difficulty range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"  Difficulty mean: {y.mean():.4f}  std: {y.std():.4f}")
        return 0

    # 训练
    model, stats = train(X, y, args)

    # 保存模型
    import torch
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "dropout": args.dropout,
        "stats": stats,
    }, args.output_model)
    print(f"Saved model: {args.output_model}")

    # 应用 correction 并保存 enhanced vocab
    corrected = apply_correction(model, X, difficulties, args)

    corrections_made = int(np.sum(np.abs(corrected - np.array(difficulties)) > 0.01))
    print(f"Words with >0.01 correction: {corrections_made} / {len(words)}")

    for i, w in enumerate(words):
        word_to_stage[w]["difficulty"] = round(float(corrected[i]), 4)
        word_to_stage[w]["difficulty_source"] = "enhanced_mlp"

    sv_data["meta"]["difficulty_enhancement"] = {
        "method": "mlp_residual_correction",
        "model_type": "DifficultyCorrector",
        "features": ["glove_300d", "word_length", "syllable_count", "baseline_difficulty"],
        "training_stats": stats,
    }

    with open(args.output_vocab, "w", encoding="utf-8") as f:
        json.dump(sv_data, f, ensure_ascii=False, indent=2)
    print(f"Saved enhanced vocab: {args.output_vocab}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
