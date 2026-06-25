#!/usr/bin/env python3
"""Generate publication-style figures from simulation data."""
import json, math, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJ = Path(__file__).resolve().parent.parent
FIGS = PROJ / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Publication style ──
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
})
CB = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
V1_C = "#B0BEC5"
V2_C = "#E76F51"


def load(path):
    with open(PROJ / path) as f:
        return json.load(f)


# ── Fig 1: Question count vs MAE / R² ──
def fig_question_count():
    qe = load("outputs/question_count_exploration.json")
    sums = qe.get("summaries")
    if not sums:
        return
    counts = sorted(int(k) for k in sums)
    mae = [sums[str(c)]["mae"] for c in counts]
    r2 = [sums[str(c)]["r2"] for c in counts]

    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))
    ax2 = ax1.twinx()
    ax1.plot(counts, mae, "o-", color=CB[4], label="MAE", zorder=3)
    ax2.plot(counts, r2, "s--", color=CB[1], label="R²", zorder=3)
    ax1.set_xlabel("Number of Questions")
    ax1.set_ylabel("MAE", color=CB[4])
    ax2.set_ylabel("R²", color=CB[1])
    ax1.tick_params(axis="y", labelcolor=CB[4])
    ax2.tick_params(axis="y", labelcolor=CB[1])
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right")
    ax1.set_xticks(counts)
    fig.tight_layout()
    fig.savefig(FIGS / "question_count_vs_mae.pdf")
    plt.close(fig)
    print("✓ Fig 1: question_count_vs_mae.pdf")


# ── Fig 2: Bucket comparison v1 vs v2 ──
def fig_bucket_compare():
    v1 = load("outputs/simulation_v1.json")["summary"]["bucket_errors"]
    v2 = load("outputs/simulation_v2_clusterv1.json")["summary"]["bucket_errors"]

    labels = [b["bucket"].replace("low_1k_3k", "1k-3k")
                           .replace("mid_3k_8k", "3k-8k")
                           .replace("high_8k_15k", "8k-15k")
              for b in v1]
    mae_v1 = [b["mae"] for b in v1]
    mae_v2 = [b["mae"] for b in v2]
    n_v1 = [b["n"] for b in v1]
    n_v2 = [b["n"] for b in v2]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    b1 = ax.bar(x - w/2, mae_v1, w, label=f"v1 (11,418)", color=V1_C, edgecolor="white")
    b2 = ax.bar(x + w/2, mae_v2, w, label=f"v2 (19,801)", color=V2_C, edgecolor="white")
    for bar, n in zip(b1, n_v1):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, f"n={n}",
                ha="center", va="bottom", fontsize=7, color="#666")
    for bar, n in zip(b2, n_v2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10, f"n={n}",
                ha="center", va="bottom", fontsize=7, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "bucket_comparison.pdf")
    plt.close(fig)
    print("✓ Fig 2: bucket_comparison.pdf")


# ── Fig 3: v2 scatter plot ──
def fig_scatter_v2():
    v2 = load("outputs/simulation_v2_clusterv1.json")
    recs = v2["records"]
    true = np.array([r["true_vocab"] for r in recs], dtype=float)
    est = np.array([r["estimated_vocab"] for r in recs], dtype=float)
    buckets = [r["bucket"] for r in recs]

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    colors = {"low_1k_3k": CB[0], "mid_3k_8k": CB[2], "high_8k_15k": CB[4]}
    labels = {"low_1k_3k": "1k-3k", "mid_3k_8k": "3k-8k", "high_8k_15k": "8k-15k"}
    for b in ["low_1k_3k", "mid_3k_8k", "high_8k_15k"]:
        mask = [bb == b for bb in buckets]
        ax.scatter(true[mask], est[mask], s=4, color=colors[b], alpha=0.5, label=labels[b])
    lims = [0, max(true.max(), est.max()) + 500]
    ax.plot(lims, lims, "--", color="#888", linewidth=0.8, label="y=x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True Vocabulary Size")
    ax.set_ylabel("Estimated (raw sum)")
    ax.legend(markerscale=3, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIGS / "scatter_v2.pdf")
    plt.close(fig)
    print("✓ Fig 3: scatter_v2.pdf")


# ── Fig 4: Calibration curves ──
def fig_calibration():
    v2 = load("outputs/simulation_v2_clusterv1.json")
    recs = v2["records"]
    true = np.array([r["true_vocab"] for r in recs], dtype=float)
    raw = np.array([r["estimated_vocab"] for r in recs], dtype=float)
    point = np.array([r["point_estimate"] for r in recs], dtype=float)

    # Sort by raw for clean lines
    idx = np.argsort(true)
    true, raw, point = true[idx], raw[idx], point[idx]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(true, raw, "-", color=CB[2], linewidth=1.5, label="Raw (identity)")
    ax.plot(true, point, "-", color=V2_C, linewidth=1.5, label="Tanh+piecewise (old)")
    ax.plot([0, 15000], [0, 15000], "--", color="#888", linewidth=0.8, label="y=x")
    ax.set_xlabel("True Vocabulary")
    ax.set_ylabel("Estimated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "calibration_curve.pdf")
    plt.close(fig)
    print("✓ Fig 4: calibration_curve.pdf")


# ── Fig 5: C/F/P/K comparison ──
def fig_cfpk():
    v1_data = {"C": 2663, "F": 2160, "P": 1028, "K": 506}
    v2_data = {"C": 3368, "F": 2457, "P": 1084, "K": 519}
    cov1 = {"C": "86.4%", "F": "89.8%", "P": "83.1%", "K": "91.9%"}
    cov2 = {"C": "93.7%", "F": "94.0%", "P": "86.8%", "K": "93.2%"}
    stages2 = {"C": "Senior H.", "F": "Senior H.", "P": "Grade 8", "K": "Grade 6"}

    labels = list(v1_data.keys())
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    b1 = ax.bar(x - w/2, [v1_data[l] for l in labels], w, label="v1", color=V1_C, edgecolor="white")
    b2 = ax.bar(x + w/2, [v2_data[l] for l in labels], w, label="v2", color=V2_C, edgecolor="white")
    for i, (bar, c) in enumerate(zip(b2, [cov2[l] for l in labels])):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30, c,
                ha="center", va="bottom", fontsize=7, color="#D55E00")
    for i, (bar, s) in enumerate(zip(b2, [stages2[l] for l in labels])):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+80, s,
                ha="center", va="bottom", fontsize=7.5, style="italic", color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Estimated Vocabulary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "cfpk_comparison.pdf")
    plt.close(fig)
    print("✓ Fig 5: cfpk_comparison.pdf")


if __name__ == "__main__":
    fig_question_count()
    fig_bucket_compare()
    fig_scatter_v2()
    fig_calibration()
    fig_cfpk()
    print(f"All figures saved to {FIGS}")
