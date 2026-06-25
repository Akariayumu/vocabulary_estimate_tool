#!/usr/bin/env python3
"""Generate publication-style PDF figures from project outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

CJK_SERIF_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
CJK_SERIF_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
for font_path in (CJK_SERIF_REGULAR, CJK_SERIF_BOLD):
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
CJK_SERIF_NAME = (
    font_manager.FontProperties(fname=str(CJK_SERIF_REGULAR)).get_name()
    if CJK_SERIF_REGULAR.exists()
    else "Noto Serif CJK SC"
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            CJK_SERIF_NAME,
            "Noto Serif CJK SC",
            "Noto Serif CJK JP",
            "Times New Roman",
            "Noto Serif",
            "DejaVu Serif",
        ],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.unicode_minus": False,
    }
)

# Okabe-Ito / Tol-style colorblind-safe colors.
COLORS = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "gray": "#7F7F7F",
    "light_blue": "#A6CEE3",
}
V1_COLOR = COLORS["light_blue"]
V2_COLOR = COLORS["blue"]


def load_json(relative_path: str) -> dict:
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as f:
        return json.load(f)


def thousands(value: float, _position: int) -> str:
    return f"{int(value):,}"


def save_pdf(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"saved {FIGURE_DIR / filename}")


def bucket_rows(summary: dict) -> dict[str, dict]:
    return {row["bucket"]: row for row in summary["bucket_errors"]}


def parse_cfpk_from_report() -> dict[str, dict[str, dict[str, object]]]:
    report_path = PROJECT_ROOT / "docs" / "course_design_report.md"
    report = report_path.read_text(encoding="utf-8")
    parsed: dict[str, dict[str, dict[str, object]]] = {"v1": {}, "v2": {}}
    current: str | None = None

    for line in report.splitlines():
        if "v1 原始词库结果如下" in line or (
            "v1 清洗词库" in line and "结果如下" in line
        ):
            current = "v1"
            continue
        if "v2 统一标定词库结果如下" in line:
            current = "v2"
            continue
        if current and line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 3 and cells[0] in {"C", "F", "P", "K"}:
                parsed[current][cells[0]] = {
                    "estimated_vocab": int(cells[1].replace(",", "")),
                    "stage": cells[2],
                }
        elif current and not line.strip() and parsed[current]:
            current = None

    missing = {
        version: sorted(set("CFPK") - set(rows))
        for version, rows in parsed.items()
        if set(rows) != set("CFPK")
    }
    if missing:
        raise ValueError(f"Could not parse C/F/P/K rows from {report_path}: {missing}")
    return parsed


def piecewise_calibrate(estimate: float) -> float:
    if estimate <= 0:
        return estimate

    knots = ((3000.0, 1.00), (8000.0, 0.45), (22000.0, 1.28))
    previous_boundary = 0.0
    previous_value = 0.0
    for boundary, slope in knots:
        if estimate <= boundary:
            return previous_value + (estimate - previous_boundary) * slope
        previous_value += (boundary - previous_boundary) * slope
        previous_boundary = boundary
    return previous_value + (estimate - previous_boundary) * knots[-1][1]


def old_tanh_piecewise(raw_sum: np.ndarray, scale: float = 0.8) -> np.ndarray:
    tanh_stage = 20000.0 * np.tanh(0.0000691 * raw_sum * scale)
    return np.array([piecewise_calibrate(float(value)) for value in tanh_stage])


def plot_question_count_vs_mae() -> None:
    data = load_json("outputs/question_count_exploration.json")
    summaries = data["summaries"]
    counts = [count for count in sorted(int(key) for key in summaries) if 10 <= count <= 40]
    mae = [summaries[str(count)]["mae"] for count in counts]
    r2 = [summaries[str(count)]["r2"] for count in counts]

    fig, ax_mae = plt.subplots(figsize=(3.8, 2.75))
    ax_r2 = ax_mae.twinx()

    ax_mae.plot(counts, mae, color=COLORS["blue"], marker="o", label="MAE", zorder=3)
    ax_r2.plot(
        counts,
        r2,
        color=COLORS["vermillion"],
        marker="^",
        linestyle="--",
        label="R²",
        zorder=3,
    )

    ax_mae.set_xlabel("Number of Questions")
    ax_mae.set_ylabel("MAE", color=COLORS["blue"])
    ax_r2.set_ylabel("R²", color=COLORS["vermillion"])
    ax_mae.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax_r2.tick_params(axis="y", labelcolor=COLORS["vermillion"])
    ax_mae.set_xlim(9, 41)
    ax_mae.set_xticks(counts)
    ax_mae.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax_r2.set_ylim(max(0.0, min(r2) - 0.03), min(1.0, max(r2) + 0.03))

    lines_mae, labels_mae = ax_mae.get_legend_handles_labels()
    lines_r2, labels_r2 = ax_r2.get_legend_handles_labels()
    ax_mae.legend(lines_mae + lines_r2, labels_mae + labels_r2, loc="best")

    fig.tight_layout()
    save_pdf(fig, "question_count_vs_mae.pdf")


def plot_bucket_comparison() -> None:
    v1 = bucket_rows(load_json("outputs/simulation_v1.json")["summary"])
    v2 = bucket_rows(load_json("outputs/simulation_v2_clusterv1.json")["summary"])
    buckets = ["low_1k_3k", "mid_3k_8k", "high_8k_15k"]
    labels = ["low\n(1k-3k)", "mid\n(3k-8k)", "high\n(8k-15k)"]
    mae_v1 = [v1[bucket]["mae"] for bucket in buckets]
    mae_v2 = [v2[bucket]["mae"] for bucket in buckets]
    n_v1 = [v1[bucket]["n"] for bucket in buckets]
    n_v2 = [v2[bucket]["n"] for bucket in buckets]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(4.05, 2.9))
    bars_v1 = ax.bar(
        x - width / 2,
        mae_v1,
        width,
        label="v1",
        color=V1_COLOR,
        edgecolor="white",
        linewidth=0.6,
    )
    bars_v2 = ax.bar(
        x + width / 2,
        mae_v2,
        width,
        label="v2",
        color=V2_COLOR,
        edgecolor="white",
        linewidth=0.6,
    )

    for bars, ns in ((bars_v1, n_v1), (bars_v2, n_v2)):
        for bar, n in zip(bars, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 18,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#444444",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE")
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_ylim(0, max(mae_v1 + mae_v2) * 1.18)
    ax.legend(loc="best")

    fig.tight_layout()
    save_pdf(fig, "bucket_comparison.pdf")


def plot_scatter_v2() -> None:
    records = load_json("outputs/simulation_v2_clusterv1.json")["records"]
    true = np.array([record["true_vocab"] for record in records], dtype=float)
    estimate = np.array([record["estimated_vocab"] for record in records], dtype=float)
    buckets = np.array([record["bucket"] for record in records])

    fig, ax = plt.subplots(figsize=(3.6, 3.35))
    bucket_colors = {
        "low_1k_3k": COLORS["orange"],
        "mid_3k_8k": COLORS["green"],
        "high_8k_15k": COLORS["blue"],
    }
    bucket_labels = {
        "low_1k_3k": "low (1k-3k)",
        "mid_3k_8k": "mid (3k-8k)",
        "high_8k_15k": "high (8k-15k)",
    }
    for bucket in ["low_1k_3k", "mid_3k_8k", "high_8k_15k"]:
        mask = buckets == bucket
        ax.scatter(
            true[mask],
            estimate[mask],
            s=6,
            color=bucket_colors[bucket],
            alpha=0.45,
            edgecolors="none",
            label=bucket_labels[bucket],
            rasterized=True,
        )

    upper = int(math.ceil(max(float(true.max()), float(estimate.max())) / 1000.0) * 1000)
    limits = [0, upper]
    ax.plot(limits, limits, "--", color="#666666", linewidth=0.9, label="Ideal y=x")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("True vocabulary size")
    ax.set_ylabel("Estimated vocabulary size")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.legend(markerscale=2.6, loc="upper left")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    save_pdf(fig, "scatter_v2.pdf")


def plot_calibration_curve() -> None:
    records = load_json("outputs/simulation_v2_clusterv1.json")["records"]
    raw_sum = np.array([record["estimated_vocab"] for record in records], dtype=float)
    true_vocab = np.array([record["true_vocab"] for record in records], dtype=float)

    order = np.argsort(raw_sum)
    ordered_indices = order.tolist()
    sample_indices = ordered_indices[:: max(1, len(ordered_indices) // 160)]
    raw_sample = raw_sum[sample_indices]
    true_sample = true_vocab[sample_indices]

    chunks = [chunk for chunk in np.array_split(order, 18) if len(chunk)]
    ideal_x = np.array([float(np.mean(raw_sum[chunk])) for chunk in chunks])
    ideal_y = np.array([float(np.mean(true_vocab[chunk])) for chunk in chunks])

    upper = int(math.ceil(max(float(raw_sum.max()), float(true_vocab.max())) / 1000.0) * 1000)
    grid = np.linspace(0, upper, 300)

    fig, ax = plt.subplots(figsize=(4.05, 2.9))
    ax.scatter(
        raw_sample,
        true_sample,
        s=8,
        color="#B0B0B0",
        alpha=0.42,
        edgecolors="none",
        label="v2 sampled records",
        zorder=1,
        rasterized=True,
    )
    ax.plot(grid, grid, color=COLORS["green"], linestyle="-", label="Identity")
    ax.plot(
        grid,
        old_tanh_piecewise(grid),
        color=COLORS["vermillion"],
        linestyle="-.",
        label="Old tanh+piecewise",
    )
    ax.plot(
        ideal_x,
        ideal_y,
        color="#222222",
        linestyle="--",
        marker="o",
        markersize=3.5,
        label="Ideal target",
    )

    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Raw sum")
    ax.set_ylabel("Calibrated estimate")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.legend(loc="best")

    fig.tight_layout()
    save_pdf(fig, "calibration_curve.pdf")


def plot_cfpk_comparison() -> None:
    cfpk = parse_cfpk_from_report()
    labels = list("CFPK")
    v1_values = [cfpk["v1"][label]["estimated_vocab"] for label in labels]
    v2_values = [cfpk["v2"][label]["estimated_vocab"] for label in labels]
    v1_stages = [str(cfpk["v1"][label]["stage"]) for label in labels]
    v2_stages = [str(cfpk["v2"][label]["stage"]) for label in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(4.2, 2.95))
    bars_v1 = ax.bar(
        x - width / 2,
        v1_values,
        width,
        label="v1",
        color=V1_COLOR,
        edgecolor="white",
        linewidth=0.6,
    )
    bars_v2 = ax.bar(
        x + width / 2,
        v2_values,
        width,
        label="v2",
        color=V2_COLOR,
        edgecolor="white",
        linewidth=0.6,
    )

    for bars, stages in ((bars_v1, v1_stages), (bars_v2, v2_stages)):
        for bar, stage in zip(bars, stages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 55,
                stage,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#444444",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Estimated vocab")
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_ylim(0, max(v1_values + v2_values) * 1.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cfpk_comparison.png", dpi=300)
    save_pdf(fig, "cfpk_comparison.pdf")


def main() -> None:
    plot_question_count_vs_mae()
    plot_bucket_comparison()
    plot_scatter_v2()
    plot_calibration_curve()
    plot_cfpk_comparison()
    print(f"All figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
