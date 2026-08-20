"""Render the public experiment figures from the committed metrics snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "final_metrics.json"
OUTPUT_DIR = ROOT / "docs" / "assets"

INK = "#17212B"
MUTED = "#64707D"
GRID = "#DCE2E7"
MEDICAL = "#D1495B"
GENERAL = "#00798C"
ACCENT = "#EDAE49"
BASE = "#66717E"
PAPER = "#FFFFFF"


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "figure.titlesize": 20,
            "figure.titleweight": "bold",
            "legend.frameon": False,
            "savefig.facecolor": PAPER,
            "savefig.bbox": "tight",
        }
    )


def polish_axis(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def annotate_bars(ax, bars, suffix: str = "%") -> None:
    for bar in bars:
        value = bar.get_height()
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )


def save(fig, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close(fig)


def plot_overview(data: dict) -> None:
    opd = data["medical_opd_diagnostic"]["rows"]
    sar = data["sar_diagnostic"]["rows"]
    selected = [opd[0], opd[-1], sar[2], sar[-1]]
    labels = ["4B Base", "Medical\nOPD@300", "SAR@100", "SAR@300"]
    medical = [row["medical_accuracy"] for row in selected]
    general = [row["general_accuracy"] for row in selected]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(13.5, 7), constrained_layout=True)
    fig.suptitle("Capability trade-off across the staged OPD pipeline", x=0.04, ha="left")
    ax.text(
        0.0,
        1.025,
        "Fixed MedQA-zh600 / C-Eval300 diagnostic. "
        "SAR@100 is the best observed balance, not the final step.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    med_bars = ax.bar(x - width / 2, medical, width, color=MEDICAL, label="Medical accuracy")
    gen_bars = ax.bar(x + width / 2, general, width, color=GENERAL, label="General accuracy")
    ax.axhline(medical[0], color=MEDICAL, linestyle="--", linewidth=1, alpha=0.55)
    ax.axhline(general[0], color=GENERAL, linestyle="--", linewidth=1, alpha=0.55)
    ax.set_ylim(60, 90)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, labels)
    ax.legend(loc="upper center", ncol=2)
    polish_axis(ax)
    annotate_bars(ax, med_bars)
    annotate_bars(ax, gen_bars)
    ax.annotate(
        "selected checkpoint",
        xy=(2, max(medical[2], general[2]) + 1.4),
        xytext=(2.45, 88.2),
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.6},
        color=INK,
        fontsize=10,
        fontweight="bold",
    )
    save(fig, "01_pipeline_overview.png")


def plot_raw_27b_diagnosis(data: dict) -> None:
    full_rows = data["full_official_test"]["rows"][:3]
    teacher_rows = data["teacher_adaptation_screen"]["rows"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), constrained_layout=True)
    fig.suptitle("Why a larger teacher was not enough", x=0.035, ha="left")

    labels = [row["checkpoint"].replace(" ", "\n", 1) for row in full_rows]
    x = np.arange(len(labels))
    width = 0.34
    med = [row["medical_accuracy"] for row in full_rows]
    gen = [row["general_accuracy"] for row in full_rows]
    med_bars = axes[0].bar(x - width / 2, med, width, color=MEDICAL, label="MedQA test")
    gen_bars = axes[0].bar(x + width / 2, gen, width, color=GENERAL, label="C-Eval-8 test")
    axes[0].set_title("Raw 27B scoring did not create a robust medical gain", loc="left")
    axes[0].set_ylim(65, 90)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(x, labels)
    axes[0].legend(loc="lower left")
    polish_axis(axes[0])
    annotate_bars(axes[0], med_bars)
    annotate_bars(axes[0], gen_bars)
    axes[0].text(
        0.02,
        0.97,
        "Raw-27B OPD@200 MedQA: 68.42%",
        transform=axes[0].transAxes,
        va="top",
        color=MUTED,
        fontsize=10,
    )

    labels = ["Raw 27B", "27B Medical\nSFT@25"]
    med = [row["medical_accuracy"] for row in teacher_rows]
    valid = [row["format_valid_rate"] for row in teacher_rows]
    x = np.arange(2)
    med_bars = axes[1].bar(x - width / 2, med, width, color=MEDICAL, label="Medical accuracy")
    valid_bars = axes[1].bar(x + width / 2, valid, width, color=ACCENT, label="Valid final answer")
    axes[1].set_title("The 100-example 27B SFT update reduced answer reliability", loc="left")
    axes[1].set_ylim(70, 105)
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_xticks(x, labels)
    axes[1].legend(loc="lower left")
    polish_axis(axes[1])
    annotate_bars(axes[1], med_bars)
    annotate_bars(axes[1], valid_bars)
    save(fig, "02_raw_27b_diagnosis.png")


def plot_training_curves(data: dict) -> None:
    opd = data["medical_opd_diagnostic"]["rows"]
    sar = data["sar_diagnostic"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), constrained_layout=True, sharey=True)
    fig.suptitle(
        "More optimization steps did not monotonically improve capability",
        x=0.035,
        ha="left",
    )

    for ax, rows, title in (
        (axes[0], opd, "Medical OPD trajectory"),
        (axes[1], sar, "Base-anchor SAR trajectory"),
    ):
        steps = [row["step"] for row in rows]
        medical = [row["medical_accuracy"] for row in rows]
        general = [
            np.nan if row["general_accuracy"] is None else row["general_accuracy"]
            for row in rows
        ]
        ax.plot(steps, medical, color=MEDICAL, marker="o", linewidth=2.4, label="Medical")
        ax.plot(steps, general, color=GENERAL, marker="o", linewidth=2.4, label="General")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Optimizer steps")
        ax.set_xticks(steps)
        ax.set_ylim(65, 88)
        polish_axis(ax)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="lower left", ncol=2)
    axes[0].scatter([100], [82.5], s=145, facecolors="none", edgecolors=ACCENT, linewidths=2.3)
    axes[1].scatter([100], [85.0], s=145, facecolors="none", edgecolors=ACCENT, linewidths=2.3)
    axes[1].annotate(
        "best observed balance",
        xy=(100, 85),
        xytext=(145, 86.3),
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.5},
        fontsize=9,
        fontweight="bold",
    )
    save(fig, "03_training_curves.png")


def plot_termination_mechanism(data: dict) -> None:
    sar = data["sar_diagnostic"]["rows"]
    chosen = [sar[0], sar[2], sar[-1]]
    labels = ["Medical\nOPD@300", "SAR@100", "SAR@300"]
    accuracy = [row["medical_accuracy"] for row in chosen]
    truncation = [row["medical_truncation"] for row in chosen]
    lengths = [row["medical_mean_output_tokens"] for row in chosen]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), constrained_layout=True)
    fig.suptitle("The late-stage regression was a termination failure", x=0.035, ha="left")
    x = np.arange(len(labels))
    width = 0.34
    acc_bars = axes[0].bar(x - width / 2, accuracy, width, color=MEDICAL, label="Medical accuracy")
    trunc_bars = axes[0].bar(x + width / 2, truncation, width, color=BASE, label="Truncation rate")
    axes[0].set_title("Accuracy fell as capped responses stopped finishing", loc="left")
    axes[0].set_ylim(0, 92)
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticks(x, labels)
    axes[0].legend(loc="upper left")
    polish_axis(axes[0])
    annotate_bars(axes[0], acc_bars)
    annotate_bars(axes[0], trunc_bars)

    length_bars = axes[1].bar(x, lengths, width=0.55, color=[GENERAL, ACCENT, BASE])
    axes[1].set_title("Medical responses became progressively longer", loc="left")
    axes[1].set_ylim(0, 930)
    axes[1].set_ylabel("Mean output tokens")
    axes[1].set_xticks(x, labels)
    polish_axis(axes[1])
    annotate_bars(axes[1], length_bars, suffix="")
    axes[1].text(
        0.02,
        0.96,
        "Longer Base-like reasoning exceeded the 1,024-token medical cap.",
        transform=axes[1].transAxes,
        va="top",
        color=MUTED,
        fontsize=10,
    )
    save(fig, "04_termination_mechanism.png")


def main() -> None:
    configure_style()
    data = load_metrics()
    plot_overview(data)
    plot_raw_27b_diagnosis(data)
    plot_training_curves(data)
    plot_termination_mechanism(data)
    print(f"Rendered figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
