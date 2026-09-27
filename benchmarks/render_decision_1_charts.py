"""Render the Decision-1 review figures from checked-in benchmark reports.

The images are presentation artifacts. The reports and their row-level evidence
remain the source of truth. Run from the repository root:

    pip install -e '.[charts]'
    python benchmarks/render_decision_1_charts.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/decision-1"
RAW = "#64748b"
TUNED = "#d14d39"
PRIOR = "#a78bfa"
INK = "#172233"
GRID = "#dbe1e8"


def checked_report(raw_name: str, evidence_name: str) -> dict:
    raw = json.loads((ROOT / "results/raw" / raw_name).read_text())
    report_path = ROOT / "results/worthify" / evidence_name / "report.json"
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != raw["report_sha256"]:
        raise ValueError(f"Evidence report hash mismatch: {report_path}")
    report = json.loads(report_path.read_text())
    if report["test_rows"] != raw["test_rows"]:
        raise ValueError(f"Test-row count mismatch: {report_path}")
    return report


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": GRID,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def transfer_chart(banking: dict, ctu: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.15), layout="constrained")
    fig.suptitle("A tuned base helped one LoRA task and hurt another", fontsize=17,
                 fontweight="bold", ha="left", x=0.03)
    cases = [
        (axes[0], banking, "BANKING77 · 16-option routing", "Test accuracy", (0.86, 0.94),
         "+1.20 percentage points", "+1.20 pp"),
        (axes[1], ctu, "CTU-13 · one sealed scenario", "Test macro-F1", (0.0, 0.5),
         "−0.316 macro-F1", "−0.316 F1"),
    ]
    for ax, report, title, xlabel, xlim, annotation, _ in cases:
        raw_name = report["selected_names"]["raw"]
        tuned_name = report["selected_names"]["tuned"]
        metric = "accuracy" if report is banking else "macro_f1"
        raw = report["runs"][raw_name]["metrics"][metric]
        tuned = report["runs"][tuned_name]["metrics"][metric]
        ax.plot([raw, tuned], [0, 0], color=GRID, lw=4, zorder=1)
        ax.scatter([raw], [0], s=175, color=RAW, zorder=3, label="LoRA on original Gemma")
        ax.scatter([tuned], [0], s=175, color=TUNED, zorder=3,
                   label="LoRA on Decision-1 preview")
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.45, 0.7)
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontweight="bold", pad=16)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        if report is banking:
            ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            label = lambda v: f"{100*v:.2f}%"
        else:
            label = lambda v: f"{v:.3f}"
        for value, color, yoffset in ((raw, RAW, -24), (tuned, TUNED, 15)):
            ax.annotate(label(value), (value, 0), xytext=(0, yoffset),
                        textcoords="offset points", ha="center", color=color,
                        fontweight="bold")
        ax.text(0.02, 0.83, annotation, transform=ax.transAxes, fontsize=13,
                fontweight="bold", color=TUNED if report is banking else INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.17))
    fig.text(0.03, -0.015,
             "Validation-selected seed pair; distinct task metrics and scales. BANKING77 was explored after CTU-13.",
             fontsize=9.6, color=INK)
    fig.savefig(OUT / "matched-lora-outcomes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def learning_chart(banking: dict) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.4), layout="constrained")
    for arm, color, label in (("raw", RAW, "LoRA on original Gemma"),
                              ("tuned", TUNED, "LoRA on Decision-1 preview")):
        curves = [banking["validation_curves"][f"{arm}-{seed}"] for seed in (42, 43)]
        steps = curves[0]["grid_steps"]
        if curves[1]["grid_steps"] != steps:
            raise ValueError("The paired seed grids differ")
        for curve in curves:
            ax.plot(steps, curve["grid_accuracy"], color=color, lw=1.2,
                    alpha=0.34, zorder=2)
        means = [sum(values) / len(values) for values in
                 zip(*(curve["grid_accuracy"] for curve in curves), strict=True)]
        ax.plot(steps, means, color=color, lw=3, label=label, zorder=3)
        ax.scatter([steps[0], steps[-1]], [means[0], means[-1]], color=color,
                   s=45, zorder=4)
    ax.set_title("BANKING77 validation: tuned base starts lower, then catches up",
                 fontsize=16, fontweight="bold", loc="left", pad=16)
    ax.set_xlabel("Optimizer updates · same fixed training grid")
    ax.set_ylabel("16-option validation accuracy")
    ax.set_xlim(0, 194)
    ax.set_ylim(0.85, 0.96)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.09, -0.02,
             "Lines: two adapter seeds per base; bold line: seed mean. Original Gemma has higher absolute accuracy AUC.",
             fontsize=9.5)
    fig.savefig(OUT / "banking77-validation-curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def heldout_chart() -> None:
    raw = json.loads((ROOT / "results/raw/full-weight-seed42-heldout-20260924.json").read_text())
    if raw["scope"] != "provisional_single_seed_heldout" or raw["two_seed_selection_complete"]:
        raise ValueError("The expected provisional seed-42 report changed")
    names = {
        "classification": "Intent routing · F1",
        "evidence_wanli": "WANLI evidence · F1",
        "evidence_mnli": "MNLI evidence · F1",
        "cyber": "Authored cyber · accuracy",
        "menu": "Supplied menu · accuracy",
        "procedural": "Authored procedure · accuracy",
        "uncertainty": "Authored uncertainty · accuracy",
    }
    families = {item["family"]: item for item in raw["families"]}
    order = list(names)
    fig, ax = plt.subplots(figsize=(11, 6.3), layout="constrained")
    ypos = list(reversed(range(len(order))))
    for y, key in zip(ypos, order, strict=True):
        item = families[key]
        base = item["variants"]["frozen"]["primary_score"]
        prior = item["variants"]["prior_lora"]["primary_score"]
        full = item["variants"]["full_provisional"]["primary_score"]
        ax.plot([base, full], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter(base, y, marker="o", color=RAW, s=75, zorder=3)
        ax.scatter(prior, y, marker="s", color=PRIOR, s=68, zorder=3)
        ax.scatter(full, y, marker="D", color=TUNED, s=66, zorder=3)
    ax.set_yticks(ypos, [names[k] for k in order])
    ax.set_xlim(0.55, 1.045)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Seed-42 held-out tasks · same BF16 scorer", fontsize=16,
                 fontweight="bold", loc="left", pad=16)
    ax.set_xlabel("Family primary score · task-specific metric")
    ax.scatter([], [], marker="o", color=RAW, label="Frozen Gemma")
    ax.scatter([], [], marker="s", color=PRIOR, label="Earlier task LoRA")
    ax.scatter([], [], marker="D", color=TUNED, label="Full-weight preview")
    ax.legend(frameon=False, loc="lower right", ncol=3, bbox_to_anchor=(1, -0.24))
    fig.text(0.27, -0.07,
             "Source-derived transformations and authored fixtures; full model has one completed training seed.",
             fontsize=9.5)
    fig.savefig(OUT / "seed42-heldout-families.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    banking = checked_report("foundation-banking77-transfer-lora-20260926-v1.json",
                             "foundation-banking77-transfer-lora-20260926-v1")
    ctu = checked_report("foundation-transfer-lora-20260926-v1.json",
                         "foundation-transfer-lora-20260926-v1")
    transfer_chart(banking, ctu)
    learning_chart(banking)
    heldout_chart()


if __name__ == "__main__":
    main()
