"""Generate result figures for the README.

Reads metrics.json from baseline and improved results directories and
produces two PNG charts in results/figures/.

Usage:
    python scripts/generate_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "results"
_FIGURES_DIR = _RESULTS_DIR / "figures"

# ── Style constants ──────────────────────────────────────────────────────

_COLOR_BASELINE = "#9CA3AF"   # neutral gray
_COLOR_IMPROVED = "#2563EB"   # blue
_COLOR_WEAK     = "#DC2626"   # red — highlights weak points
_COLOR_STRONG   = "#2563EB"   # blue — normal bars

_FONT_TITLE = 13
_FONT_LABEL = 11
_FONT_TICK  = 10
_FONT_ANNOT = 9

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _load_metrics(pipeline: str) -> dict:
    path = _RESULTS_DIR / pipeline / "metrics.json"
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


# ── Figure 1: M3 Position Recall ─────────────────────────────────────────

def _generate_position_recall(baseline: dict, improved: dict) -> None:
    b_m3 = baseline["metrics"]["M3_lost_in_middle"]
    i_m3 = improved["metrics"]["M3_lost_in_middle"]

    buckets = ["Top", "Middle", "Bottom"]
    b_vals = [b_m3["recall_top"], b_m3["recall_middle"], b_m3["recall_bottom"]]
    i_vals = [i_m3["recall_top"], i_m3["recall_middle"], i_m3["recall_bottom"]]

    x = np.arange(len(buckets))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars_b = ax.bar(x - width/2, [v * 100 for v in b_vals], width,
                     label="Baseline", color=_COLOR_BASELINE, edgecolor="white", linewidth=0.5)
    bars_i = ax.bar(x + width/2, [v * 100 for v in i_vals], width,
                     label="Improved", color=_COLOR_IMPROVED, edgecolor="white", linewidth=0.5)

    # Annotations
    for bar, val in zip(bars_b, b_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val*100:.0f}%", ha="center", va="bottom",
                fontsize=_FONT_ANNOT, color=_COLOR_BASELINE, fontweight="bold")
    for bar, val in zip(bars_i, i_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val*100:.0f}%", ha="center", va="bottom",
                fontsize=_FONT_ANNOT, color=_COLOR_IMPROVED, fontweight="bold")

    ax.set_ylabel("Edit Recall (%)", fontsize=_FONT_LABEL)
    ax.set_title("Edit Recall by WI Position (M3)", fontsize=_FONT_TITLE, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=_FONT_TICK)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.set_ylim(0, 105)
    ax.legend(fontsize=_FONT_TICK, loc="lower right")

    # Subtle grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = _FIGURES_DIR / "position_recall.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


# ── Figure 2: M4 Action Rate by Type ─────────────────────────────────────

def _generate_m4_by_type(improved: dict) -> None:
    m4 = improved["metrics"]["M4_rule_consistency"]["by_type"]

    # Sort by rate descending, renumber last (weak point highlighted)
    types_sorted = sorted(m4.items(), key=lambda kv: -kv[1]["rate"])
    labels = [t for t, _ in types_sorted]
    rates  = [d["rate"] * 100 for _, d in types_sorted]
    counts = [d["n"] for _, d in types_sorted]
    colors = [_COLOR_WEAK if r < 50 else _COLOR_STRONG for r in rates]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars = ax.barh(range(len(labels)), rates, color=colors, edgecolor="white", linewidth=0.5)

    # Labels and annotations
    for i, (bar, rate, n) in enumerate(zip(bars, rates, counts)):
        # Rate label at end of bar
        x_text = bar.get_width() + 1.5
        ax.text(x_text, bar.get_y() + bar.get_height()/2,
                f"{rate:.0f}%  (n={n})", va="center", fontsize=_FONT_ANNOT, fontweight="bold",
                color=colors[i])

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=_FONT_TICK)
    ax.set_xlabel("Action Rate (%)", fontsize=_FONT_LABEL)
    ax.set_title("Rule Consistency: Action Rate by Transformation Type (M4)",
                 fontsize=_FONT_TITLE, fontweight="bold", pad=12)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()

    # Subtle grid
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = _FIGURES_DIR / "m4_by_type.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    baseline = _load_metrics("baseline")
    improved = _load_metrics("improved")

    print("Generating figures...")
    _generate_position_recall(baseline, improved)
    _generate_m4_by_type(improved)
    print("Done.")


if __name__ == "__main__":
    main()
