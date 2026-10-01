"""Reusable matplotlib charting helpers for notebook pipelines.

Adheres to 13x5in, 100 DPI styling with accessible design tokens:
Blue #2b5c8f, Orange #d97706, Neutral #475569.
Titles are 12pt bold, labels/annotations 10pt.
Each function returns a matplotlib Figure for inline display/saving.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

COLOR_BLUE, COLOR_ORANGE, COLOR_NEUTRAL = "#2b5c8f", "#d97706", "#475569"
COLOR_BG, COLOR_BORDER = "#f8fafc", "#cbd5e1"
FONT_TITLE, FONT_LABEL = 12, 10

STATUS_PALETTE: dict[str, str] = {
    "ZERO FAULTS": "#15803d", "VERIFIED": "#15803d", "PASS": "#15803d", "PASSED": "#15803d",
    "BASELINE": COLOR_BLUE, "APPLICABLE": COLOR_BLUE, "CHALLENGER": COLOR_ORANGE,
    "PENDING": COLOR_ORANGE, "PENDING APPROVAL": COLOR_ORANGE, "UNMEASURED": COLOR_NEUTRAL,
    "NOT RUN": COLOR_NEUTRAL, "EXCLUDED": COLOR_NEUTRAL, "BLOCKED": COLOR_ORANGE, "REJECTED": "#b91c1c",
}


def _init_fig(title: str, nrows: int = 1, ncols: int = 1, height: float = 5.0) -> tuple[Figure, Any]:
    fig, ax = plt.subplots(nrows, ncols, figsize=(13, height), dpi=100)
    fig.patch.set_facecolor("white")
    if isinstance(ax, np.ndarray):
        fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold", color="#0f172a", y=0.98)
    else:
        ax.set_facecolor(COLOR_BG)
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", color="#0f172a", pad=12)
    return fig, ax


def _empty_ax(ax: Any, msg: str = "No data available") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=FONT_LABEL, color=COLOR_NEUTRAL)
    ax.set_xticks([]); ax.set_yticks([])


def plot_comparative_count_bars(data: dict[str, dict[str, int]], title: str, xlabel: str = "Count") -> Figure:
    fig, ax = _init_fig(title)
    if not data:
        _empty_ax(ax); fig.tight_layout(); return fig
    categories = list(data.keys())
    subgroups = list(dict.fromkeys(k for d in data.values() for k in d))
    y = np.arange(len(categories))
    height = 0.8 / max(len(subgroups), 1)
    colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_NEUTRAL, "#64748b"]
    for i, sub in enumerate(subgroups):
        vals = [data[c].get(sub, 0) for c in categories]
        ax.barh(y + i * height, vals, height=height, label=sub, color=colors[i % len(colors)])
    ax.set_yticks(y + height * (len(subgroups) - 1) / 2)
    ax.set_yticklabels(categories, fontsize=FONT_LABEL)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL); ax.set_xlim(left=0)
    ax.legend(frameon=True, facecolor="white", edgecolor=COLOR_BORDER, fontsize=FONT_LABEL)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=COLOR_NEUTRAL)
    fig.tight_layout(); return fig


def plot_horizontal_missingness(data: dict[str, float], title: str = "Feature Missingness Rate (%)") -> Figure:
    fig, ax = _init_fig(title)
    if not data:
        _empty_ax(ax); fig.tight_layout(); return fig
    keys = list(data.keys())[::-1]
    vals = [float(data[k]) for k in keys]
    colors = [COLOR_ORANGE if v > 0.0 else COLOR_BLUE for v in vals]
    bars = ax.barh(keys, vals, color=colors, height=0.6)
    ax.set_xlim(left=0.0, right=max(max(vals, default=0.0) * 1.15, 1.0))
    ax.set_xlabel("Missing Percentage (%)", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_LABEL)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", fontsize=FONT_LABEL, color=COLOR_NEUTRAL)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=COLOR_NEUTRAL)
    fig.tight_layout(); return fig


def plot_domain_auc_dotchart(data: dict[str, float], title: str = "Adversarial Domain AUC (0.5 Baseline)") -> Figure:
    fig, ax = _init_fig(title)
    if not data:
        _empty_ax(ax); fig.tight_layout(); return fig
    domains = list(data.keys())[::-1]
    scores = [float(data[d]) for d in domains]
    ax.axvspan(0.0, 0.50, color="#f1f5f9", alpha=0.6, label="Inverse Ranking (<0.50)")
    ax.axvspan(0.50, 0.55, color="#e2e8f0", alpha=0.5, label="Weak Shift (0.50-0.55)")
    ax.axvspan(0.55, 0.60, color="#fef3c7", alpha=0.5, label="Moderate Shift (0.55-0.60)")
    ax.axvspan(0.60, 1.00, color="#fee2e2", alpha=0.4, label="Material Shift (>0.60)")
    ax.axvline(0.50, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.5, label="Baseline (0.50)")
    y_pos = np.arange(len(domains))
    for y, score in zip(y_pos, scores):
        color = COLOR_ORANGE if (score > 0.60 or score < 0.50) else COLOR_BLUE
        ax.hlines(y, xmin=min(0.5, score), xmax=max(0.5, score), color=COLOR_BORDER, linewidth=1.5)
        ax.scatter(score, y, color=color, s=80, zorder=4)
        label_text = f"{score:.4f}" if round(score, 4) == score else f"{score:.6f}"
        ax.text(score + 0.01, y, label_text, va="center", fontsize=FONT_LABEL, fontweight="bold", color=color)
    ax.set_yticks(y_pos); ax.set_yticklabels(domains, fontsize=FONT_LABEL)
    ax.set_xlim(0.0, 1.0); ax.set_xlabel("Adversarial AUC ROC", fontsize=FONT_LABEL)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=COLOR_BORDER, fontsize=FONT_LABEL)
    ax.grid(axis="x", linestyle=":", alpha=0.4, color=COLOR_NEUTRAL)
    fig.tight_layout(); return fig


def plot_fold_composition(data: dict[str, dict[str, int]], title: str = "Partition Composition by Fold") -> Figure:
    fig, ax = _init_fig(title)
    if not data:
        _empty_ax(ax); fig.tight_layout(); return fig
    folds = list(data.keys())[::-1]
    partitions = list(dict.fromkeys(p for d in data.values() for p in d))
    color_map = {"normal": COLOR_BLUE, "anomaly": COLOR_ORANGE}
    palette = [COLOR_BLUE, COLOR_ORANGE, COLOR_NEUTRAL, "#64748b"]
    bottom = np.zeros(len(folds))
    for i, part in enumerate(partitions):
        counts = [data[f].get(part, 0) for f in folds]
        col = color_map.get(part.lower(), palette[i % len(palette)])
        ax.barh(folds, counts, left=bottom, color=col, height=0.6, label=f"{part} Partitions")
        bottom += np.array(counts)
    ax.set_xlim(left=0); ax.set_xlabel("Partition Sample Count", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_LABEL)
    ax.legend(frameon=True, facecolor="white", edgecolor=COLOR_BORDER, fontsize=FONT_LABEL)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=COLOR_NEUTRAL)
    fig.tight_layout(); return fig


def plot_feature_gate_bars(data: dict[str, int], title: str = "Feature Gate Evaluation Counts") -> Figure:
    fig, ax = _init_fig(title)
    if not data:
        _empty_ax(ax); fig.tight_layout(); return fig
    keys = list(data.keys())[::-1]
    vals = [data[k] for k in keys]
    colors = [
        COLOR_BLUE if any(tok in k.lower() for tok in ("verified", "pass"))
        else COLOR_NEUTRAL if "pending" in k.lower()
        else COLOR_ORANGE
        for k in keys
    ]
    bars = ax.barh(keys, vals, color=colors, height=0.55)
    ax.set_xlim(left=0, right=max(vals, default=0) * 1.15 + 1)
    ax.set_xlabel("Feature Count", fontsize=FONT_LABEL)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=FONT_LABEL)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=COLOR_NEUTRAL)
    fig.tight_layout(); return fig


def plot_experiment_status_bars(data: dict[str, dict[str, int]], title: str = "Experiment Execution by Category") -> Figure:
    return plot_comparative_count_bars(data, title=title, xlabel="Design Rows")


def render_tiles(data: list[dict[str, str]], title: str) -> Figure:
    if not data:
        fig, ax = _init_fig(title); _empty_ax(ax); fig.tight_layout(); return fig
    n = len(data)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, max(2.2 * rows, 3.0)), dpi=100)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold", color="#0f172a", y=0.98)
    ax_list = np.array(axes).reshape(-1)
    for idx, ax in enumerate(ax_list):
        ax.set_facecolor(COLOR_BG); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(COLOR_BORDER)
        if idx < n:
            item = data[idx]
            header = item.get("header") or item.get("label", "")
            status = str(item.get("status") or item.get("value", "")).upper()
            meta = item.get("meta") or item.get("subtext", "")
            st_color = STATUS_PALETTE.get(status, COLOR_NEUTRAL)
            ax.text(0.5, 0.72, header, ha="center", va="center", fontsize=FONT_LABEL, fontweight="bold", color="#1e293b")
            ax.text(0.5, 0.42, status, ha="center", va="center", fontsize=FONT_LABEL, fontweight="bold", color=st_color,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": st_color, "lw": 1.2})
            if meta:
                ax.text(0.5, 0.16, meta, ha="center", va="center", fontsize=FONT_LABEL - 1, color=COLOR_NEUTRAL)
        else:
            ax.set_visible(False)
    fig.tight_layout(); return fig


def render_integrity_tiles(data: list[dict[str, str]], title: str = "Data & System Integrity Status (Zero Faults)") -> Figure:
    return render_tiles(data, title=title)


def render_baseline_role_tiles(data: list[dict[str, str]], title: str = "Baseline & Model Role Assignments (No Fake Scores)") -> Figure:
    return render_tiles(data, title=title)


def render_family_execution_tiles(data: list[dict[str, str]], title: str = "Model Family Applicability & Execution Status") -> Figure:
    return render_tiles(data, title=title)


def render_unmeasured_cap_tiles(data: list[dict[str, str]], title: str = "Resource Governance & Pending Approval Tiles") -> Figure:
    return render_tiles(data, title=title)
