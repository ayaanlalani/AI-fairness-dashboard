"""
visualize_track_q.py

Figures for the Track Q analysis, driven by artifacts/consolidated/track_q_analysis.json.

Written as a separate script because visualize_openai_benchmark.py and
visualize_benchmark.py both read the Track H artifacts, have no CLI, and
hardcode a two-dataset dict that excludes lending_club.

Palette: validated light-mode categorical slots 1-2 (#2a78d6, #eb6834) and a
single-hue ordinal blue ramp for the severity matrix. Every series is direct-
labelled so identity never rests on colour alone, and both palettes were
checked with the data-viz validator (all checks pass) rather than eyeballed.

No network, no API key.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Keep matplotlib's cache inside the repo and force a headless backend.
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- palette (light mode, from the validated reference instance) -----------
SERIES_1 = "#2a78d6"   # blue
SERIES_2 = "#eb6834"   # orange
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
STATUS_CRITICAL = "#d03b3b"

# Ordinal severity ramp: one hue, monotone lightness, light end clears 2:1.
SEVERITY_COLORS = {
    "LOW": "#86b6ef",
    "MODERATE": "#3987e5",
    "HIGH": "#1c5cab",
    "CRITICAL": "#0d366b",
}
SEVERITY_TEXT = {"LOW": INK_PRIMARY, "MODERATE": "#ffffff", "HIGH": "#ffffff", "CRITICAL": "#ffffff"}
SEVERITY_ABBR = {"LOW": "LOW", "MODERATE": "MOD", "HIGH": "HIGH", "CRITICAL": "CRIT"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, fontweight="600", loc="left", pad=26 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
                color=INK_SECONDARY, ha="left", va="bottom", wrap=True)
    ax.set_axisbelow(True)


def plot_agreement_by_usecase(data: dict, out_dir: Path) -> Path:
    """Magnitude of one measure across three categories -> single-hue bars."""
    rows = list(data["agreement"]["by_dataset"].items())
    labels = [d for d, _ in rows]
    rates = [v["rate"] * 100 for _, v in rows]
    counts = [(v["agreed"], v["scored"]) for _, v in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(labels, rates, color=SERIES_1, width=0.55)
    for bar in bars:
        bar.set_linewidth(0)
    # Direct labels rather than a legend: one series, identity is in the title.
    for bar, rate, (a, n) in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 2.5,
                f"{rate:.0f}%", ha="center", va="bottom",
                fontsize=11, fontweight="600", color=INK_PRIMARY)
        ax.text(bar.get_x() + bar.get_width() / 2, rate / 2,
                f"{a}/{n}", ha="center", va="center",
                fontsize=9, color="#ffffff")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Severity agreement (%)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}")
    ax.xaxis.grid(False)
    _style(
        ax,
        "gpt-4o agreement with the deterministic severity baseline",
        "Higher is not better performance: lending_club's classifier is near-degenerate, so every "
        "attribute is a trivial LOW.",
    )
    fig.tight_layout()
    path = out_dir / "track_q_agreement_by_usecase.png"
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_strategy_behaviour(data: dict, out_dir: Path) -> Path:
    """Two measures, both on a 0-100 scale -> grouped bars on ONE axis."""
    rows = data["by_strategy"]
    labels = [f"{r['cycle']}. {r['strategy']}" for r in rows]
    scores = [r["mean_score"] for r in rows]
    agree = [r["severity_agreement_rate"] * 100 for r in rows]

    x = range(len(rows))
    w = 0.34
    gap = 0.02  # 2px-equivalent surface gap between adjacent fills
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    b1 = ax.bar([i - w / 2 - gap for i in x], scores, width=w, color=SERIES_1, label="Mean score (/100)")
    b2 = ax.bar([i + w / 2 + gap for i in x], agree, width=w, color=SERIES_2, label="Severity agreement (%)")

    for bars in (b1, b2):
        for bar in bars:
            bar.set_linewidth(0)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    fontsize=9, color=INK_SECONDARY)

    # Mark where the brevity heuristic fires, so the constrained dip is legible.
    # Placed above the bars on the surface — red-on-orange inside a fill is
    # unreadable.
    for i, r in enumerate(rows):
        if r.get("brevity_only_refusal_count"):
            ax.annotate(
                f"{r['brevity_only_refusal_count']}/{r['n']} flagged by the brevity\n"
                f"heuristic, 0 genuine refusals",
                xy=(i, max(scores[i], agree[i]) + 7), ha="center", va="bottom",
                fontsize=8, color=STATUS_CRITICAL, fontweight="600", linespacing=1.4,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Score / agreement")
    ax.xaxis.grid(False)
    leg = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16), ncols=2, fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)
    _style(
        ax,
        "Prompt strategy vs interpretive reliability",
        "Both measures share a 0-100 scale, so they sit on one axis. self_critique peaks on both.",
    )
    fig.tight_layout()
    path = out_dir / "track_q_strategy_behaviour.png"
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_severity_matrix(data: dict, out_dir: Path) -> Path:
    """Identity of an ordered label per cell -> ordinal one-hue matrix."""
    rows = data["stability"]
    fig, ax = plt.subplots(figsize=(8.0, 0.52 * len(rows) + 2.0))

    for r_i, row in enumerate(rows):
        # Baseline column, then the four cycles.
        cells = [row["baseline_severity"]] + list(row["cycle_severities"])
        for c_i, sev in enumerate(cells):
            colour = SEVERITY_COLORS.get(sev, "#f0efec")
            # 2px surface gap between fills.
            ax.add_patch(plt.Rectangle(
                (c_i + 0.02, r_i + 0.02), 0.96, 0.96,
                facecolor=colour, edgecolor=SURFACE, linewidth=2,
            ))
            ax.text(c_i + 0.5, r_i + 0.5, SEVERITY_ABBR.get(sev, "—"),
                    ha="center", va="center", fontsize=8,
                    color=SEVERITY_TEXT.get(sev, INK_MUTED), fontweight="600")
        if not row["stable_across_cycles"]:
            ax.text(5.15, r_i + 0.5, "unstable", ha="left", va="center",
                    fontsize=8, color=STATUS_CRITICAL, fontweight="600")

    ax.set_xlim(0, 6.4)
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([i + 0.5 for i in range(5)])
    ax.set_xticklabels(
        ["baseline", "1\nzero-shot", "2\nchain-of-\nthought", "3\nself-\ncritique", "4\nconstrained"],
        fontsize=8,
    )
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([f"{r['dataset']} · {r['attribute']}" for r in rows], fontsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    handles = [Patch(facecolor=SEVERITY_COLORS[s], label=s) for s in ["LOW", "MODERATE", "HIGH", "CRITICAL"]]
    leg = ax.legend(handles=handles, frameon=False, loc="upper left",
                    bbox_to_anchor=(0, -0.06 - 0.5 / len(rows)), ncols=4, fontsize=8)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)
    ax.set_axisbelow(True)
    # Lay out first, then place figure-level text: tight_layout() reflows the
    # axes and leaves ghosted glyphs if the text is added before it. An
    # axes-relative subtitle collides with the title on an axes this tall.
    fig.tight_layout()
    fig.suptitle(
        "Severity label per prompt strategy, against identical input context",
        color=INK_PRIMARY, fontsize=12, fontweight="600", x=0.01, ha="left", y=1.05,
    )
    fig.text(
        0.01, 1.005,
        "Every cell in a row saw the same metrics. Where a row changes, prompt phrasing alone moved the verdict.",
        fontsize=9, color=INK_SECONDARY, ha="left", va="bottom",
    )
    path = out_dir / "track_q_severity_matrix.png"
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_grounding_effect(data: dict, out_dir: Path) -> Path:
    """The research-grounding contrast: the headline Track Q result."""
    legacy = data.get("legacy_contrast")
    if not legacy:
        return None

    labels = ["Hallucination\nflags", "Mean score\n(/100)"]
    before = [legacy["hallucination_flags"], legacy["mean_score"]]
    after = [legacy["current_hallucination_flags"], legacy["current_mean"]]

    # One hue, two steps: these bars are two conditions of the SAME measure, not
    # two entities. Reusing the categorical blue/orange pair here would make
    # colour track rank ("before"/"after") while it tracks measure in the
    # strategy figure. The lighter step is the baseline, the focal step is the
    # result; direct labels carry the values since step 250 is sub-3:1.
    COND_BEFORE, COND_AFTER = "#86b6ef", SERIES_1
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for ax, label, b, a, fmt in zip(axes, labels, before, after, ["{:.0f}", "{:.1f}"]):
        bars = ax.bar(["no evidence", "evidence\nembedded"], [b, a],
                      color=[COND_BEFORE, COND_AFTER], width=0.5)
        for bar, val in zip(bars, [b, a]):
            bar.set_linewidth(0)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt.format(val), ha="center", va="bottom",
                    fontsize=12, fontweight="600", color=INK_PRIMARY)
        ax.set_ylabel(label.replace("\n", " "))
        ax.set_ylim(0, max(b, a) * 1.28 or 1)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Embedding Semantic Scholar evidence cut fabricated citations 8-fold",
        color=INK_PRIMARY, fontsize=12, fontweight="600", x=0.02, ha="left", y=1.04,
    )
    fig.text(0.02, 0.97,
             f"Same model, same two use cases ({', '.join(legacy['shared_datasets'])}), "
             f"{legacy['n_records']} vs {legacy['current_n']} records.",
             fontsize=9, color=INK_SECONDARY, ha="left")
    fig.tight_layout()
    path = out_dir / "track_q_grounding_effect.png"
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Track Q figures (no API calls)")
    parser.add_argument("--analysis_json",
                        default=str(REPO_ROOT / "artifacts" / "consolidated" / "track_q_analysis.json"))
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "artifacts" / "visualizations"))
    args = parser.parse_args()

    data = json.loads(Path(args.analysis_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fn in (plot_agreement_by_usecase, plot_strategy_behaviour,
               plot_severity_matrix, plot_grounding_effect):
        path = fn(data, out_dir)
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
