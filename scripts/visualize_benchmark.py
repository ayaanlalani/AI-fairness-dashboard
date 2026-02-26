"""
visualize_benchmark.py

Generates comparison charts between our deterministic pipeline (AIF360) and
the Gemini LLM benchmark (fairlearn).

Reads from artifacts/ and writes PNGs to artifacts/visualizations/.

Usage:
  python3 scripts/visualize_benchmark.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
VIZ_DIR = ARTIFACTS / "visualizations"

DATASETS = {
    "german_credit": {"label": "German Credit", "color": "#2563EB"},
    "hmda": {"label": "HMDA (Georgia)", "color": "#D97706"},
}

METRIC_KEYS = {
    "disparate_impact": ("DisparateImpact", "Disparate Impact"),
    "demographic_parity_diff": ("DemographicParityDiff", "Demographic Parity Diff"),
    "equal_opportunity_diff": ("EqualOpportunityDiff", "Equal Opportunity Diff"),
    "average_odds_diff": ("AverageOddsDiff", "Average Odds Diff"),
    "theil_index": ("TheilIndex", "Theil Index"),
}


def load_data() -> dict:
    """Load our metrics + Gemini metrics for each dataset."""
    data = {}
    for ds_key, ds_info in DATASETS.items():
        ds_dir = ARTIFACTS / ds_key

        our_csv = ds_dir / "fairness" / "fairness_metrics.csv"
        llm_json = ds_dir / "fairness" / "llm_raw_response.json"

        if not our_csv.exists() or not llm_json.exists():
            log.warning(f"Missing data for {ds_key}, skipping")
            continue

        our_df = pd.read_csv(our_csv)
        with open(llm_json) as f:
            llm_data = json.load(f)

        data[ds_key] = {
            "label": ds_info["label"],
            "color": ds_info["color"],
            "our": our_df,
            "llm_metrics": llm_data["result"]["metrics"],
            "llm_qual": llm_data["result"]["qualitative"],
            "usage": llm_data["usage"],
        }
    return data


def _match_our_row(our_df: pd.DataFrame, attr: str) -> dict | None:
    candidates = [attr, attr.replace("_original", ""), attr.split("_original")[0]]
    for c in candidates:
        mask = our_df["Attribute"].astype(str) == c
        if mask.any():
            return our_df[mask].iloc[0].to_dict()
    return None


def plot_metric_comparison(data: dict) -> None:
    """Side-by-side grouped bar chart: Our Pipeline vs Gemini for each metric."""
    for ds_key, ds in data.items():
        attrs = [m["attribute"] for m in ds["llm_metrics"]]
        short_attrs = [a.replace("_original", "") for a in attrs]

        for gem_key, (our_key, display_name) in METRIC_KEYS.items():
            our_vals = []
            gem_vals = []
            for m in ds["llm_metrics"]:
                gem_vals.append(m[gem_key])
                our_row = _match_our_row(ds["our"], m["attribute"])
                our_vals.append(float(our_row[our_key]) if our_row and our_key in our_row else 0)

            x = np.arange(len(short_attrs))
            width = 0.35

            fig, ax = plt.subplots(figsize=(max(6, len(attrs) * 2), 5))
            bars1 = ax.bar(x - width / 2, our_vals, width, label="Our Pipeline (AIF360)",
                           color="#3B82F6", edgecolor="white", linewidth=0.5)
            bars2 = ax.bar(x + width / 2, gem_vals, width, label="Gemini LLM (fairlearn)",
                           color="#F59E0B", edgecolor="white", linewidth=0.5)

            ax.set_ylabel(display_name, fontsize=11)
            ax.set_title(f"{ds['label']}: {display_name}", fontsize=13, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(short_attrs, fontsize=10)
            ax.legend(fontsize=9)
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

            if gem_key == "disparate_impact":
                ax.axhline(y=0.8, color="red", linewidth=1, linestyle="--", alpha=0.7, label="Bias threshold (0.8)")
                ax.legend(fontsize=9)

            for bar in bars1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
            for bar in bars2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

            plt.tight_layout()
            fname = f"{ds_key}_{gem_key}.png"
            fig.savefig(VIZ_DIR / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f"  Saved {fname}")


def plot_delta_heatmap(data: dict) -> None:
    """Heatmap of metric deltas (Gemini - Ours) across all datasets and attributes."""
    rows = []
    for ds_key, ds in data.items():
        for m in ds["llm_metrics"]:
            our_row = _match_our_row(ds["our"], m["attribute"])
            for gem_key, (our_key, display_name) in METRIC_KEYS.items():
                gem_val = m[gem_key]
                our_val = float(our_row[our_key]) if our_row and our_key in our_row else None
                delta = gem_val - our_val if our_val is not None else None
                attr_short = m["attribute"].replace("_original", "")
                rows.append({
                    "Dataset": ds["label"],
                    "Attribute": attr_short,
                    "Metric": display_name,
                    "Delta": delta,
                })

    df = pd.DataFrame(rows)
    df["Label"] = df["Dataset"] + "\n" + df["Attribute"]

    pivot = df.pivot(index="Label", columns="Metric", values="Delta")
    col_order = [v[1] for v in METRIC_KEYS.values()]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.7)))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=-0.3, vmax=0.3)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if val is not None and not np.isnan(val):
                color = "white" if abs(val) > 0.15 else "black"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    ax.set_title("Metric Deltas (Gemini - Our Pipeline)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Delta (green = Gemini lower, red = Gemini higher)", shrink=0.8)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "delta_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved delta_heatmap.png")


def plot_severity_comparison(data: dict) -> None:
    """Side-by-side severity classification comparison."""
    sev_order = ["CRITICAL", "HIGH", "MODERATE", "LOW"]
    sev_to_num = {s: i for i, s in enumerate(sev_order)}

    rows = []
    for ds_key, ds in data.items():
        our_df = ds["our"]
        for q in ds["llm_qual"]:
            attr = q["attribute"]
            gem_sev = q["severity"].upper()
            for label in sev_order:
                if label in gem_sev:
                    gem_sev = label
                    break

            our_row = _match_our_row(our_df, attr)
            if our_row:
                di = float(our_row.get("DisparateImpact", 1.0))
                if di < 0.72:
                    our_sev = "CRITICAL"
                elif di < 0.80:
                    our_sev = "HIGH"
                elif di < 0.95:
                    our_sev = "MODERATE"
                else:
                    our_sev = "LOW"
            else:
                our_sev = "LOW"

            rows.append({
                "Label": f"{ds['label']}\n{attr.replace('_original', '')}",
                "Our Pipeline": sev_to_num.get(our_sev, 3),
                "Gemini LLM": sev_to_num.get(gem_sev, 3),
                "Our Sev": our_sev,
                "Gem Sev": gem_sev,
            })

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.35

    colors_map = {"CRITICAL": "#DC2626", "HIGH": "#EA580C", "MODERATE": "#F59E0B", "LOW": "#22C55E"}

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 2), 5))

    for i, row in df.iterrows():
        ax.bar(i - width / 2, 1, width, bottom=row["Our Pipeline"],
               color=colors_map.get(row["Our Sev"], "gray"), edgecolor="white", linewidth=0.5)
        ax.bar(i + width / 2, 1, width, bottom=row["Gemini LLM"],
               color=colors_map.get(row["Gem Sev"], "gray"), edgecolor="white", linewidth=0.5,
               hatch="///" if row["Our Sev"] != row["Gem Sev"] else "")

    ax.set_yticks(range(len(sev_order)))
    ax.set_yticklabels(sev_order, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"], fontsize=9)
    ax.set_ylabel("Severity Level")
    ax.set_title("Severity Classification: Our Pipeline vs Gemini", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#3B82F6", label="Our Pipeline (AIF360)"),
        Patch(facecolor="#F59E0B", label="Gemini LLM (fairlearn)"),
        Patch(facecolor="lightgray", hatch="///", label="Disagreement"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(VIZ_DIR / "severity_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved severity_comparison.png")


def plot_cost_summary(data: dict) -> None:
    """Bar chart of token usage and cost per dataset."""
    rows = []
    for ds_key, ds in data.items():
        u = ds["usage"]
        rows.append({
            "Dataset": ds["label"],
            "Input Tokens": u["input_tokens"],
            "Output Tokens": u["output_tokens"],
            "Cost ($)": u["total_cost_usd"],
            "Time (s)": u["wall_clock_s"],
        })

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Tokens
    ax = axes[0]
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["Input Tokens"], w, label="Input", color="#3B82F6")
    ax.bar(x + w / 2, df["Output Tokens"], w, label="Output", color="#F59E0B")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Dataset"], fontsize=10)
    ax.set_ylabel("Token Count")
    ax.set_title("Token Usage", fontweight="bold")
    ax.legend(fontsize=9)
    for i, row in df.iterrows():
        ax.text(i - w / 2, row["Input Tokens"], f'{row["Input Tokens"]:,}',
                ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, row["Output Tokens"], f'{row["Output Tokens"]:,}',
                ha="center", va="bottom", fontsize=8)

    # Cost
    ax = axes[1]
    bars = ax.bar(df["Dataset"], df["Cost ($)"], color=["#3B82F6", "#D97706"])
    ax.set_ylabel("Estimated Cost (USD)")
    ax.set_title("API Cost", fontweight="bold")
    for bar, cost in zip(bars, df["Cost ($)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${cost:.4f}", ha="center", va="bottom", fontsize=9)

    # Time
    ax = axes[2]
    bars = ax.bar(df["Dataset"], df["Time (s)"], color=["#3B82F6", "#D97706"])
    ax.set_ylabel("Wall-Clock Time (s)")
    ax.set_title("Latency", fontweight="bold")
    for bar, t in zip(bars, df["Time (s)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.1f}s", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Gemini LLM Benchmark: Napkin Math", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "cost_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved cost_summary.png")


def plot_di_overview(data: dict) -> None:
    """Single chart: DI values from both pipelines across all datasets/attributes."""
    rows = []
    for ds_key, ds in data.items():
        for m in ds["llm_metrics"]:
            attr_short = m["attribute"].replace("_original", "")
            our_row = _match_our_row(ds["our"], m["attribute"])
            our_di = float(our_row["DisparateImpact"]) if our_row else None
            rows.append({
                "Label": f"{ds['label']}\n{attr_short}",
                "Our DI": our_di,
                "Gemini DI": m["disparate_impact"],
            })

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.8), 5))
    ax.bar(x - width / 2, df["Our DI"], width, label="Our Pipeline (AIF360)", color="#3B82F6")
    ax.bar(x + width / 2, df["Gemini DI"], width, label="Gemini LLM (fairlearn)", color="#F59E0B")
    ax.axhline(y=0.8, color="red", linewidth=1.5, linestyle="--", alpha=0.8, label="Bias threshold (DI = 0.8)")
    ax.axhline(y=1.0, color="green", linewidth=0.8, linestyle=":", alpha=0.5, label="Parity (DI = 1.0)")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"], fontsize=9)
    ax.set_ylabel("Disparate Impact", fontsize=11)
    ax.set_title("Disparate Impact: Pipeline vs LLM Benchmark", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, max(df[["Our DI", "Gemini DI"]].max().max() * 1.15, 1.2))

    for i, row in df.iterrows():
        if row["Our DI"] is not None:
            ax.text(i - width / 2, row["Our DI"], f"{row['Our DI']:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, row["Gemini DI"], f"{row['Gemini DI']:.3f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(VIZ_DIR / "di_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved di_overview.png")


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    if not data:
        log.error("No data found in artifacts/. Run the pipeline first.")
        return

    log.info(f"Loaded data for: {list(data.keys())}")
    log.info("Generating visualizations...")

    plot_di_overview(data)
    plot_metric_comparison(data)
    plot_delta_heatmap(data)
    plot_severity_comparison(data)
    plot_cost_summary(data)

    log.info(f"All visualizations saved to {VIZ_DIR}")


if __name__ == "__main__":
    main()
