"""
generate_consolidated_visuals.py

Aggregate fairness metrics across all datasets in the project and generate
poster-ready visualizations that highlight disparate impact, fairness gaps,
and bias flags in a single glance.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LOGGER = logging.getLogger("consolidated_visuals")

DATASET_SPECS: List[Dict[str, str]] = [
    {
        "name": "Diabetes",
        "path": "diabetes_dataset/metrics/fairness/fairness_metrics.csv",
    },
    {
        "name": "Healthcare Insurance",
        "path": "Healthcare-insurance-dataset/metrics/fairness/fairness_metrics.csv",
    },
    {
        "name": "German Credit",
        "path": "german_credit_dataset/metrics/fairness/fairness_metrics.csv",
    },
    {
        "name": "Lending Club",
        "path": "lending_club_dataset/metrics/fairness/fairness_metrics.csv",
    },
]

STANDARD_COLUMNS = {
    "Attribute": "Attribute",
    "protected_attribute": "Attribute",
    "attribute": "Attribute",
    "PrivilegedValue": "PrivilegedValue",
    "privileged_group": "PrivilegedValue",
    "DisparateImpact": "DisparateImpact",
    "disparate_impact": "DisparateImpact",
    "DemographicParityDiff": "DemographicParityDiff",
    "demographic_parity_difference": "DemographicParityDiff",
    "EqualOpportunityDiff": "EqualOpportunityDiff",
    "equal_opportunity_difference": "EqualOpportunityDiff",
    "AverageOddsDiff": "AverageOddsDiff",
    "average_odds_difference": "AverageOddsDiff",
    "BiasFlag": "BiasFlag",
    "bias_detected": "BiasFlag",
}


def configure_logging(level: int = logging.INFO) -> None:
    if LOGGER.handlers:
        LOGGER.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
    )
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


def load_fairness_table(path: Path, dataset_name: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        LOGGER.warning("Skipping %s (missing file: %s)", dataset_name, path)
        return None

    raw_df = pd.read_csv(path)
    rename_map = {c: STANDARD_COLUMNS[c] for c in raw_df.columns if c in STANDARD_COLUMNS}
    df = raw_df.rename(columns=rename_map)
    required_cols = {"Attribute", "DisparateImpact"}
    if not required_cols.issubset(df.columns):
        LOGGER.warning("Skipping %s (missing required columns)", dataset_name)
        return None

    df["Dataset"] = dataset_name
    df["Attribute"] = df["Attribute"].astype(str)
    for col in [
        "DisparateImpact",
        "DemographicParityDiff",
        "EqualOpportunityDiff",
        "AverageOddsDiff",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "BiasFlag" in df.columns:
        df["BiasFlag"] = df["BiasFlag"].astype(bool)
    else:
        df["BiasFlag"] = df["DisparateImpact"] < 0.8

    return df


def load_all_datasets(specs: List[Dict[str, str]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for spec in specs:
        df = load_fairness_table(Path(spec["path"]), spec["name"])
        if df is not None:
            frames.append(df)
    if not frames:
        raise RuntimeError("No fairness data found. Ensure per-dataset metrics exist.")
    combined = pd.concat(frames, ignore_index=True)
    combined["DisplayLabel"] = (
        combined["Dataset"]
        + " · "
        + combined["Attribute"].str.replace("_", " ").str.title()
    )
    return combined


def ensure_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_disparate_impact(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.dropna(subset=["DisparateImpact"]).copy()
    plot_df = plot_df.sort_values("DisparateImpact")
    plt.figure(figsize=(11, 8))
    sns.barplot(
        data=plot_df,
        x="DisparateImpact",
        y="DisplayLabel",
        hue="Dataset",
        dodge=False,
        palette="Set2",
    )
    plt.axvline(0.8, color="red", linestyle="--", linewidth=1, label="80% Rule Threshold")
    plt.axvline(1.0, color="gray", linestyle=":", linewidth=1, label="Parity (1.0)")
    plt.xlabel("Disparate Impact (Privileged / Unprivileged selection rate)")
    plt.ylabel("")
    plt.title("Disparate Impact Across All Pipelines")
    plt.xlim(0, max(1.8, plot_df["DisparateImpact"].max() + 0.1))
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gap_panels(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("DemographicParityDiff", "Demographic Parity Δ"),
        ("EqualOpportunityDiff", "Equal Opportunity Δ"),
        ("AverageOddsDiff", "Average Odds Δ"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 6), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        plot_df = df.dropna(subset=[col]).copy()
        plot_df = plot_df.sort_values(col)
        sns.barplot(
            data=plot_df,
            x=col,
            y="DisplayLabel",
            hue="Dataset",
            dodge=False,
            palette="Set2",
            ax=ax,
        )
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Difference (privileged - unprivileged)")
        ax.set_ylabel("")
        if ax != axes[0]:
            ax.set_yticklabels([])
        ax.legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    plt.suptitle("Fairness Gap Comparisons", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bias_matrix(df: pd.DataFrame, out_path: Path) -> None:
    pivot = (
        df.assign(BiasValue=lambda d: d["BiasFlag"].astype(int))
        .pivot_table(
            index="Dataset",
            columns="Attribute",
            values="BiasValue",
            aggfunc="max",
        )
        .fillna(0)
    )
    plt.figure(figsize=(10, 4 + 0.4 * len(pivot)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap=sns.color_palette(["#4CAF50", "#FFC107", "#D32F2F"], as_cmap=True),
        cbar=False,
    )
    plt.title("Bias Flags (1 = DI below 0.8)")
    plt.xlabel("Attribute")
    plt.ylabel("Dataset")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    summary_cols = [
        "Dataset",
        "Attribute",
        "PrivilegedValue",
        "DisparateImpact",
        "DemographicParityDiff",
        "EqualOpportunityDiff",
        "AverageOddsDiff",
        "BiasFlag",
    ]
    df.to_csv(out_path, columns=[c for c in summary_cols if c in df.columns], index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate consolidated fairness visualizations")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="poster_assets",
        help="Directory to place consolidated plots and tables",
    )
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    combined_df = load_all_datasets(DATASET_SPECS)

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    write_summary_table(combined_df, output_dir / "consolidated_fairness_metrics.csv")
    plot_disparate_impact(combined_df, output_dir / "disparate_impact_overview.png")
    plot_gap_panels(combined_df, output_dir / "fairness_gap_panels.png")
    plot_bias_matrix(combined_df, output_dir / "bias_flag_matrix.png")

    LOGGER.info("Consolidated visuals saved to %s", output_dir)


if __name__ == "__main__":
    main()

