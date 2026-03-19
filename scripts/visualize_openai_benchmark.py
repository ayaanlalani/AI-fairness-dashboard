"""
visualize_openai_benchmark.py

Generate OpenAI-vs-deterministic benchmark visuals from artifacts/.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ARTIFACTS = ROOT / "artifacts"
VIZ_DIR = ARTIFACTS / "visualizations"

DATASETS = {
    "german_credit": {"label": "German Credit", "color": "#2563EB"},
    "hmda": {"label": "HMDA (Georgia)", "color": "#D97706"},
}

SUBSCORE_KEYS = [
    "completeness",
    "severity_agreement",
    "cause_alignment",
    "mitigation_specificity",
    "research_grounding",
]


def _normalize_severity(value: str) -> str:
    value = (value or "").upper()
    for label in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        if label in value:
            return label
    return value.strip() or "UNKNOWN"


def _load_usage(openai_dir: Path, raw: dict) -> dict:
    usage = dict(raw.get("usage", {}))
    napkin_path = openai_dir / "llm_napkin_math.json"
    if napkin_path.exists():
        napkin_usage = json.loads(napkin_path.read_text(encoding="utf-8")).get("usage", {})
        usage = {**napkin_usage, **usage}
    return usage


def load_data() -> dict[str, dict]:
    data: dict[str, dict] = {}
    for ds_key, meta in DATASETS.items():
        base = ARTIFACTS / ds_key / "fairness"
        openai_dir = base / "openai"
        raw_path = openai_dir / "llm_raw_response.json"
        ctx_path = openai_dir / "llm_context_payload.json"
        baseline_path = openai_dir / "llm_baseline_payload.json"
        if not raw_path.exists() or not ctx_path.exists() or not baseline_path.exists():
            log.warning("Missing OpenAI artifacts for %s; skipping", ds_key)
            continue

        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        context = json.loads(ctx_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        data[ds_key] = {
            "label": meta["label"],
            "color": meta["color"],
            "result": raw["result"],
            "cycles": raw["cycles"],
            "usage": _load_usage(openai_dir, raw),
            "context": context,
            "baseline": baseline,
        }
    return data


def plot_cycle_scores(data: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds in data.values():
        xs = [item["cycle"] for item in ds["cycles"]]
        ys = [item["score"]["total_score"] for item in ds["cycles"]]
        ax.plot(xs, ys, marker="o", linewidth=2, label=ds["label"], color=ds["color"])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.8, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

    xticks = sorted({item["cycle"] for ds in data.values() for item in ds["cycles"]})
    ax.set_xticks(xticks)
    ax.set_xlabel("Refinement Cycle")
    ax.set_ylabel("Benchmark Score / 100")
    ax.set_title("OpenAI Benchmark: Cycle Scores", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "openai_cycle_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved openai_cycle_scores.png")


def plot_final_subscores(data: dict[str, dict]) -> None:
    labels = [data[k]["label"] for k in data]
    x = np.arange(len(labels))
    width = 0.14

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, key in enumerate(SUBSCORE_KEYS):
        vals = [data[k]["cycles"][-1]["score"]["subscores"][key] for k in data]
        ax.bar(x + (idx - 2) * width, vals, width, label=key.replace("_", " ").title())

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Subscore")
    ax.set_title("OpenAI Final Subscores vs Deterministic Baseline", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "openai_final_subscores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved openai_final_subscores.png")


def plot_severity_agreement(data: dict[str, dict]) -> None:
    rows = []
    for ds in data.values():
        baseline = {item["attribute"]: item["severity"] for item in ds["baseline"]["attributes"]}
        agree = 0
        total = 0
        for item in ds["result"]["qualitative"]:
            total += 1
            if _normalize_severity(item["severity"]) == _normalize_severity(baseline.get(item["attribute"], "")):
                agree += 1
        rows.append({"Dataset": ds["label"], "Agree": agree, "Disagree": total - agree})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["Dataset"], df["Agree"], label="Agreement", color="#22C55E")
    ax.bar(df["Dataset"], df["Disagree"], bottom=df["Agree"], label="Disagreement", color="#DC2626")
    ax.set_ylabel("Attribute Count")
    ax.set_title("OpenAI Severity Agreement with Deterministic Model", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "openai_severity_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved openai_severity_agreement.png")


def plot_cost_latency(data: dict[str, dict]) -> None:
    rows = []
    for ds in data.values():
        usage = ds["usage"]
        latency = usage.get("stage_timings_s", {}).get("api_call", usage.get("wall_clock_s", 0.0))
        rows.append({
            "Dataset": ds["label"],
            "Cost": usage.get("total_cost_usd", 0.0),
            "Latency": latency,
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(df["Dataset"], df["Cost"], color=["#2563EB", "#D97706"])
    axes[0].set_ylabel("USD")
    axes[0].set_title("OpenAI Cost", fontweight="bold")
    for bar, cost in zip(axes[0].patches, df["Cost"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"${cost:.4f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(df["Dataset"], df["Latency"], color=["#2563EB", "#D97706"])
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("OpenAI API Latency", fontweight="bold")
    for bar, latency in zip(axes[1].patches, df["Latency"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{latency:.1f}s", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(VIZ_DIR / "openai_cost_latency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved openai_cost_latency.png")


def plot_di_context(data: dict[str, dict]) -> None:
    rows = []
    for ds in data.values():
        final_qual = {item["attribute"]: item["severity"] for item in ds["result"]["qualitative"]}
        for attr in ds["context"]["attributes"]:
            rows.append({
                "Label": f"{ds['label']}\n{attr['attribute'].replace('_original', '')}",
                "DI": attr["metrics"]["disparate_impact"],
                "OpenAI Severity": _normalize_severity(final_qual.get(attr["attribute"], "")),
            })

    df = pd.DataFrame(rows)
    colors = {
        "CRITICAL": "#DC2626",
        "HIGH": "#EA580C",
        "MODERATE": "#F59E0B",
        "LOW": "#22C55E",
    }
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.5), 5))
    bars = ax.bar(df["Label"], df["DI"], color=[colors.get(s, "#6B7280") for s in df["OpenAI Severity"]])
    ax.axhline(0.8, color="red", linestyle="--", linewidth=1.2, label="Bias threshold (DI=0.8)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, label="Parity (DI=1.0)")
    ax.set_ylabel("Deterministic Disparate Impact")
    ax.set_title("Deterministic DI Context with OpenAI Severity Labels", fontweight="bold")
    ax.legend(fontsize=8)
    for bar, di, sev in zip(bars, df["DI"], df["OpenAI Severity"]):
        ax.text(bar.get_x() + bar.get_width() / 2, di, f"{di:.3f}\n{sev}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "openai_di_context.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved openai_di_context.png")


def main() -> None:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    if not data:
        log.error("No OpenAI artifact data found under artifacts/")
        return

    plot_cycle_scores(data)
    plot_final_subscores(data)
    plot_severity_agreement(data)
    plot_cost_latency(data)
    plot_di_context(data)


if __name__ == "__main__":
    main()
