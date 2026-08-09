"""
clean_hmda.py

Preprocess the raw HMDA download into a clean dataset for modelling
and fairness analysis.

Target variable:
  approved  (1 = loan originated, 0 = application denied)

Protected attributes (real, legally protected):
  race      – derived_race  (White / Black / Asian / Other)
  sex       – derived_sex   (Male / Female)
  age_group – applicant_age (young <35, mid 35-54, senior 55+)

Features for the model:
  loan_amount, income, debt_to_income_ratio, loan_type,
  loan_purpose, property_type, occupancy_type, interest_rate,
  loan_term, combined_loan_to_value_ratio

Usage:
  cd hmda_dataset
  python3 clean_hmda.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── column definitions ─────────────────────────────────────────────────
RAW_COLS_NEEDED = [
    "action_taken",
    "derived_race",
    "derived_sex",
    "derived_ethnicity",
    "applicant_age",
    "applicant_age_above_62",
    "loan_amount",
    "income",
    "debt_to_income_ratio",
    "loan_type",
    "loan_purpose",
    "property_type",
    "occupancy_type",
    "interest_rate",
    "loan_term",
    "combined_loan_to_value_ratio",
]

# ── TARGET LEAKAGE: interest_rate is deliberately NOT a feature ────────
#
# interest_rate is populated only where a loan was actually originated. In
# hmda_raw.csv it is missing for 100.0% of denials (action_taken=3, n=5226) and
# 2.4% of originations (action_taken=1, n=14774). Median-imputing it and handing
# it to a classifier leaks the target: the model learns "is interest_rate
# observed", which is the outcome. With it included the random forest scored
# AUC 0.9942 -- not a plausible mortgage approval model, and the near-zero
# EOD/AOD it produced were a property of a model that makes almost no errors in
# any group rather than evidence about fairness.
#
# It stays in RAW_COLS_NEEDED so the missingness pattern remains auditable, and
# is dropped before feature assembly. Do not re-add it to NUMERIC_FEATURES.
LEAKED_COLUMNS = ["interest_rate"]

NUMERIC_FEATURES = [
    "loan_amount",
    "income",
    "loan_term",
    "combined_loan_to_value_ratio",
    "dti_numeric",
]
CATEGORICAL_FEATURES = [
    "loan_type",
    "loan_purpose",
    "property_type",
    "occupancy_type",
]

# age_group is the banded view (young/mid/senior). age_62_plus is the cut ECOA
# actually specifies: Regulation B Sec. 1002.2(o) defines "elderly" as 62 or
# older. The commonly used 40+ threshold comes from the ADEA, an employment
# statute, and does not apply to credit. Both are audited: the contrast is a
# finding, because pooling young with senior against mid cancels the two groups
# against each other and returns a clean result on the only age class the
# statute protects.
PROTECTED_ATTRS = ["race", "sex", "age_group", "age_62_plus"]

TARGET = "approved"

# ── race collapsing ────────────────────────────────────────────────────
RACE_MAP = {
    "White": "White",
    "Black or African American": "Black",
    "Asian": "Asian",
    "American Indian or Alaska Native": "American Indian",
    "Native Hawaiian or Other Pacific Islander": "Pacific Islander",
    "2 or more minority races": "Multiracial",
    "Joint": None,       # exclude (co-applicant pair, not a single race)
    "Free Form Text Only": None,
    "Race Not Available": None,
}

SEX_KEEP = {"Male", "Female"}

AGE_BUCKETS = {
    "<25": "young",
    "25-34": "young",
    "35-44": "mid",
    "45-54": "mid",
    "55-64": "senior",
    "65-74": "senior",
    ">74": "senior",
}


def load_and_clean(input_path: Path) -> pd.DataFrame:
    log.info(f"Loading {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    log.info(f"Raw shape: {df.shape}")

    # Keep only needed columns (those that exist)
    available = [c for c in RAW_COLS_NEEDED if c in df.columns]
    missing = set(RAW_COLS_NEEDED) - set(available)
    if missing:
        log.warning(f"Columns not found (will skip): {missing}")
    df = df[available].copy()

    # ── target: action_taken → approved (binary) ──────────────────
    df[TARGET] = (df["action_taken"] == 1).astype(int)
    df.drop(columns=["action_taken"], inplace=True)

    # ── race ──────────────────────────────────────────────────────
    if "derived_race" in df.columns:
        df["race"] = df["derived_race"].map(RACE_MAP)
        df = df.dropna(subset=["race"])
        df.drop(columns=["derived_race"], inplace=True)
    else:
        log.warning("derived_race missing; race attribute unavailable")

    # ── sex ───────────────────────────────────────────────────────
    if "derived_sex" in df.columns:
        df = df[df["derived_sex"].isin(SEX_KEEP)].copy()
        df.rename(columns={"derived_sex": "sex"}, inplace=True)
    else:
        log.warning("derived_sex missing; sex attribute unavailable")

    # ── ethnicity (keep as feature, not protected attr for now) ──
    if "derived_ethnicity" in df.columns:
        df.drop(columns=["derived_ethnicity"], inplace=True)

    # ── age group ─────────────────────────────────────────────────
    if "applicant_age" in df.columns:
        df["age_group"] = df["applicant_age"].map(AGE_BUCKETS)
        df = df.dropna(subset=["age_group"])
        df.drop(columns=["applicant_age"], inplace=True)
    else:
        log.warning("applicant_age missing; age attribute unavailable")

    # ── ECOA-correct age cut ──────────────────────────────────────
    # The public LAR age bands straddle 62 (the "55-64" band cannot be split),
    # so the banded age_group cannot isolate the protected class. HMDA supplies
    # applicant_age_above_62 precisely because Reg B Sec. 1002.2(o) turns on it.
    # Rows where the flag is blank are dropped from this attribute only (set to
    # NA), not from the dataset, so age_group and race/sex keep their full n.
    if "applicant_age_above_62" in df.columns:
        mapped = df["applicant_age_above_62"].astype(str).str.strip().str.lower()
        df["age_62_plus"] = mapped.map({"yes": "62_plus", "no": "under_62"})
        n_unknown = int(df["age_62_plus"].isna().sum())
        if n_unknown:
            log.info(f"age_62_plus unknown for {n_unknown} rows (flag blank in source)")
        df.drop(columns=["applicant_age_above_62"], inplace=True)
    else:
        log.warning("applicant_age_above_62 missing; ECOA age cut unavailable")

    # ── drop leaked columns before feature assembly ───────────────
    # action_taken has already been converted to TARGET and dropped above, so
    # the missingness audit groups by the target itself.
    leaked_present = [c for c in LEAKED_COLUMNS if c in df.columns]
    if leaked_present:
        for col in leaked_present:
            by_outcome = df.groupby(TARGET)[col].apply(lambda s: s.isna().mean())
            log.info(
                f"dropping leaked column '{col}'; missingness by {TARGET}: "
                + ", ".join(
                    f"{'approved' if k == 1 else 'denied'}={v:.1%}"
                    for k, v in by_outcome.items()
                )
            )
        df.drop(columns=leaked_present, inplace=True)

    # ── debt-to-income: convert ranges to midpoints ───────────────
    if "debt_to_income_ratio" in df.columns:
        dti = df["debt_to_income_ratio"].astype(str)
        dti_numeric = pd.to_numeric(dti, errors="coerce")

        range_map = {
            "<20%": 15.0,
            "20%-<30%": 25.0,
            "30%-<36%": 33.0,
            "50%-60%": 55.0,
            ">60%": 65.0,
        }
        for pattern, val in range_map.items():
            mask = dti == pattern
            dti_numeric[mask] = val

        df["dti_numeric"] = dti_numeric
        df.drop(columns=["debt_to_income_ratio"], inplace=True)
    else:
        df["dti_numeric"] = np.nan

    # ── coerce remaining numerics ─────────────────────────────────
    for col in ["loan_amount", "income", "interest_rate", "loan_term",
                "combined_loan_to_value_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── coerce categoricals to str ────────────────────────────────
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": np.nan, "": np.nan})

    # ── drop rows missing the target or all features ──────────────
    df = df.dropna(subset=[TARGET])

    log.info(f"Cleaned shape: {df.shape}")
    log.info(f"Approval rate: {df[TARGET].mean():.3f}")
    for attr in PROTECTED_ATTRS:
        if attr in df.columns:
            log.info(f"  {attr}: {df[attr].value_counts().to_dict()}")

    return df


def preprocess_and_split(
    df: pd.DataFrame,
    out_dir: Path,
    seed: int = 42,
    test_size: float = 0.2,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Separate features, target, protected attrs
    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    prot_cols = [c for c in PROTECTED_ATTRS if c in df.columns]

    X = df[feature_cols + prot_cols].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    log.info(f"Split: train={len(X_train)}, test={len(X_test)}")

    # Build preprocessing pipeline (numerics + categoricals only, not protected)
    num_feats = [c for c in NUMERIC_FEATURES if c in X_train.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    def _ohe():
        try:
            return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _ohe()),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats),
    ])

    preprocessor.fit(X_train[num_feats + cat_feats])

    X_train_proc = preprocessor.transform(X_train[num_feats + cat_feats])
    X_test_proc = preprocessor.transform(X_test[num_feats + cat_feats])

    # Feature names
    ohe_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(cat_feats)
    )
    feature_names = num_feats + ohe_names

    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)

    # Append protected attrs (unprocessed) for fairness analysis
    for attr in prot_cols:
        X_train_df[attr] = X_train[attr].values
        X_test_df[attr] = X_test[attr].values

    # ── export ─────────────────────────────────────────────────────
    X_train_df.to_csv(out_dir / "X_train_processed.csv", index=False)
    X_test_df.to_csv(out_dir / "X_test_processed.csv", index=False)
    pd.DataFrame({TARGET: y_train}).to_csv(out_dir / "y_train.csv", index=False)
    pd.DataFrame({TARGET: y_test}).to_csv(out_dir / "y_test.csv", index=False)

    with (out_dir / "feature_map.json").open("w") as f:
        json.dump(feature_names, f, indent=2)

    joblib.dump({
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "protected_attrs": prot_cols,
    }, out_dir / "fitted_transformers.joblib")

    # ── data report ────────────────────────────────────────────────
    report = [
        "# Data Report: HMDA Mortgage Lending",
        "",
        f"- Total rows (after cleaning): {len(df)}",
        f"- Train rows: {len(X_train_df)}",
        f"- Test rows: {len(X_test_df)}",
        f"- Approval rate: {y.mean():.3f}",
        f"- Features (processed): {len(feature_names)}",
        "",
        "## Protected Attributes (real, legally protected)",
        "",
    ]
    for attr in prot_cols:
        report.append(f"### {attr}")
        report.append(f"```\n{df[attr].value_counts().to_string()}\n```")
        report.append("")

    with (out_dir / "data_report.md").open("w") as f:
        f.write("\n".join(report) + "\n")

    log.info(f"Exported processed data to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hmda_raw.csv")
    parser.add_argument("--out_dir", default="processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_and_clean(Path(args.input))
    preprocess_and_split(df, Path(args.out_dir), seed=args.seed, test_size=args.test_size)


if __name__ == "__main__":
    main()
