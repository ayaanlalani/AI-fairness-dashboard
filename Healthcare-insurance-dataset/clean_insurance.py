"""
clean_insurance.py

A self-contained script to clean, validate, preprocess, and export the Kaggle Healthcare Cost Prediction dataset (insurance.csv).

- Protected attributes: sex, age, smoker, region
- Outcome variable: charges (continuous)

Example run:

python clean_insurance.py \
  --input_path insurance.csv \
  --out_dir processed \
  --seed 42 \
  --cap_outliers True \
  --make_high_cost_label True \
  --test_size 0.2
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -------------------------
# CLI utilities
# -------------------------

def parse_bool(value: str) -> bool:
    """Robust boolean flag parser for argparse.

    Accepts common representations of truthy/falsey values.
    """
    true_values = {"true", "t", "1", "yes", "y"}
    false_values = {"false", "f", "0", "no", "n"}
    val = str(value).strip().lower()
    if val in true_values:
        return True
    if val in false_values:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# -------------------------
# Constants
# -------------------------

EXPECTED_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges",
]

CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]
NUMERIC_COLUMNS = ["age", "bmi", "children"]
TARGET_COLUMN = "charges"
AGE_BUCKET_COLUMN = "age_bucket"

VALID_SEX = {"male", "female"}
VALID_SMOKER = {"yes", "no"}
VALID_REGION = {"southwest", "southeast", "northwest", "northeast"}

AGE_BUCKET_BINS = [-np.inf, 30, 50, np.inf]
AGE_BUCKET_LABELS = ["lt_30", "30_50", "gt_50"]


# -------------------------
# Logging setup
# -------------------------

def get_logger() -> logging.Logger:
    logger = logging.getLogger("clean_insurance")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# -------------------------
# Helpers
# -------------------------

def normalize_categorical_text(s: pd.Series) -> pd.Series:
    """Lowercase, strip, and convert to snake_case-like representation."""
    s = s.astype(str).str.strip().str.lower()
    # Replace spaces/hyphens with underscores
    s = s.str.replace("-", "_", regex=False)
    s = s.str.replace(" ", "_", regex=False)
    # Treat common missing tokens as NaN
    s = s.replace({"", "nan", "none", "null"}, np.nan)
    return s


def coerce_to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series to numeric, invalid parsing becomes NaN."""
    return pd.to_numeric(series, errors="coerce")


def compute_iqr_bounds(values: pd.Series) -> Tuple[float, float]:
    """Compute IQR-based capping bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR)."""
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(lower), float(upper)


def cap_series_with_bounds(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Cap a Series to [lower, upper] bounds, preserving dtype as float."""
    capped = series.astype(float).clip(lower=lower, upper=upper)
    return capped


# -------------------------
# Core steps
# -------------------------

def load_and_validate(input_path: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV, normalize types and categories, validate, and report missingness.

    Returns: (df, missingness_table)
    """
    logger.info(f"Reading CSV from: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Validate columns
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Normalize categorical text
    for col in CATEGORICAL_COLUMNS:
        df[col] = normalize_categorical_text(df[col])

    # Coerce numerics
    for col in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        df[col] = coerce_to_numeric(df[col])

    # Validate categorical memberships; invalid values -> NaN
    def validate_membership(series: pd.Series, valid_set: set, name: str) -> pd.Series:
        invalid_mask = ~series.isna() & ~series.isin(valid_set)
        if invalid_mask.any():
            invalid_vals = series[invalid_mask].unique()
            logger.warning(f"Column '{name}' has invalid categories {invalid_vals}; setting to NaN for imputation.")
            series.loc[invalid_mask] = np.nan
        return series

    df["sex"] = validate_membership(df["sex"], VALID_SEX, "sex")
    df["smoker"] = validate_membership(df["smoker"], VALID_SMOKER, "smoker")
    df["region"] = validate_membership(df["region"], VALID_REGION, "region")

    # Validate numeric ranges; out-of-range -> NaN for imputation
    def enforce_range(series: pd.Series, condition: pd.Series, name: str) -> pd.Series:
        invalid_mask = ~series.isna() & ~condition
        if invalid_mask.any():
            n_invalid = int(invalid_mask.sum())
            logger.warning(f"Column '{name}' has {n_invalid} invalid values (range); setting to NaN for imputation.")
            series.loc[invalid_mask] = np.nan
        return series

    df["age"] = enforce_range(df["age"], df["age"] >= 18, "age")
    df["bmi"] = enforce_range(df["bmi"], df["bmi"] > 0, "bmi")
    df["children"] = enforce_range(df["children"], df["children"] >= 0, "children")
    df["charges"] = enforce_range(df["charges"], df["charges"] >= 0, "charges")

    # Dtype checks: age and children should be integer-like. If fractional values appear, round and log.
    for col in ["age", "children"]:
        series = df[col]
        frac_mask = (~series.isna()) & ((series % 1) != 0)
        if frac_mask.any():
            n_frac = int(frac_mask.sum())
            logger.warning(
                f"Column '{col}' has {n_frac} non-integer values; rounding to nearest integer."
            )
            df.loc[frac_mask, col] = np.round(series.loc[frac_mask])

    # Age bucket for fairness analysis (used as categorical feature)
    df[AGE_BUCKET_COLUMN] = pd.cut(
        df["age"], bins=AGE_BUCKET_BINS, labels=AGE_BUCKET_LABELS, right=True
    )
    df[AGE_BUCKET_COLUMN] = df[AGE_BUCKET_COLUMN].astype(object)

    # Shape and missingness summary
    logger.info(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df)) * 100.0
    missingness_table = pd.DataFrame({
        "column": df.columns,
        "n_missing": [int(missing_counts[c]) for c in df.columns],
        "pct_missing": [float(missing_pct[c]) for c in df.columns],
    })
    logger.info("Missing values per column (top 10 shown):\n" + missingness_table.head(10).to_string(index=False))

    return df, missingness_table


def split_data(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
    make_high_cost_label: bool,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train and test, with optional provisional stratification by high_cost.

    Uses a provisional high_cost label based on the global median for stratification only.
    """
    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [AGE_BUCKET_COLUMN]].copy()
    y = df[TARGET_COLUMN].copy()

    stratify_vec: Optional[pd.Series] = None
    if make_high_cost_label:
        global_median = float(np.nanmedian(y))
        stratify_vec = (y > global_median).astype(int)
        logger.info(f"Using provisional high_cost stratify with global median charges={global_median:,.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify_vec
    )

    logger.info(
        f"Split complete: train={len(X_train)} rows, test={len(X_test)} rows, test_size={test_size}"
    )
    return X_train, X_test, y_train, y_test


def build_pipeline() -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build a ColumnTransformer-based pipeline for features.

    Returns: (preprocessor, numeric_features, categorical_features)
    """
    numeric_features = NUMERIC_COLUMNS.copy()
    categorical_features = CATEGORICAL_COLUMNS.copy() + [AGE_BUCKET_COLUMN]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Build a version-tolerant OneHotEncoder (handles sklearn versions with either sparse_output or sparse)
    def make_one_hot_encoder() -> OneHotEncoder:
        try:
            # Newer sklearn versions
            return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # Older sklearn versions
            return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def get_output_feature_names(
    preprocessor: ColumnTransformer,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    """Extract output feature names from a fitted ColumnTransformer."""
    # Numeric names: same as input
    numeric_output_names = numeric_features

    # Categorical OHE names
    cat_pipeline: Pipeline = preprocessor.named_transformers_["categorical"]
    ohe: OneHotEncoder = cat_pipeline.named_steps["onehot"]
    # For sklearn >=1.0, get_feature_names_out supports input_features
    ohe_feature_names = list(ohe.get_feature_names_out(categorical_features))

    # Combine, preserving order: numeric then categorical
    combined = numeric_output_names + ohe_feature_names

    # Clean up any prefixes if present (usually not needed here)
    cleaned = [name.replace("categorical__", "").replace("numeric__", "") for name in combined]
    return cleaned


def fit_transform_export(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    out_dir: Path,
    seed: int,
    cap_outliers: bool,
    make_high_cost_label: bool,
    logger: logging.Logger,
) -> None:
    """Fit the pipeline, transform data, perform optional outlier capping, derive labels, and export outputs."""
    rng = np.random.RandomState(seed)

    # Optional: Outlier capping bounds for 'bmi' and 'charges' using TRAIN ONLY
    bmi_bounds: Optional[Tuple[float, float]] = None
    charges_bounds: Optional[Tuple[float, float]] = None
    if cap_outliers:
        bmi_bounds = compute_iqr_bounds(X_train["bmi"])  # type: ignore[index]
        charges_bounds = compute_iqr_bounds(y_train)
        logger.info(
            f"IQR capping bounds (train): bmi in [{bmi_bounds[0]:.3f}, {bmi_bounds[1]:.3f}], "
            f"charges in [{charges_bounds[0]:.2f}, {charges_bounds[1]:.2f}]"
        )
        # Apply to train and test
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train["bmi"] = cap_series_with_bounds(X_train["bmi"], *bmi_bounds)
        X_test["bmi"] = cap_series_with_bounds(X_test["bmi"], *bmi_bounds)
        y_train = cap_series_with_bounds(y_train, *charges_bounds)
        y_test = cap_series_with_bounds(y_test, *charges_bounds)
    else:
        logger.info("Outlier capping disabled.")

    # Build and fit preprocessor on TRAIN only
    preprocessor, numeric_features, categorical_features = build_pipeline()
    logger.info("Fitting preprocessing pipeline on train data...")
    preprocessor.fit(X_train)

    # Extract imputation stats for report
    num_imputer: SimpleImputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    cat_imputer: SimpleImputer = preprocessor.named_transformers_["categorical"].named_steps["imputer"]
    numeric_impute_stats = {
        feature: float(stat) for feature, stat in zip(numeric_features, num_imputer.statistics_)
    }
    categorical_impute_stats = {
        feature: (stat if isinstance(stat, (str, int, float)) else str(stat))
        for feature, stat in zip(categorical_features, cat_imputer.statistics_)
    }

    # Transform train/test
    logger.info("Transforming train and test features...")
    X_train_processed_arr = preprocessor.transform(X_train)
    X_test_processed_arr = preprocessor.transform(X_test)

    # Feature names
    feature_names = get_output_feature_names(preprocessor, numeric_features, categorical_features)

    # Convert to DataFrame with names
    X_train_processed = pd.DataFrame(X_train_processed_arr, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed_arr, columns=feature_names, index=X_test.index)

    # Derive high_cost labels using TRAIN median of CHARGES
    high_cost_train: Optional[pd.Series] = None
    high_cost_test: Optional[pd.Series] = None
    train_median_for_high_cost: Optional[float] = None
    if make_high_cost_label:
        train_median_for_high_cost = float(np.nanmedian(y_train))
        high_cost_train = (y_train > train_median_for_high_cost).astype(int)
        high_cost_test = (y_test > train_median_for_high_cost).astype(int)
        logger.info(f"Derived high_cost using train median charges={train_median_for_high_cost:,.2f}")
    else:
        logger.info("Skipping high_cost label generation.")

    # Assertions: No NaNs in processed features
    assert not np.isnan(X_train_processed.values).any(), "NaNs found in X_train_processed"
    assert not np.isnan(X_test_processed.values).any(), "NaNs found in X_test_processed"

    # Encoded categorical columns should be 0/1 only
    # Identify OHE columns: those not in numeric_features
    ohe_cols = [c for c in feature_names if c not in numeric_features]
    if ohe_cols:
        unique_vals = np.unique(np.concatenate([X_train_processed[ohe_cols].values.ravel(), X_test_processed[ohe_cols].values.ravel()]))
        allowed = {0.0, 1.0, 0, 1}
        assert set(np.unique(unique_vals)).issubset(allowed), "Categorical encoded columns contain values other than 0/1."

    # Ensure train/test columns identical and same order
    assert list(X_train_processed.columns) == list(X_test_processed.columns), "Train/Test columns mismatch"

    # Create out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exports
    x_train_path = out_dir / "X_train_processed.csv"
    x_test_path = out_dir / "X_test_processed.csv"
    y_train_path = out_dir / "y_train.csv"
    y_test_path = out_dir / "y_test.csv"

    X_train_processed.to_csv(x_train_path, index=False)
    X_test_processed.to_csv(x_test_path, index=False)

    pd.DataFrame({TARGET_COLUMN: y_train}).to_csv(y_train_path, index=False)
    pd.DataFrame({TARGET_COLUMN: y_test}).to_csv(y_test_path, index=False)

    if make_high_cost_label and high_cost_train is not None and high_cost_test is not None:
        pd.DataFrame({"high_cost": high_cost_train}).to_csv(out_dir / "high_cost_train.csv", index=False)
        pd.DataFrame({"high_cost": high_cost_test}).to_csv(out_dir / "high_cost_test.csv", index=False)

    # Feature map
    feature_map_path = out_dir / "feature_map.json"
    with feature_map_path.open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Save fitted transformers and metadata
    transformers_artifact = {
        "feature_pipeline": preprocessor,
        "bmi_cap_bounds": {"lower": bmi_bounds[0], "upper": bmi_bounds[1]} if bmi_bounds else None,
        "charges_cap_bounds": {"lower": charges_bounds[0], "upper": charges_bounds[1]} if charges_bounds else None,
        "train_median_for_high_cost": train_median_for_high_cost,
        "numeric_imputer_statistics": numeric_impute_stats,
        "categorical_imputer_statistics": categorical_impute_stats,
        "feature_names": feature_names,
    }
    joblib.dump(transformers_artifact, out_dir / "fitted_transformers.joblib")

    # Build data report
    # Counts
    n_train = len(X_train_processed)
    n_test = len(X_test_processed)
    n_features_final = len(feature_names)

    # Feature counts per block
    # OneHotEncoder categories per each categorical feature (drop='first') => n_cat - 1
    cat_pipeline: Pipeline = preprocessor.named_transformers_["categorical"]
    ohe: OneHotEncoder = cat_pipeline.named_steps["onehot"]
    categories = ohe.categories_
    cat_feature_counts = {
        feature: int(len(cats) - 1) for feature, cats in zip(categorical_features, categories)
    }

    # Compose Markdown report
    report_lines: List[str] = []
    report_lines.append("# Data Report: insurance.csv")
    report_lines.append("")
    report_lines.append("## Row counts")
    report_lines.append(f"- Total rows: {n_train + n_test}")
    report_lines.append(f"- Train rows: {n_train}")
    report_lines.append(f"- Test rows: {n_test}")
    report_lines.append("")
    report_lines.append("")
    report_lines.append("## Imputation statistics")
    report_lines.append("### Numeric (median)")
    for k, v in numeric_impute_stats.items():
        report_lines.append(f"- {k}: {v}")
    report_lines.append("### Categorical (most frequent)")
    for k, v in categorical_impute_stats.items():
        report_lines.append(f"- {k}: {v}")
    report_lines.append("")

    report_lines.append("## Outlier capping bounds")
    if cap_outliers:
        report_lines.append(f"- bmi: lower={bmi_bounds[0]:.3f}, upper={bmi_bounds[1]:.3f}")
        report_lines.append(f"- charges: lower={charges_bounds[0]:.2f}, upper={charges_bounds[1]:.2f}")
    else:
        report_lines.append("- Outlier capping disabled")
    report_lines.append("")

    report_lines.append("## Train median for high_cost")
    report_lines.append(
        f"- train_median_charges: {train_median_for_high_cost:,.2f}" if train_median_for_high_cost is not None else "- high_cost label not created"
    )
    report_lines.append("")

    report_lines.append("## Final feature counts")
    report_lines.append(f"- Numeric features (scaled): {len(numeric_features)}")
    report_lines.append("- One-hot encoded categorical features:")
    for feat, cnt in cat_feature_counts.items():
        report_lines.append(f"  - {feat}: {cnt}")
    report_lines.append(f"- Total features after transform: {n_features_final}")
    report_lines.append("")

    # Placeholder for top 5 feature names
    example_features = feature_names[:5]
    report_lines.append("## Example feature names")
    report_lines.append("- " + ", ".join(example_features))
    report_lines.append("")

    # Write report header/body now; missingness table will be appended by main
    report_path = out_dir / "data_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Acceptance criteria checks (files exist and readable, feature_map length matches)
    # Reload processed to ensure readable
    df_xtr = pd.read_csv(x_train_path)
    df_xte = pd.read_csv(x_test_path)
    assert len(json.load(open(feature_map_path))) == df_xtr.shape[1], (
        "feature_map.json length must equal processed column count"
    )
    assert not df_xtr.isna().any().any(), "NaNs present in X_train_processed.csv"
    assert not df_xte.isna().any().any(), "NaNs present in X_test_processed.csv"

    # Final summary
    logger.info(
        f"Processing complete: n_train={n_train}, n_test={n_test}, n_features_final={n_features_final}. "
        f"Example features: {example_features}"
    )


def append_missingness_to_report(
    report_path: Path, missingness_table: pd.DataFrame
) -> None:
    """Append the missingness table (markdown) to the report file."""
    lines: List[str] = []
    lines.append("## Missingness table (pre-imputation)")
    lines.append("")
    lines.append("| column | n_missing | pct_missing |")
    lines.append("|---|---:|---:|")
    for _, row in missingness_table.iterrows():
        lines.append(
            f"| {row['column']} | {int(row['n_missing'])} | {row['pct_missing']:.2f} |"
        )
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(lines) + "\n")


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and preprocess insurance.csv dataset")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input insurance.csv")
    parser.add_argument("--out_dir", type=str, default="processed", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cap_outliers",
        type=parse_bool,
        default=True,
        help="Whether to apply IQR capping to bmi and charges on train and apply to test",
    )
    parser.add_argument(
        "--make_high_cost_label",
        type=parse_bool,
        default=True,
        help="Whether to derive binary high_cost label based on train median charges",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test size fraction for train/test split"
    )

    args = parser.parse_args()

    logger = get_logger()

    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)

    logger.info(
        "Starting cleaning and preprocessing with arguments: "
        f"input_path={input_path}, out_dir={out_dir}, seed={args.seed}, "
        f"cap_outliers={args.cap_outliers}, make_high_cost_label={args.make_high_cost_label}, "
        f"test_size={args.test_size}"
    )

    # Load and validate
    df, missingness_table = load_and_validate(input_path, logger)

    # Split
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        seed=args.seed,
        test_size=args.test_size,
        make_high_cost_label=args.make_high_cost_label,
        logger=logger,
    )

    # Fit/transform/export
    fit_transform_export(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        out_dir=out_dir,
        seed=args.seed,
        cap_outliers=args.cap_outliers,
        make_high_cost_label=args.make_high_cost_label,
        logger=logger,
    )

    # Append missingness table to report
    append_missingness_to_report(out_dir / "data_report.md", missingness_table)

    # Final acceptance checks on artifacts
    required_files = [
        out_dir / "X_train_processed.csv",
        out_dir / "X_test_processed.csv",
        out_dir / "y_train.csv",
        out_dir / "y_test.csv",
        out_dir / "feature_map.json",
        out_dir / "fitted_transformers.joblib",
        out_dir / "data_report.md",
    ]
    for fpath in required_files:
        assert fpath.exists(), f"Required output missing: {fpath}"

    # If high_cost enabled, ensure files exist
    if args.make_high_cost_label:
        hc_train = out_dir / "high_cost_train.csv"
        hc_test = out_dir / "high_cost_test.csv"
        assert hc_train.exists() and hc_test.exists(), "Missing high_cost label exports"

    logger.info("All acceptance criteria satisfied.")


if __name__ == "__main__":
    main()
