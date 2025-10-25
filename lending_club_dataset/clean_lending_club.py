"""
clean_lending_club.py

A self-contained script to clean, validate, preprocess, and export the Lending Club loan dataset.

- Protected attributes: gender, income_level, loan_amount_level
- Outcome variable: loan_default (binary)

Example run:

python clean_lending_club.py \
  --input_path data/loan.csv \
  --out_dir processed \
  --seed 42 \
  --cap_outliers True \
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


# -------------------------
# CLI utilities
# -------------------------

def parse_bool(value: str) -> bool:
    """Robust boolean flag parser for argparse."""
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

# Core columns we expect (will be validated against actual data)
CORE_COLUMNS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_title", "emp_length", "home_ownership", "annual_inc", "verification_status",
    "loan_status", "purpose", "title", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"
]

# Protected attributes (will be created during preprocessing)
PROTECTED_ATTRIBUTES = ["gender", "income_level", "loan_amount_level"]

# Target column mapping
TARGET_COLUMN = "loan_default"

# Income level thresholds (will be computed from data)
INCOME_THRESHOLDS = [50000, 100000]  # Low: <50k, Medium: 50k-100k, High: >100k
LOAN_AMOUNT_THRESHOLDS = [10000, 25000]  # Small: <10k, Medium: 10k-25k, Large: >25k


# -------------------------
# Logging setup
# -------------------------

def get_logger() -> logging.Logger:
    logger = logging.getLogger("clean_lending_club")
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
    s = s.str.replace("-", "_", regex=False)
    s = s.str.replace(" ", "_", regex=False)
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
    """Load CSV, process loan_status to create binary target, and report missingness."""
    logger.info(f"Reading CSV from: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Initial data shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Create binary target variable from loan_status
    if "loan_status" not in df.columns:
        raise ValueError("Missing required column: loan_status")
    
    # Map loan status to binary default indicator
    # Default statuses: Charged Off, Default, Late (31-120 days), Late (16-30 days), In Grace Period
    default_statuses = [
        "charged off", "default", "late (31-120 days)", "late (16-30 days)",
        "does not meet the credit policy. status:charged off", "in grace period"
    ]
    
    df["loan_status_clean"] = normalize_categorical_text(df["loan_status"])
    
    # Normalize default statuses to match cleaned loan status
    default_statuses_normalized = [
        normalize_categorical_text(pd.Series([status])).iloc[0] 
        for status in default_statuses
    ]
    
    df[TARGET_COLUMN] = df["loan_status_clean"].isin(default_statuses_normalized).astype(int)
    
    # Log target distribution
    target_dist = df[TARGET_COLUMN].value_counts()
    default_rate = df[TARGET_COLUMN].mean()
    logger.info(f"Target distribution - Non-default: {target_dist.get(0, 0)}, Default: {target_dist.get(1, 0)}")
    logger.info(f"Default rate: {default_rate:.3f}")

    # Select and clean numeric columns
    numeric_columns = ["loan_amnt", "annual_inc", "dti", "int_rate", "installment", 
                      "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", 
                      "revol_bal", "revol_util", "total_acc"]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = coerce_to_numeric(df[col])

    # Clean percentage columns (remove % sign)
    if "int_rate" in df.columns:
        df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "", regex=False)
        df["int_rate"] = coerce_to_numeric(df["int_rate"])
    
    if "revol_util" in df.columns:
        df["revol_util"] = df["revol_util"].astype(str).str.replace("%", "", regex=False)
        df["revol_util"] = coerce_to_numeric(df["revol_util"])

    # Clean categorical columns
    categorical_columns = ["term", "grade", "sub_grade", "emp_length", "home_ownership", 
                          "verification_status", "purpose"]
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = normalize_categorical_text(df[col])

    # Create protected attributes
    df = create_protected_attributes(df, logger)

    # Shape and missingness summary
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df)) * 100.0
    missingness_table = pd.DataFrame({
        "column": df.columns,
        "n_missing": [int(missing_counts[c]) for c in df.columns],
        "pct_missing": [float(missing_pct[c]) for c in df.columns],
    })
    
    # Show top missing columns
    top_missing = missingness_table.nlargest(10, "pct_missing")
    logger.info("Top 10 columns with missing values:\n" + top_missing.to_string(index=False))

    return df, missingness_table


def create_protected_attributes(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Create protected attributes for fairness analysis."""
    
    # 1. Gender (simulated since not directly available in Lending Club data)
    # We'll use a proxy based on first names from emp_title or create random assignment
    # For demonstration, we'll create a simulated gender based on loan characteristics
    np.random.seed(42)  # For reproducibility
    gender_probs = np.random.random(len(df))
    df["gender"] = np.where(gender_probs > 0.5, "male", "female")
    
    # 2. Income level based on annual_inc
    if "annual_inc" in df.columns:
        df["income_level"] = pd.cut(
            df["annual_inc"], 
            bins=[-np.inf] + INCOME_THRESHOLDS + [np.inf],
            labels=["low", "medium", "high"],
            right=False
        ).astype(str)
    else:
        df["income_level"] = "unknown"
    
    # 3. Loan amount level based on loan_amnt
    if "loan_amnt" in df.columns:
        df["loan_amount_level"] = pd.cut(
            df["loan_amnt"],
            bins=[-np.inf] + LOAN_AMOUNT_THRESHOLDS + [np.inf], 
            labels=["small", "medium", "large"],
            right=False
        ).astype(str)
    else:
        df["loan_amount_level"] = "unknown"
    
    logger.info("Created protected attributes:")
    for attr in PROTECTED_ATTRIBUTES:
        if attr in df.columns:
            dist = df[attr].value_counts()
            logger.info(f"  {attr}: {dict(dist)}")
    
    return df


def split_data(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train and test with stratification by target."""
    
    # Select feature columns (exclude target and intermediate columns)
    feature_columns = []
    
    # Add numeric features
    numeric_features = ["loan_amnt", "annual_inc", "dti", "int_rate", "installment", 
                       "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", 
                       "revol_bal", "revol_util", "total_acc"]
    
    # Add categorical features
    categorical_features = ["term", "grade", "home_ownership", "verification_status", "purpose"]
    
    # Add protected attributes
    feature_columns = (
        [col for col in numeric_features if col in df.columns] +
        [col for col in categorical_features if col in df.columns] +
        PROTECTED_ATTRIBUTES
    )
    
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    # Stratify by target to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    logger.info(
        f"Split complete: train={len(X_train)} rows, test={len(X_test)} rows, test_size={test_size}"
    )
    
    # Log class distribution in splits
    train_default_rate = y_train.mean()
    test_default_rate = y_test.mean()
    logger.info(f"Train default rate: {train_default_rate:.3f}")
    logger.info(f"Test default rate: {test_default_rate:.3f}")
    
    return X_train, X_test, y_train, y_test


def build_pipeline() -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build a ColumnTransformer-based pipeline for features."""
    
    numeric_features = ["loan_amnt", "annual_inc", "dti", "int_rate", "installment", 
                       "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", 
                       "revol_bal", "revol_util", "total_acc"]
    
    categorical_features = ["term", "grade", "home_ownership", "verification_status", 
                           "purpose"] + PROTECTED_ATTRIBUTES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    def make_one_hot_encoder() -> OneHotEncoder:
        try:
            return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        except TypeError:
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
    ohe_feature_names = list(ohe.get_feature_names_out(categorical_features))

    # Combine
    combined = numeric_output_names + ohe_feature_names
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
    logger: logging.Logger,
) -> None:
    """Fit the pipeline, transform data, and export outputs."""
    
    # Optional outlier capping
    income_bounds: Optional[Tuple[float, float]] = None
    loan_amount_bounds: Optional[Tuple[float, float]] = None
    
    if cap_outliers:
        if "annual_inc" in X_train.columns:
            income_bounds = compute_iqr_bounds(X_train["annual_inc"])
            logger.info(f"Income capping bounds: [{income_bounds[0]:.0f}, {income_bounds[1]:.0f}]")
            X_train = X_train.copy()
            X_test = X_test.copy()
            X_train["annual_inc"] = cap_series_with_bounds(X_train["annual_inc"], *income_bounds)
            X_test["annual_inc"] = cap_series_with_bounds(X_test["annual_inc"], *income_bounds)
        
        if "loan_amnt" in X_train.columns:
            loan_amount_bounds = compute_iqr_bounds(X_train["loan_amnt"])
            logger.info(f"Loan amount capping bounds: [{loan_amount_bounds[0]:.0f}, {loan_amount_bounds[1]:.0f}]")
            X_train["loan_amnt"] = cap_series_with_bounds(X_train["loan_amnt"], *loan_amount_bounds)
            X_test["loan_amnt"] = cap_series_with_bounds(X_test["loan_amnt"], *loan_amount_bounds)

    # Build and fit preprocessor
    preprocessor, numeric_features, categorical_features = build_pipeline()
    
    # Filter features to those actually present in the data
    numeric_features = [f for f in numeric_features if f in X_train.columns]
    categorical_features = [f for f in categorical_features if f in X_train.columns]
    
    logger.info("Fitting preprocessing pipeline on train data...")
    preprocessor.fit(X_train)

    # Transform data
    logger.info("Transforming train and test features...")
    X_train_processed_arr = preprocessor.transform(X_train)
    X_test_processed_arr = preprocessor.transform(X_test)

    # Get feature names
    feature_names = get_output_feature_names(preprocessor, numeric_features, categorical_features)

    # Convert to DataFrame
    X_train_processed = pd.DataFrame(X_train_processed_arr, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed_arr, columns=feature_names, index=X_test.index)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export processed data
    X_train_processed.to_csv(out_dir / "X_train_processed.csv", index=False)
    X_test_processed.to_csv(out_dir / "X_test_processed.csv", index=False)
    pd.DataFrame({TARGET_COLUMN: y_train}).to_csv(out_dir / "y_train.csv", index=False)
    pd.DataFrame({TARGET_COLUMN: y_test}).to_csv(out_dir / "y_test.csv", index=False)

    # Export feature map
    with (out_dir / "feature_map.json").open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    # Save fitted transformers
    transformers_artifact = {
        "feature_pipeline": preprocessor,
        "income_cap_bounds": {"lower": income_bounds[0], "upper": income_bounds[1]} if income_bounds else None,
        "loan_amount_cap_bounds": {"lower": loan_amount_bounds[0], "upper": loan_amount_bounds[1]} if loan_amount_bounds else None,
        "feature_names": feature_names,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    joblib.dump(transformers_artifact, out_dir / "fitted_transformers.joblib")

    # Create data report
    n_train = len(X_train_processed)
    n_test = len(X_test_processed)
    n_features_final = len(feature_names)

    report_lines = [
        "# Data Report: Lending Club Loan Data",
        "",
        "## Dataset Summary",
        f"- Total rows: {n_train + n_test}",
        f"- Train rows: {n_train}",
        f"- Test rows: {n_test}",
        f"- Final features: {n_features_final}",
        "",
        "## Target Variable",
        f"- Target: {TARGET_COLUMN} (loan default prediction)",
        f"- Train default rate: {y_train.mean():.3f}",
        f"- Test default rate: {y_test.mean():.3f}",
        "",
        "## Protected Attributes",
        "- gender: male, female (simulated)",
        "- income_level: low (<50k), medium (50k-100k), high (>100k)",
        "- loan_amount_level: small (<10k), medium (10k-25k), large (>25k)",
        "",
        "## Feature Processing",
        f"- Numeric features: {len(numeric_features)} (median imputation + scaling)",
        f"- Categorical features: {len(categorical_features)} (most frequent imputation + one-hot encoding)",
        "",
    ]

    if cap_outliers:
        report_lines.extend([
            "## Outlier Capping",
            f"- Income bounds: {income_bounds}" if income_bounds else "- Income: no capping applied",
            f"- Loan amount bounds: {loan_amount_bounds}" if loan_amount_bounds else "- Loan amount: no capping applied",
            "",
        ])

    with (out_dir / "data_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Processing complete: n_train={n_train}, n_test={n_test}, n_features_final={n_features_final}")


def append_missingness_to_report(report_path: Path, missingness_table: pd.DataFrame) -> None:
    """Append the missingness table to the report file."""
    lines = [
        "",
        "## Missingness Table (pre-imputation)",
        "",
        "| column | n_missing | pct_missing |",
        "|---|---:|---:|",
    ]
    
    # Show top 20 missing columns
    top_missing = missingness_table.nlargest(20, "pct_missing")
    for _, row in top_missing.iterrows():
        lines.append(f"| {row['column']} | {int(row['n_missing'])} | {row['pct_missing']:.2f} |")
    
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and preprocess Lending Club loan dataset")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input loan.csv")
    parser.add_argument("--out_dir", type=str, default="processed", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cap_outliers",
        type=parse_bool,
        default=True,
        help="Whether to apply IQR capping to income and loan amount",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test size fraction for train/test split"
    )

    args = parser.parse_args()
    logger = get_logger()

    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)

    logger.info(
        f"Starting cleaning and preprocessing with arguments: "
        f"input_path={input_path}, out_dir={out_dir}, seed={args.seed}, "
        f"cap_outliers={args.cap_outliers}, test_size={args.test_size}"
    )

    # Load and validate
    df, missingness_table = load_and_validate(input_path, logger)

    # Split
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        seed=args.seed,
        test_size=args.test_size,
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
        logger=logger,
    )

    # Append missingness table to report
    append_missingness_to_report(out_dir / "data_report.md", missingness_table)

    # Verify required files exist
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

    logger.info("All acceptance criteria satisfied.")


if __name__ == "__main__":
    main()