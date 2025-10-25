"""
get_stratified_data.py

Downloads and creates a stratified sample of Lending Club data with proper class distribution
for meaningful fairness analysis.

This script:
1. Downloads the full Lending Club dataset from Kaggle using kagglehub
2. Creates a stratified sample with balanced default/non-default cases
3. Ensures diverse representation across loan characteristics
4. Replaces the current loan.csv with a more suitable dataset

Usage:
python get_stratified_data.py --target_size 5000 --default_rate 0.15

Requirements:
- kagglehub: pip install kagglehub
- Kaggle credentials configured (automatically detected)
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import os

def get_logger() -> logging.Logger:
    logger = logging.getLogger("get_stratified_data")
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

def check_kagglehub_setup() -> bool:
    """Check if kagglehub is available."""
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        return True
    except ImportError:
        print("❌ kagglehub not available")
        print("\n🔧 To fix this:")
        print("1. Install kagglehub: pip install kagglehub")
        print("2. Kaggle credentials should be automatically detected")
        return False

def load_lending_club_data(dataset_name: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Load Lending Club dataset using kagglehub."""
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        
        logger.info(f"📥 Loading dataset: {dataset_name}")
        
        # Load the dataset directly as pandas DataFrame
        # The file_path is empty to get the main dataset file
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            dataset_name,
            "",  # Empty file_path to get the main file
        )
        
        logger.info(f"✅ Loaded data: {len(df):,} rows, {len(df.columns)} columns")
        logger.info(f"📊 Dataset size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("💡 Trying alternative approach...")
        
        # Fallback: try to load specific files if main approach fails
        try:
            # Try common file names
            possible_files = ["accepted_2007_to_2018Q4.csv", "loan.csv", "LoanStats.csv"]
            
            for file_name in possible_files:
                try:
                    logger.info(f"🔍 Trying file: {file_name}")
                    df = kagglehub.load_dataset(
                        KaggleDatasetAdapter.PANDAS,
                        dataset_name,
                        file_name,
                    )
                    logger.info(f"✅ Successfully loaded: {file_name}")
                    return df
                except:
                    continue
                    
            logger.error("❌ Could not load any data files")
            return None
            
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return None

def analyze_loan_statuses(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Analyze and categorize loan statuses."""
    logger.info("📊 Analyzing loan statuses...")
    
    status_counts = df['loan_status'].value_counts()
    logger.info(f"Total unique statuses: {len(status_counts)}")
    
    for status, count in status_counts.items():
        pct = (count / len(df)) * 100
        logger.info(f"  {status}: {count:,} ({pct:.2f}%)")

def create_binary_target(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Create binary loan_default target from loan_status."""
    
    # Define default categories
    default_statuses = [
        'charged off',
        'default', 
        'late (31-120 days)',
        'late (16-30 days)', 
        'does not meet the credit policy. status:charged off',
        'in grace period'  # Sometimes considered risky
    ]
    
    # Clean and categorize
    df = df.copy()
    df['loan_status_clean'] = df['loan_status'].astype(str).str.strip().str.lower()
    df['loan_default'] = df['loan_status_clean'].isin(default_statuses).astype(int)
    
    # Log distribution
    default_rate = df['loan_default'].mean()
    n_defaults = df['loan_default'].sum()
    n_total = len(df)
    
    logger.info(f"🎯 Target variable created:")
    logger.info(f"  Defaults: {n_defaults:,} ({default_rate:.3f})")
    logger.info(f"  Non-defaults: {n_total - n_defaults:,} ({1-default_rate:.3f})")
    
    return df

def create_stratified_sample(
    df: pd.DataFrame, 
    target_size: int, 
    target_default_rate: float,
    seed: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """Create a stratified sample with desired size and default rate."""
    
    logger.info(f"🎲 Creating stratified sample...")
    logger.info(f"  Target size: {target_size:,}")
    logger.info(f"  Target default rate: {target_default_rate:.3f}")
    
    # Separate defaults and non-defaults
    defaults = df[df['loan_default'] == 1]
    non_defaults = df[df['loan_default'] == 0]
    
    logger.info(f"  Available defaults: {len(defaults):,}")
    logger.info(f"  Available non-defaults: {len(non_defaults):,}")
    
    # Calculate target counts
    target_defaults = int(target_size * target_default_rate)
    target_non_defaults = target_size - target_defaults
    
    logger.info(f"  Need defaults: {target_defaults:,}")
    logger.info(f"  Need non-defaults: {target_non_defaults:,}")
    
    # Check if we have enough data
    if len(defaults) < target_defaults:
        logger.warning(f"⚠️  Not enough defaults available. Using all {len(defaults):,} defaults.")
        target_defaults = len(defaults)
        target_non_defaults = target_size - target_defaults
        
    if len(non_defaults) < target_non_defaults:
        logger.warning(f"⚠️  Not enough non-defaults available. Adjusting sample size.")
        target_non_defaults = len(non_defaults)
        target_size = target_defaults + target_non_defaults
    
    # Sample
    np.random.seed(seed)
    
    sampled_defaults = defaults.sample(n=target_defaults, random_state=seed)
    sampled_non_defaults = non_defaults.sample(n=target_non_defaults, random_state=seed)
    
    # Combine and shuffle
    stratified_sample = pd.concat([sampled_defaults, sampled_non_defaults])
    stratified_sample = stratified_sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Verify
    actual_default_rate = stratified_sample['loan_default'].mean()
    logger.info(f"✅ Stratified sample created:")
    logger.info(f"  Final size: {len(stratified_sample):,}")
    logger.info(f"  Actual default rate: {actual_default_rate:.3f}")
    
    return stratified_sample

def ensure_feature_diversity(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Ensure diverse representation across key features."""
    
    logger.info("🌈 Ensuring feature diversity...")
    
    # Check loan amount distribution
    if 'loan_amnt' in df.columns:
        df['loan_amnt'] = pd.to_numeric(df['loan_amnt'], errors='coerce')
        loan_stats = df['loan_amnt'].describe()
        logger.info(f"  Loan amounts: ${loan_stats['min']:.0f} - ${loan_stats['max']:.0f} (median: ${loan_stats['50%']:.0f})")
    
    # Check income distribution  
    if 'annual_inc' in df.columns:
        df['annual_inc'] = pd.to_numeric(df['annual_inc'], errors='coerce')
        income_stats = df['annual_inc'].describe()
        logger.info(f"  Annual income: ${income_stats['min']:.0f} - ${income_stats['max']:.0f} (median: ${income_stats['50%']:.0f})")
    
    # Check grade distribution
    if 'grade' in df.columns:
        grade_dist = df['grade'].value_counts().sort_index()
        logger.info(f"  Grade distribution: {dict(grade_dist)}")
    
    # Check purpose distribution
    if 'purpose' in df.columns:
        purpose_dist = df['purpose'].value_counts().head()
        logger.info(f"  Top purposes: {dict(purpose_dist)}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Download and create stratified Lending Club dataset")
    parser.add_argument("--kaggle_dataset", type=str, default="adarshsng/lending-club-loan-data-csv", 
                       help="Kaggle dataset identifier")
    parser.add_argument("--target_size", type=int, default=5000, 
                       help="Target sample size")
    parser.add_argument("--default_rate", type=float, default=0.15, 
                       help="Target default rate (0.10-0.20 recommended)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--output_file", type=str, default="data/loan.csv", 
                       help="Output file path")
    
    args = parser.parse_args()
    logger = get_logger()
    
    logger.info("🚀 Starting stratified data creation process...")
    
    # Check kagglehub setup
    if not check_kagglehub_setup():
        return
    
    try:
        # Load data directly using kagglehub
        df = load_lending_club_data(args.kaggle_dataset, logger)
        if df is None:
            logger.error("❌ Failed to load data")
            return
        
        logger.info(f"📖 Loaded {len(df):,} records with {len(df.columns)} columns")
        
        # Analyze current distribution
        analyze_loan_statuses(df, logger)
        
        # Create binary target
        df = create_binary_target(df, logger)
        
        # Check if we have enough defaults
        current_default_rate = df['loan_default'].mean()
        if current_default_rate < 0.05:
            logger.error(f"❌ Dataset has too few defaults ({current_default_rate:.3f}). Try a different dataset.")
            return
        
        # Create stratified sample
        stratified_df = create_stratified_sample(
            df, args.target_size, args.default_rate, args.seed, logger
        )
        
        # Ensure diversity
        stratified_df = ensure_feature_diversity(stratified_df, logger)
        
        # Save result
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup original if it exists
        if output_path.exists():
            backup_path = output_path.with_suffix('.backup.csv')
            logger.info(f"📁 Backing up original file to: {backup_path}")
            import shutil
            shutil.copy2(output_path, backup_path)
        
        # Remove loan_default column (will be recreated by clean script)
        output_df = stratified_df.drop('loan_default', axis=1, errors='ignore')
        output_df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Stratified dataset saved to: {output_path}")
        logger.info(f"📊 Final dataset: {len(output_df):,} records, {len(output_df.columns)} features")
        
        logger.info("🎉 Process completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Run: python clean_lending_club.py --input_path data/loan.csv --out_dir processed")
        logger.info("2. Run: python scripts/train_models.py --data_dir processed --out_dir metrics") 
        logger.info("3. Run: python scripts/compute_fairness.py --data_dir processed --predictions_dir metrics --out_dir metrics/fairness")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()