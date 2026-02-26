"""
download_hmda.py

Download a manageable subset of HMDA (Home Mortgage Disclosure Act) data
from the CFPB Data Browser API for fairness analysis.

HMDA is the gold-standard dataset for fair-lending analysis. It contains
real demographic attributes (race, sex, age, ethnicity) that are legally
protected under the Equal Credit Opportunity Act (ECOA) and the
Fair Housing Act.

The script downloads data for a single state (default: Wyoming, the
smallest-volume state) filtered to originated and denied applications,
then samples to a target size.

Usage:
  cd hmda_dataset
  python3 download_hmda.py [--state WY] [--year 2022] [--max_rows 15000]
"""
from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# action_taken: 1 = originated, 3 = denied
CFPB_URL_TEMPLATE = (
    "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
    "?states={state}&years={year}&actions_taken=1,3"
)


def download(state: str, year: int, max_rows: int, out_dir: Path) -> None:
    url = CFPB_URL_TEMPLATE.format(state=state, year=year)
    log.info(f"Downloading HMDA data from CFPB: state={state}, year={year}")
    log.info(f"URL: {url}")

    try:
        df = pd.read_csv(url, low_memory=False)
    except Exception as e:
        log.error(
            f"Download failed: {e}\n\n"
            "Manual fallback: visit https://ffiec.cfpb.gov/data-browser/data/\n"
            f"  1. Select year={year}, state={state}\n"
            "  2. Filter action_taken to 'Loan originated' and 'Application denied'\n"
            f"  3. Download CSV and save as {out_dir / 'hmda_raw.csv'}\n"
        )
        raise

    log.info(f"Downloaded {len(df)} rows, {df.shape[1]} columns")

    # Sample if larger than target
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
        log.info(f"Sampled down to {max_rows} rows")

    out_path = out_dir / "hmda_raw.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved to {out_path}")

    # Quick summary
    log.info(f"action_taken distribution:\n{df['action_taken'].value_counts().to_string()}")
    if "derived_race" in df.columns:
        log.info(f"Race distribution:\n{df['derived_race'].value_counts().head(10).to_string()}")
    if "derived_sex" in df.columns:
        log.info(f"Sex distribution:\n{df['derived_sex'].value_counts().to_string()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HMDA data from CFPB")
    parser.add_argument("--state", default="WY", help="Two-letter state code (default: WY)")
    parser.add_argument("--year", type=int, default=2022, help="HMDA filing year")
    parser.add_argument("--max_rows", type=int, default=15000, help="Max rows to keep")
    parser.add_argument("--out_dir", default="data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    download(state=args.state, year=args.year, max_rows=args.max_rows, out_dir=out_dir)


if __name__ == "__main__":
    main()
