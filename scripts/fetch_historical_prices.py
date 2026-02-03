#!/usr/bin/env python3
"""
Fetch historical daily prices from CoinGecko and save to data/historical_prices.csv.
Use this to populate price data for 12-week momentum and testing.
Run from project root: python scripts/fetch_historical_prices.py
"""

import sys
import logging
from pathlib import Path

import yaml

# Add project root so lib imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.data.price_fetcher import PriceFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    historical_days = config.get("data", {}).get("historical_days", 200)
    coin_to_cg = {
        c: info.get("coingecko_id", c)
        for c, info in config.get("tokens", {}).items()
    }
    if not coin_to_cg:
        logger.error("No tokens with coingecko_id in config.yaml")
        sys.exit(1)
    logger.info("Fetching %s days of prices for %s", historical_days, list(coin_to_cg.keys()))
    fetcher = PriceFetcher(coin_to_coingecko_id=coin_to_cg)
    df = fetcher.fetch_historical_prices(days=historical_days)
    if df.empty:
        logger.error("No price data returned")
        sys.exit(1)
    out_dir = PROJECT_ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "historical_prices.csv"
    df.to_csv(out_path)
    logger.info("Saved %s rows x %s coins to %s", len(df), len(df.columns), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
