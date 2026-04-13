"""
Fetch all precious/industrial metals OHLCV data from yfinance.

Metals fetched:
  - GC=F   : Gold Futures (XAU/USD)
  - SI=F   : Silver Futures (XAG/USD)
  - HG=F   : Copper Futures (XCU/USD)
  - PL=F   : Platinum Futures (XPT/USD)
  - PA=F   : Palladium Futures (XPD/USD)

Also fetches macro correlations:
  - DX-Y.NYB : DXY (US Dollar Index)
  - ^GSPC    : S&P 500 (risk sentiment)
  - ^TNX     : US 10-Year Treasury (opportunity cost)
  - CL=F     : WTI Crude Oil (commodity correlation)

Usage:
    python data/fetch_metals.py
    python data/fetch_metals.py --interval 1h  # hourly
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


METALS = {
    "GC=F": "gold",
    "SI=F": "silver",
    "HG=F": "copper",
    "PL=F": "platinum",
    "PA=F": "palladium",
}

MACRO = {
    "DX-Y.NYB": "dxy",
    "^GSPC": "spx",
    "^TNX": "us10y",
    "CL=F": "oil",
}


def fetch_one(ticker: str, name: str, interval: str, period: str, out_dir: Path) -> pd.DataFrame:
    print(f"  Fetching {name} ({ticker}) [{interval}]...", end=" ", flush=True)
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            print("EMPTY")
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"datetime": "time", "date": "time"})
        for c in ["open", "high", "low", "close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["time", "open", "high", "low", "close"]).copy()
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        cols = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[cols].copy()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"OK  {len(df):,} rows  {df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()}")
        return df
    except Exception as e:
        print(f"FAILED  {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="1d", help="yfinance interval (1d, 1h, 1wk)")
    parser.add_argument("--period", default="max", help="yfinance period (max, 10y, 5y)")
    parser.add_argument("--out-dir", default="data/metals", help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    print(f"=== Fetching metals data [{args.interval}] ===\n")
    for ticker, name in METALS.items():
        fetch_one(ticker, name, args.interval, args.period, out_dir)
        time.sleep(1)

    print(f"\n=== Fetching macro data [{args.interval}] ===\n")
    macro_dir = Path("data")
    for ticker, name in MACRO.items():
        fetch_one(ticker, name, args.interval, args.period, macro_dir)
        time.sleep(1)

    print("\n=== Summary ===")
    for p in sorted(out_dir.glob("*.csv")):
        df = pd.read_csv(p)
        print(f"  {p.name:20s}  {len(df):>8,} rows  {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    for p in sorted(macro_dir.glob("*.csv")):
        if p.stem in MACRO.values():
            df = pd.read_csv(p)
            print(f"  {p.stem:20s}  {len(df):>8,} rows")


if __name__ == "__main__":
    main()
