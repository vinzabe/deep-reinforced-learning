"""Download hourly metals data for higher-frequency training."""

import os
import sys
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "data/metals_hourly"
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "copper": "HG=F",
    "platinum": "PL=F",
    "palladium": "PA=F",
}

MACRO_TICKERS = {
    "dxy": "DX-Y.NYB",
    "spx": "^GSPC",
    "us10y": "^TNX",
    "oil": "CL=F",
}


def download_hourly(ticker, name, data_dir):
    path = os.path.join(data_dir, f"{name}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  {name}: {len(df)} rows (existing)")
        return df

    print(f"  Downloading {name} ({ticker}) hourly...")
    df = yf.download(ticker, period="5y", interval="1h", progress=False, auto_adjust=True)
    if df.empty:
        print(f"    WARNING: No data for {name}")
        return pd.DataFrame()

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"datetime": "time"})
    if "time" not in df.columns:
        df = df.rename(columns={"date": "time"})

    required = ["time", "open", "high", "low", "close"]
    df = df[required].copy()
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"    Saved {len(df)} rows to {path}")
    return df


def main():
    print(f"Downloading hourly metals data to {DATA_DIR}/")

    for name, ticker in TICKERS.items():
        download_hourly(ticker, name, DATA_DIR)

    print("\nDownloading hourly macro data...")
    macro_dir = "data/hourly_macro"
    os.makedirs(macro_dir, exist_ok=True)
    for name, ticker in MACRO_TICKERS.items():
        download_hourly(ticker, name, macro_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
