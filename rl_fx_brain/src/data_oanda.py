"""Robust OANDA v20 REST historical candle downloader.

Design goals:
- Read credentials ONLY from env vars (OANDA_API_TOKEN / OANDA_ACCOUNT_ID /
  OANDA_ENVIRONMENT). Never hardcode. Never log the token.
- Use a single consistent price basis (mid by default).
- Paginate/backfill: OANDA caps each response at ~5000 candles. We walk
  forward in time using `from`/`count` requests.
- Retry with exponential backoff on transient errors.
- Validate response payloads: reject malformed/non-complete candles.
- Clean the dataframe: drop duplicates, sort UTC, handle weekend gaps.
- Persist parquet per-instrument for fast re-use.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .utils import ensure_dir, get_logger, require_env

LOG = get_logger(__name__)


OANDA_HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}


class OANDAError(Exception):
    """Raised for unrecoverable OANDA API errors."""


class TransientOANDAError(OANDAError):
    """Raised for errors that should trigger a retry."""


@dataclass
class CandleRequest:
    instrument: str
    granularity: str         # e.g. "H1" or "M15"
    from_iso: str            # ISO8601 UTC, e.g. "2019-01-01T00:00:00Z"
    to_iso: str              # ISO8601 UTC
    price_type: str = "M"    # "M" | "B" | "A" | "BA" etc. Keep consistent.
    batch_count: int = 4500


class OANDADataClient:
    """Thin HTTP wrapper around OANDA v20 /instruments/{}/candles."""

    def __init__(
        self,
        token: Optional[str] = None,
        account_id: Optional[str] = None,
        environment: Optional[str] = None,
        max_retries: int = 6,
        retry_backoff_seconds: float = 2.0,
        request_timeout: float = 30.0,
    ) -> None:
        # Load env if not provided (never log actual token value).
        self._token = token or require_env("OANDA_API_TOKEN")
        self._account_id = account_id or require_env("OANDA_ACCOUNT_ID")
        env = environment or os.environ.get("OANDA_ENVIRONMENT", "practice")
        env = env.strip().lower()
        if env not in OANDA_HOSTS:
            raise OANDAError(
                f"OANDA_ENVIRONMENT must be one of {list(OANDA_HOSTS)}, got '{env}'"
            )
        self._host = OANDA_HOSTS[env]
        self._env = env
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff_seconds
        self._timeout = request_timeout

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._token}",
                "Accept-Datetime-Format": "RFC3339",
                "Content-Type": "application/json",
                "User-Agent": "rl_fx_brain/0.1.0",
            }
        )
        LOG.info(
            "OANDA client ready (env=%s, account=%s***)",
            self._env,
            self._account_id[:4],
        )

    # ------------------------------------------------------------------
    # Low level single GET with retry (tenacity handles retry logic).
    # ------------------------------------------------------------------
    def _do_get(self, url: str, params: Dict[str, str]) -> Dict:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(
                multiplier=self._retry_backoff, min=self._retry_backoff, max=60
            ),
            retry=retry_if_exception_type(
                (TransientOANDAError, requests.exceptions.RequestException)
            ),
        )
        def _call() -> Dict:
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)
            except requests.exceptions.RequestException as e:
                # Network-level failure: retry.
                raise TransientOANDAError(f"network error: {e}") from e

            # Handle rate limiting and server errors explicitly.
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "2"))
                LOG.warning("429 rate limited, sleeping %.1fs", retry_after)
                time.sleep(retry_after)
                raise TransientOANDAError("rate limited (429)")
            if 500 <= resp.status_code < 600:
                raise TransientOANDAError(f"server error {resp.status_code}")
            if resp.status_code == 401:
                raise OANDAError("401 unauthorized: check OANDA_API_TOKEN")
            if resp.status_code >= 400:
                # Try to grab OANDA's error payload without leaking it further.
                msg = ""
                try:
                    j = resp.json()
                    msg = j.get("errorMessage", "") or j.get("error", "")
                except Exception:
                    msg = resp.text[:200]
                raise OANDAError(
                    f"OANDA API error {resp.status_code}: {msg}"
                )
            try:
                return resp.json()
            except ValueError as e:
                raise TransientOANDAError(f"malformed JSON: {e}") from e

        return _call()

    # ------------------------------------------------------------------
    # Public: fetch full history between two dates for one instrument.
    # ------------------------------------------------------------------
    def fetch_candles(self, req: CandleRequest) -> pd.DataFrame:
        """Fetch all candles between req.from_iso and req.to_iso.

        Uses `from` + `count` pagination loop. Advances `from` by the last
        candle time until we cross `to_iso` or get an empty response.
        """
        url = f"{self._host}/v3/instruments/{req.instrument}/candles"
        params_base = {
            "price": req.price_type,
            "granularity": req.granularity,
            "smooth": "false",
        }

        end_dt = _parse_iso(req.to_iso)
        cursor_iso = req.from_iso
        frames: List[pd.DataFrame] = []
        total_rows = 0

        LOG.info(
            "Downloading %s %s from %s to %s (batch=%d)",
            req.instrument,
            req.granularity,
            req.from_iso,
            req.to_iso,
            req.batch_count,
        )

        while True:
            params = dict(params_base)
            params["from"] = cursor_iso
            params["count"] = str(int(req.batch_count))

            payload = self._do_get(url, params)
            raw = payload.get("candles", [])
            if not raw:
                LOG.debug("empty batch at cursor=%s", cursor_iso)
                break

            df = _candles_to_df(raw)
            if df.empty:
                break

            frames.append(df)
            total_rows += len(df)

            last_ts = df["time"].iloc[-1]
            # Advance cursor by 1 second past last candle to avoid reposting.
            next_cursor = last_ts + pd.Timedelta(seconds=1)
            if next_cursor >= end_dt:
                break
            cursor_iso = next_cursor.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Safety: if we didn't move forward, break to avoid infinite loop.
            if df["time"].iloc[-1] <= df["time"].iloc[0]:
                LOG.warning("No forward progress, stopping pagination")
                break

        if not frames:
            raise OANDAError(
                f"No candles returned for {req.instrument} {req.granularity} "
                f"{req.from_iso}..{req.to_iso}"
            )

        df = pd.concat(frames, ignore_index=True)
        df = _clean_candles(df, req.instrument)
        LOG.info("%s: downloaded %d cleaned rows", req.instrument, len(df))
        return df


# ---------------------------------------------------------------------------
# Helpers: candle parsing + cleanup
# ---------------------------------------------------------------------------


def _parse_iso(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).tz_convert("UTC") if "+" in s or "Z" in s else pd.Timestamp(s, tz="UTC")


def _candles_to_df(raw: List[Dict]) -> pd.DataFrame:
    """Parse OANDA candles into a dataframe. Drop incomplete candles."""
    rows: List[Dict] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        if not c.get("complete", False):
            continue                     # ignore partial/current bar
        # Mid price by default. Support bid/ask if configured.
        price = c.get("mid") or c.get("bid") or c.get("ask")
        if price is None:
            continue
        try:
            row = {
                "time": c["time"],
                "open": float(price["o"]),
                "high": float(price["h"]),
                "low": float(price["l"]),
                "close": float(price["c"]),
                "volume": float(c.get("volume", 0)),
            }
        except (KeyError, TypeError, ValueError):
            # Malformed payload, skip.
            continue
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    return df


def _clean_candles(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Drop duplicates, sort, null-handle, weekend-gap-safe."""
    if df.empty:
        return df

    # Drop duplicate timestamps (keep first).
    df = df.drop_duplicates(subset="time", keep="first")
    # Sort by time ascending.
    df = df.sort_values("time").reset_index(drop=True)

    # Drop rows with any NaN in OHLC.
    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    dropped = before - len(df)
    if dropped:
        LOG.warning("%s: dropped %d rows with NaN OHLC", instrument, dropped)

    # Sanity: OHLC consistency.
    bad = (
        (df["high"] < df["low"])
        | (df["open"] <= 0)
        | (df["close"] <= 0)
    )
    if bad.any():
        LOG.warning(
            "%s: %d candles failed OHLC sanity check, removing",
            instrument,
            int(bad.sum()),
        )
        df = df[~bad].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# High-level: download entire universe and persist parquet
# ---------------------------------------------------------------------------


def download_universe(
    instruments: List[str],
    granularity: str,
    years_of_history: int,
    raw_dir: str | Path,
    price_type: str = "M",
    batch_count: int = 4500,
    max_retries: int = 6,
    retry_backoff_seconds: float = 2.0,
    client: Optional[OANDADataClient] = None,
    force: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Download candles for a list of instruments and persist as parquet.

    Returns dict[instrument -> dataframe].
    """
    raw_dir = ensure_dir(raw_dir)

    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=int(years_of_history * 365.25))
    from_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    to_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    out: Dict[str, pd.DataFrame] = {}
    need_download = False
    for sym in instruments:
        cache_path = Path(raw_dir) / f"{sym}_{granularity}.parquet"
        if cache_path.exists() and not force:
            LOG.info("%s: using cached %s", sym, cache_path)
            df = pd.read_parquet(cache_path)
            out[sym] = df
        else:
            need_download = True
            break

    if not need_download:
        return out

    if client is None:
        client = OANDADataClient(
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    for sym in instruments:
        cache_path = Path(raw_dir) / f"{sym}_{granularity}.parquet"
        if cache_path.exists() and not force:
            df = pd.read_parquet(cache_path)
            out[sym] = df
            continue

        req = CandleRequest(
            instrument=sym,
            granularity=granularity,
            from_iso=from_iso,
            to_iso=to_iso,
            price_type=price_type,
            batch_count=batch_count,
        )
        try:
            df = client.fetch_candles(req)
        except OANDAError as e:
            LOG.error("%s: download failed: %s", sym, e)
            raise

        df.to_parquet(cache_path, index=False)
        out[sym] = df
        LOG.info("%s: wrote %s (%d rows)", sym, cache_path, len(df))

    return out
