"""
Portfolio data and valuation.

Pulls prices via yfinance and turns a list of (ticker, quantity, currency)
holdings into current values, weights, and a daily return history.

v2 adds an FX layer: if a holding's currency differs from the portfolio's
base currency, its value gets converted before weights and totals are
computed. The conversion rate is a live snapshot at the time you fetch,
not a hedge - if you're tracking real exposure to currency risk
separately, that's a different (and bigger) tool.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf


def fetch_price_history(tickers: list[str], period: str = "3y") -> pd.DataFrame:
    """Daily close prices for each ticker, columns = tickers."""
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if len(tickers) == 1:
        # yfinance flattens differently when there's only one ticker
        closes = data["Close"].to_frame(name=tickers[0])
    else:
        closes = data["Close"]
    return closes.dropna(how="all")


def latest_prices(price_history: pd.DataFrame) -> dict[str, float]:
    if price_history.empty:
        return {}
    last_row = price_history.ffill().iloc[-1]
    return last_row.to_dict()


def daily_returns(price_history: pd.DataFrame) -> pd.DataFrame:
    return price_history.pct_change().dropna(how="all")


def fetch_fx_rate(from_currency: str, to_currency: str) -> float:
    """Units of to_currency per 1 unit of from_currency, using a live snapshot.

    Tries the direct pair first (e.g. USDINR=X), falls back to the inverse
    pair if Yahoo only lists it that way round. Returns 1.0 if the two
    currencies are the same - no network call needed for that case.
    """
    if from_currency == to_currency:
        return 1.0

    direct_ticker = f"{from_currency}{to_currency}=X"
    try:
        data = yf.download(direct_ticker, period="5d", progress=False)
        rate = data["Close"].dropna().iloc[-1]
        return float(rate)
    except Exception:
        pass

    inverse_ticker = f"{to_currency}{from_currency}=X"
    try:
        data = yf.download(inverse_ticker, period="5d", progress=False)
        rate = data["Close"].dropna().iloc[-1]
        return 1.0 / float(rate)
    except Exception:
        raise ValueError(
            f"Couldn't find an FX rate for {from_currency} to {to_currency} on Yahoo Finance. "
            f"Check the currency codes are valid ISO codes (USD, INR, EUR, GBP, etc)."
        )


def fetch_fx_rates(currencies: list[str], base_currency: str) -> dict[str, float]:
    """FX rate for each currency in the list, converting into base_currency."""
    unique_currencies = sorted(set(currencies))
    return {c: fetch_fx_rate(c, base_currency) for c in unique_currencies}


def value_holdings(
    holdings: pd.DataFrame, prices: dict[str, float], fx_rates: dict[str, float] | None = None
):
    """holdings needs columns: ticker, quantity, currency.

    fx_rates maps currency -> rate into the base currency. If not provided,
    everything is assumed to already be in the same currency (rate of 1.0).
    Returns (holdings_with_value_columns, total_value_in_base_currency).
    """
    out = holdings.copy()
    out["price"] = out["ticker"].map(prices)
    out["value_local"] = out["price"] * out["quantity"]

    if fx_rates is None:
        out["fx_rate"] = 1.0
    else:
        out["fx_rate"] = out["currency"].map(fx_rates).fillna(1.0)

    out["value"] = out["value_local"] * out["fx_rate"]
    total = out["value"].sum()
    out["weight"] = out["value"] / total if total else 0.0
    return out, total
