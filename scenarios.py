"""
Scenario math for the stress test.

Two separate models here, used for different questions:

1. Instant shock - "if the market drops X% and rates move Y bps today,
   what happens to my portfolio right now". Deterministic, uses
   historical betas as the sensitivity coefficients.

2. Forward distribution - "what's a plausible range of outcomes over
   the next N days". Built by bootstrapping actual historical daily
   returns rather than assuming a normal distribution - real markets
   have fatter tails than a normal curve gives you credit for, and
   bootstrapping sidesteps having to defend a distributional assumption.

Both models are intentionally simple enough that you could explain the
mechanics in an interview without hand-waving. A more "accurate" model
(GARCH volatility, multi-factor risk models, real FX hedging) would
take a lot more time to build and would be much harder to explain
honestly, so it was left out of v1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


MARKET_PROXIES = {
    "S&P 500": "^GSPC",
    "Nifty 50": "^NSEI",
}
RATE_PROXY = "^TNX"  # US 10-year treasury yield, in percentage points (e.g. 4.25)


def fetch_proxy_series(ticker: str, period: str = "3y") -> pd.Series:
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    return data["Close"].squeeze().dropna()


def compute_market_betas(returns: pd.DataFrame, market_returns: pd.Series) -> dict[str, float]:
    """Slope of each holding's daily return regressed on the market proxy's daily return.

    Kept for the case where every holding uses the same benchmark. For
    mixed portfolios (e.g. some US stocks, some Indian stocks), use
    compute_market_betas_per_holding instead.
    """
    aligned = returns.join(market_returns.rename("_market"), how="inner").dropna()
    betas = {}
    if aligned.empty or "_market" not in aligned:
        return {col: np.nan for col in returns.columns}
    market_var = aligned["_market"].var()
    for col in returns.columns:
        if market_var == 0 or col not in aligned:
            betas[col] = np.nan
            continue
        cov = aligned[[col, "_market"]].cov().iloc[0, 1]
        betas[col] = cov / market_var
    return betas


def compute_market_betas_per_holding(
    returns: pd.DataFrame,
    ticker_benchmark_map: dict[str, str],
    benchmark_returns: dict[str, pd.Series],
) -> dict[str, float]:
    """Like compute_market_betas, but each ticker is regressed against its own
    assigned benchmark instead of one benchmark for the whole portfolio.

    ticker_benchmark_map: {"AAPL": "S&P 500", "RELIANCE.NS": "Nifty 50"}
    benchmark_returns: {"S&P 500": <daily return series>, "Nifty 50": <daily return series>}
    """
    betas = {}
    for ticker in returns.columns:
        benchmark_name = ticker_benchmark_map.get(ticker)
        bench_series = benchmark_returns.get(benchmark_name)
        if bench_series is None:
            betas[ticker] = np.nan
            continue
        aligned = pd.concat([returns[ticker], bench_series.rename("_market")], axis=1, join="inner").dropna()
        if aligned.empty:
            betas[ticker] = np.nan
            continue
        market_var = aligned["_market"].var()
        if market_var == 0:
            betas[ticker] = np.nan
            continue
        cov = aligned.cov().iloc[0, 1]
        betas[ticker] = cov / market_var
    return betas


def compute_rate_betas(returns: pd.DataFrame, rate_level: pd.Series) -> dict[str, float]:
    """Sensitivity of each holding's daily return to a 1-percentage-point (100bps) move
    in the rate proxy's level. rate_level is the raw yield series (e.g. 4.25), not a return."""
    rate_changes = rate_level.diff().dropna()  # change in percentage points, day over day
    aligned = returns.join(rate_changes.rename("_rate"), how="inner").dropna()
    betas = {}
    if aligned.empty or "_rate" not in aligned:
        return {col: np.nan for col in returns.columns}
    rate_var = aligned["_rate"].var()
    for col in returns.columns:
        if rate_var == 0 or col not in aligned:
            betas[col] = np.nan
            continue
        cov = aligned[[col, "_rate"]].cov().iloc[0, 1]
        betas[col] = cov / rate_var  # return per 1pp (100bps) move in yield
    return betas


def instant_shock_impact(
    holdings: pd.DataFrame,
    market_betas: dict[str, float],
    rate_betas: dict[str, float],
    market_shock_pct: float,
    rate_shock_bps: float,
) -> pd.DataFrame:
    """Deterministic dollar impact per holding from a simultaneous market move and rate move.

    market_shock_pct: e.g. -0.15 for a 15% market drawdown
    rate_shock_bps: e.g. 100 for a 100bps rate increase
    """
    out = holdings.copy()
    rate_shock_pp = rate_shock_bps / 100.0
    out["market_beta"] = out["ticker"].map(market_betas)
    out["rate_beta"] = out["ticker"].map(rate_betas)
    out["implied_return"] = (
        out["market_beta"].fillna(0) * market_shock_pct
        + out["rate_beta"].fillna(0) * rate_shock_pp
    )
    out["dollar_impact"] = out["value"] * out["implied_return"]
    out["new_value"] = out["value"] + out["dollar_impact"]
    return out


def bootstrap_portfolio_paths(
    returns: pd.DataFrame,
    weights: pd.Series,
    horizon_days: int,
    n_sims: int,
    shock_pct: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate terminal portfolio returns by resampling historical daily returns.

    For each simulation, draw `horizon_days` days at random (with replacement)
    from the joint history of all holdings on the same dates, so correlation
    between holdings on the same historical day is preserved. An optional
    instant shock is applied once at the start of each path - this is how
    a scenario like "market drops 20% then continues to behave normally"
    gets layered onto the baseline volatility.
    """
    rng = np.random.default_rng(seed)
    aligned = returns.dropna(how="any")
    if aligned.empty:
        return np.array([])

    tickers = [t for t in weights.index if t in aligned.columns]
    w = weights[tickers].values
    daily_portfolio_returns = aligned[tickers].values @ w  # one number per historical day

    n_days_available = len(daily_portfolio_returns)
    sampled_idx = rng.integers(0, n_days_available, size=(n_sims, horizon_days))
    sampled_returns = daily_portfolio_returns[sampled_idx]  # shape (n_sims, horizon_days)

    path_returns = (1 + sampled_returns).prod(axis=1) - 1
    if shock_pct:
        path_returns = (1 + shock_pct) * (1 + path_returns) - 1

    return path_returns
