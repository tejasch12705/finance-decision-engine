# Personal Finance Decision Engine

A portfolio stress-testing tool built as a sharper alternative to generic budgeting apps. The question this answers is "how exposed am I to a market move or a rate move, and what's a realistic range of outcomes over the next few months" — not "where did my money go last month."

Built because most retail investor tools either give you historical charts (backward-looking) or vague risk scores (uninterpretable). This one lets you set a scenario and see the impact in your own currency, on your own holdings, using math you can explain.

## What it does

- Pulls 3 years of daily price history for any tickers you enter via Yahoo Finance
- Converts mixed-currency holdings into one base currency using a live FX snapshot, so a US stock and an Indian stock can sit in the same total without the number being meaningless
- Assigns each holding its own benchmark (S&P 500 for US holdings, Nifty 50 for Indian holdings) and estimates market beta separately per holding via OLS regression on daily returns
- Estimates each holding's sensitivity to moves in the US 10-year Treasury yield
- Given a market shock and a rate shock, computes the deterministic dollar impact per holding and in total
- Runs a bootstrap simulation over resampled historical daily returns to show a distribution of forward outcomes — with and without the shock applied — over a horizon you choose
- Saves and loads named portfolios to a local JSON file

## Stack

Python, Streamlit, yfinance, NumPy, pandas, Plotly. No API key required.

## Why bootstrap instead of Monte Carlo with assumed distributions

Monte Carlo simulations typically assume returns follow a normal distribution. Real equity returns don't — they have fatter tails, meaning large moves happen more often than a normal curve predicts. The bootstrap approach here resamples actual historical trading days (preserving the joint distribution across holdings on the same date), so the fat tails in your data show up in the simulation naturally. It also means the simulation's behaviour is directly traceable to real market history, which makes it easier to explain and interrogate.

The trade-off is that the bootstrap assumes the next period will behave statistically like the last three years. That assumption breaks down around genuine regime changes — which is exactly when you'd want a stress test most. Worth keeping in mind.

## Known limitations, stated upfront

- The FX conversion is a snapshot at fetch time, not a forward estimate. It tells you what your portfolio is worth right now in one currency, not anything about currency risk going forward.
- One rate proxy (US 10-year yield) is used regardless of which market a holding trades in. Fine as a rough directional read, not precise for holdings sensitive to non-US rates.
- If a ticker doesn't have enough overlapping history with its benchmark to compute a reliable beta, the beta shows as zero rather than returning a noisy estimate. Zero is more honest than a number that looks precise but isn't — the fix in a future version would be to fall back to a sector-average beta.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/finance-decision-engine.git
cd finance-decision-engine

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running it

```bash
source venv/bin/activate
streamlit run app.py
```

Opens at `http://localhost:8501`. No backend process needed.

## Using it

1. Set your base currency at the top
2. Enter holdings: ticker (as on Yahoo Finance — use `RELIANCE.NS` for NSE-listed stocks), quantity, currency, and which benchmark fits each holding
3. Save the portfolio if you want to reload it next session
4. Click Fetch — pulls price history and live FX rates
5. Check the sensitivity table: a beta of 1.8 means that holding has historically moved 1.8x the benchmark
6. Set your scenario assumptions and run the stress test
