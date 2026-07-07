"""
Personal Finance Decision Engine.

Run with: streamlit run app.py

The math lives in portfolio.py and scenarios.py so it can be tested on its
own without a browser open. storage.py handles saving/loading named
portfolios to a local JSON file.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import portfolio
import scenarios
import storage

st.set_page_config(page_title="Personal Finance Decision Engine", layout="wide")

st.title("Personal Finance Decision Engine")
st.write(
    "Stress-tests a portfolio against a market move and a rate move, and shows "
    "a range of likely outcomes over a chosen horizon based on historical volatility. "
    "It is not a prediction. It is a way to see how exposed a portfolio actually is "
    "before deciding whether to do anything about it."
)

st.divider()

# ---------------------------------------------------------------------------
# Step 0: base currency + load a saved portfolio
# ---------------------------------------------------------------------------
COMMON_CURRENCIES = ["USD", "INR", "EUR", "GBP", "JPY", "AUD", "CAD", "SGD"]
BENCHMARK_NAMES = list(scenarios.MARKET_PROXIES.keys())

if "holdings_df" not in st.session_state:
    st.session_state.holdings_df = pd.DataFrame(
        [
            {"ticker": "AAPL", "quantity": 10, "currency": "USD", "benchmark": "S&P 500"},
            {"ticker": "MSFT", "quantity": 5, "currency": "USD", "benchmark": "S&P 500"},
        ]
    )
if "base_currency" not in st.session_state:
    st.session_state.base_currency = "USD"

top_col1, top_col2 = st.columns([1, 2])
with top_col1:
    st.session_state.base_currency = st.selectbox(
        "Base currency for totals",
        COMMON_CURRENCIES,
        index=COMMON_CURRENCIES.index(st.session_state.base_currency),
    )

with top_col2:
    saved_names = storage.list_portfolio_names()
    if saved_names:
        load_col, load_btn_col, del_btn_col = st.columns([2, 1, 1])
        with load_col:
            selected_saved = st.selectbox("Saved portfolios", saved_names, label_visibility="visible")
        with load_btn_col:
            st.write("")
            if st.button("Load"):
                loaded = storage.load_all_portfolios()[selected_saved]
                st.session_state.holdings_df = pd.DataFrame(loaded["holdings"])
                st.session_state.base_currency = loaded["base_currency"]
                st.rerun()
        with del_btn_col:
            st.write("")
            if st.button("Delete"):
                storage.delete_portfolio(selected_saved)
                st.rerun()
    else:
        st.caption("No saved portfolios yet - enter holdings below and save them once you're happy with the list.")

st.divider()

# ---------------------------------------------------------------------------
# Step 1: holdings input
# ---------------------------------------------------------------------------
st.header("1. Your holdings")
st.caption(
    "Enter tickers as they appear on Yahoo Finance (e.g. RELIANCE.NS for NSE-listed stocks). "
    "Set each holding's actual trading currency and the benchmark that fits it best - "
    "a US stock against the S&P 500, an Indian stock against the Nifty 50."
)

edited = st.data_editor(
    st.session_state.holdings_df,
    num_rows="dynamic",
    use_container_width=True,
    key="holdings_editor",
    column_config={
        "currency": st.column_config.SelectboxColumn("currency", options=COMMON_CURRENCIES),
        "benchmark": st.column_config.SelectboxColumn("benchmark", options=BENCHMARK_NAMES),
    },
)
st.session_state.holdings_df = edited

save_col, fetch_col = st.columns([1, 1])
with save_col:
    portfolio_name = st.text_input("Name this portfolio to save it", value="")
    if st.button("Save portfolio") and portfolio_name.strip():
        records = edited.dropna(subset=["ticker"]).to_dict("records")
        storage.save_portfolio(portfolio_name.strip(), records, st.session_state.base_currency)
        st.success(f"Saved as '{portfolio_name.strip()}'.")
        st.rerun()
with fetch_col:
    st.write("")
    fetch_clicked = st.button("Fetch prices and history", type="primary")

# ---------------------------------------------------------------------------
# Step 2: fetch prices, FX rates, and history
# ---------------------------------------------------------------------------
if fetch_clicked:
    rows = edited.dropna(subset=["ticker"])
    rows = rows[rows["ticker"].str.strip() != ""]
    if rows.empty:
        st.warning("Add at least one ticker first.")
    else:
        tickers = [t.strip().upper() for t in rows["ticker"].tolist()]
        currencies = rows["currency"].fillna(st.session_state.base_currency).tolist()

        with st.spinner("Pulling price history..."):
            price_history = portfolio.fetch_price_history(tickers)
            prices = portfolio.latest_prices(price_history)
            returns = portfolio.daily_returns(price_history)

            fx_rates = portfolio.fetch_fx_rates(currencies, st.session_state.base_currency)

            needed_benchmarks = sorted(set(rows["benchmark"].fillna(BENCHMARK_NAMES[0]).tolist()))
            benchmark_returns = {}
            for bench_name in needed_benchmarks:
                bench_ticker = scenarios.MARKET_PROXIES[bench_name]
                series = scenarios.fetch_proxy_series(bench_ticker)
                benchmark_returns[bench_name] = series.pct_change().dropna()

            rate_series = scenarios.fetch_proxy_series(scenarios.RATE_PROXY)

        st.session_state.price_history = price_history
        st.session_state.returns = returns
        st.session_state.fx_rates = fx_rates
        st.session_state.benchmark_returns = benchmark_returns
        st.session_state.rate_series = rate_series
        st.session_state.fetched_rows = rows.assign(ticker=tickers)
        st.success(
            f"Pulled {len(tickers)} ticker(s), {len(price_history)} trading days of history, "
            f"FX rates for {len(set(currencies))} currenc{'y' if len(set(currencies))==1 else 'ies'}."
        )

# ---------------------------------------------------------------------------
# Step 3: snapshot + sensitivities
# ---------------------------------------------------------------------------
if "returns" in st.session_state and st.session_state.get("fetched_rows") is not None:
    rows = st.session_state.fetched_rows
    prices = portfolio.latest_prices(st.session_state.price_history)
    valued, total_value = portfolio.value_holdings(rows, prices, st.session_state.fx_rates)

    st.header("2. Current snapshot")
    col1, col2 = st.columns([2, 1])
    with col1:
        display_cols = ["ticker", "quantity", "currency", "price", "value_local", "fx_rate", "value", "weight"]
        st.dataframe(
            valued[display_cols].style.format(
                {"price": "{:.2f}", "value_local": "{:,.2f}", "fx_rate": "{:.4f}", "value": "{:,.2f}", "weight": "{:.1%}"}
            ),
            use_container_width=True,
        )
    with col2:
        st.metric(f"Total portfolio value ({st.session_state.base_currency})", f"{total_value:,.2f}")
        fig_alloc = go.Figure(data=[go.Pie(labels=valued["ticker"], values=valued["weight"], hole=0.4)])
        fig_alloc.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig_alloc, use_container_width=True)

    ticker_benchmark_map = dict(zip(rows["ticker"], rows["benchmark"].fillna(BENCHMARK_NAMES[0])))
    market_betas = scenarios.compute_market_betas_per_holding(
        st.session_state.returns, ticker_benchmark_map, st.session_state.benchmark_returns
    )
    rate_betas = scenarios.compute_rate_betas(st.session_state.returns, st.session_state.rate_series)

    st.subheader("Estimated sensitivities (3-year daily history)")
    sens_df = pd.DataFrame(
        {
            "ticker": list(market_betas.keys()),
            "benchmark_used": [ticker_benchmark_map.get(t) for t in market_betas.keys()],
            "market_beta": list(market_betas.values()),
            "rate_beta_per_100bps": [rate_betas.get(t) for t in market_betas.keys()],
        }
    )
    st.dataframe(sens_df.style.format({"market_beta": "{:.2f}", "rate_beta_per_100bps": "{:.3f}"}), use_container_width=True)
    st.caption(
        "market_beta: how much a holding tends to move for every 1% move in its assigned "
        "benchmark. rate_beta_per_100bps: how much a holding's daily return has historically "
        "shifted for a 1 percentage point move in the 10-year US Treasury yield, used here as "
        "a general rate proxy even for non-US holdings - treat that as a rough approximation, "
        "not a precise hedge ratio."
    )

    # -----------------------------------------------------------------------
    # Step 4: scenario assumptions
    # -----------------------------------------------------------------------
    st.header("3. Scenario assumptions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        market_shock_pct = st.slider("Market move (%)", -50, 30, -15) / 100.0
    with c2:
        rate_shock_bps = st.slider("Rate move (bps)", -200, 200, 100, step=25)
    with c3:
        horizon_days = st.selectbox("Forward horizon (trading days)", [21, 63, 126, 252], index=1)
    with c4:
        n_sims = st.selectbox("Number of simulations", [1000, 2000, 5000], index=1)

    st.caption(
        "The market move below is applied through each holding's own beta, even though "
        "holdings may be tracked against different benchmarks - read it as 'a move of this "
        "size in whichever benchmark applies to each holding,' not one single global index "
        "moving by this amount."
    )

    run_clicked = st.button("Run stress test")

    if run_clicked:
        st.header("4. Results")

        # --- instant shock ---
        shock_result = scenarios.instant_shock_impact(
            valued, market_betas, rate_betas, market_shock_pct, rate_shock_bps
        )
        total_impact = shock_result["dollar_impact"].sum()
        total_new_value = total_value + total_impact

        st.subheader("Instant shock - if this happened today")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric(
                f"Portfolio value after shock ({st.session_state.base_currency})",
                f"{total_new_value:,.2f}",
                delta=f"{total_impact:,.2f} ({total_impact/total_value:.1%})",
            )
        with col_b:
            fig_waterfall = go.Figure(
                go.Bar(
                    x=shock_result["ticker"],
                    y=shock_result["dollar_impact"],
                    marker_color=["#c0392b" if v < 0 else "#27632a" for v in shock_result["dollar_impact"]],
                )
            )
            fig_waterfall.update_layout(
                title="Dollar impact per holding",
                margin=dict(t=40, b=10, l=10, r=10),
                height=280,
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

        # --- forward distribution ---
        st.subheader(f"Forward distribution - {horizon_days} trading days out")
        weights = pd.Series(valued.set_index("ticker")["weight"])

        paths_baseline = scenarios.bootstrap_portfolio_paths(
            st.session_state.returns, weights, horizon_days, n_sims, shock_pct=0.0, seed=1
        )
        paths_shocked = scenarios.bootstrap_portfolio_paths(
            st.session_state.returns, weights, horizon_days, n_sims, shock_pct=market_shock_pct, seed=1
        )

        if paths_baseline.size and paths_shocked.size:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=paths_baseline * 100, name="No shock applied", opacity=0.6, nbinsx=60))
            fig_dist.add_trace(go.Histogram(x=paths_shocked * 100, name="With market shock applied on day 0", opacity=0.6, nbinsx=60))
            fig_dist.update_layout(
                barmode="overlay",
                xaxis_title="Portfolio return over horizon (%)",
                yaxis_title="Number of simulated paths",
                height=400,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            p5, p50, p95 = np.percentile(paths_shocked, [5, 50, 95])
            colp1, colp2, colp3 = st.columns(3)
            colp1.metric("5th percentile outcome", f"{p5:.1%}")
            colp2.metric("Median outcome", f"{p50:.1%}")
            colp3.metric("95th percentile outcome", f"{p95:.1%}")
        else:
            st.warning("Not enough overlapping history across holdings to run the simulation.")

        st.caption(
            "The distribution is built by resampling actual historical daily returns, not by "
            "assuming a bell curve - so it picks up real fat-tail behavior from the data you "
            "fetched. It assumes the next stretch of trading days behaves statistically like "
            "the last three years. That assumption breaks down around genuine regime changes, "
            "which is exactly when you'd want it least - worth keeping in mind, not a flaw to "
            "fix by adding more decimal places."
        )
else:
    st.info("Add your holdings above and click 'Fetch prices and history' to continue.")
