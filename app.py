"""
Futures & Derivatives Analytics Platform
==========================================
An interactive Streamlit app combining technical analysis, ML price
forecasting, options pricing/Greeks, futures basis analytics, and risk
metrics -- built as a portfolio project for quant/derivatives-analyst roles.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from modules import data_fetcher as df_
from modules import indicators as ind
from modules import ml_predictor as mlp
from modules import options_pricing as opx
from modules import risk_metrics as risk
from modules import futures_curve as fut

# ----------------------------------------------------------------------------
# Page config + sleek dark theme styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Derivatives Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at 10% 0%, #0f1420 0%, #090c13 55%, #06080d 100%);
        color: #e6e9f0;
    }

    section[data-testid="stSidebar"] {
        background: #0b0f18;
        border-right: 1px solid #1c2333;
    }

    .metric-card {
        background: linear-gradient(145deg, #121826, #0d1220);
        border: 1px solid #1f2b40;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }
    .metric-label {
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8a93a8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #f2f4f8;
    }
    .metric-delta-pos { color: #35d68f; font-weight: 600; font-size: 0.9rem; }
    .metric-delta-neg { color: #ff5c72; font-weight: 600; font-size: 0.9rem; }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f2f4f8;
        border-left: 4px solid #4f7cff;
        padding-left: 12px;
        margin: 1.2rem 0 0.8rem 0;
    }

    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .pill-contango { background: rgba(53,214,143,0.15); color: #35d68f; }
    .pill-backwardation { background: rgba(255,92,114,0.15); color: #ff5c72; }

    div[data-testid="stMetric"] {
        background: #121826;
        border: 1px solid #1f2b40;
        border-radius: 12px;
        padding: 10px 14px;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #121826;
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        color: #9aa4b8;
    }
    .stTabs [aria-selected="true"] {
        background: #1a2436 !important;
        color: #f2f4f8 !important;
        border-bottom: 2px solid #4f7cff;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#4f7cff"
GREEN = "#35d68f"
RED = "#ff5c72"

# ----------------------------------------------------------------------------
# Sidebar -- global controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📈 Derivatives Analytics")
    st.caption("ML-driven futures, options & risk platform")
    st.divider()

    asset_class = st.radio("Asset class", ["Equities / Index", "Futures", "Upload CSV (Kaggle dataset)"])

    if asset_class == "Equities / Index":
        universe = {**df_.INDEX_TICKERS, "Custom ticker": None}
        choice = st.selectbox("Instrument", list(universe.keys()))
        ticker = st.text_input("Ticker (Yahoo Finance symbol)", value="AAPL") if choice == "Custom ticker" else universe[choice]
    elif asset_class == "Futures":
        choice = st.selectbox("Instrument", list(df_.FUTURES_TICKERS.keys()))
        ticker = df_.FUTURES_TICKERS[choice]
    else:
        ticker = None
        uploaded = st.file_uploader("Upload historical CSV", type=["csv"])

    period = st.selectbox("History window", ["6mo", "1y", "2y", "5y", "10y"], index=2)
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

    st.divider()
    st.caption("Data: Yahoo Finance (free, delayed) via `yfinance`. Upload a Kaggle CSV for fully offline backtests.")

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
if asset_class == "Upload CSV (Kaggle dataset)":
    if uploaded is None:
        st.title("📈 Derivatives Analytics Platform")
        st.info("Upload a historical OHLCV CSV in the sidebar to get started, or switch to a live Equities/Futures ticker.")
        st.stop()
    raw = df_.load_csv_fallback(uploaded)
    display_name = uploaded.name
else:
    if not ticker:
        st.stop()
    raw = df_.get_price_history(ticker, period=period, interval=interval)
    display_name = f"{choice} ({ticker})"

if raw.empty:
    st.error("No data returned. Check the ticker symbol or try a different instrument.")
    st.stop()

data = ind.add_all_indicators(raw)

# ----------------------------------------------------------------------------
# Header + live metrics
# ----------------------------------------------------------------------------
st.markdown(f"## {display_name}")

last_close = data["Close"].iloc[-1]
prev_close = data["Close"].iloc[-2] if len(data) > 1 else last_close
chg = last_close - prev_close
chg_pct = (chg / prev_close * 100) if prev_close else 0
vol_20 = data["Volatility_20"].iloc[-1]
rsi_last = data["RSI_14"].iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Close", f"{last_close:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
c2.metric("20D Annualized Vol", f"{vol_20 * 100:,.1f}%" if pd.notna(vol_20) else "—")
c3.metric("RSI (14)", f"{rsi_last:,.1f}" if pd.notna(rsi_last) else "—")
c4.metric("52W High / Low", f"{data['Close'].tail(252).max():,.2f} / {data['Close'].tail(252).min():,.2f}")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_dash, tab_ta, tab_ml, tab_opt, tab_fut, tab_risk = st.tabs(
    ["🏠 Dashboard", "📊 Technical Analysis", "🤖 ML Prediction", "🧮 Options Analytics", "⛓️ Futures & Basis", "🛡️ Risk"]
)

# ---------------- Dashboard ----------------
with tab_dash:
    st.markdown('<div class="section-header">Price & Volume</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
        name="Price", increasing_line_color=GREEN, decreasing_line_color=RED,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20", line=dict(color=ACCENT, width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], name="SMA 50", line=dict(color="#ffb84f", width=1.3)), row=1, col=1)
    vol_colors = np.where(data["Close"] >= data["Open"], GREEN, RED)
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color=vol_colors), row=2, col=1)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, xaxis_rangeslider_visible=False,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Recent Data</div>', unsafe_allow_html=True)
    st.dataframe(data[["Open", "High", "Low", "Close", "Volume"]].tail(10).sort_index(ascending=False), use_container_width=True)

# ---------------- Technical Analysis ----------------
with tab_ta:
    st.markdown('<div class="section-header">Momentum & Trend Indicators</div>', unsafe_allow_html=True)

    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.04,
                          subplot_titles=("Price with Bollinger Bands", "RSI (14)", "MACD"))
    fig2.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#e6e9f0", width=1.5)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], name="BB Upper", line=dict(color=ACCENT, width=1, dash="dot")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], name="BB Lower", line=dict(color=ACCENT, width=1, dash="dot"),
                               fill="tonexty", fillcolor="rgba(79,124,255,0.06)"), row=1, col=1)

    fig2.add_trace(go.Scatter(x=data.index, y=data["RSI_14"], name="RSI", line=dict(color="#ffb84f", width=1.5)), row=2, col=1)
    fig2.add_hline(y=70, line_dash="dash", line_color=RED, row=2, col=1)
    fig2.add_hline(y=30, line_dash="dash", line_color=GREEN, row=2, col=1)

    fig2.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color=ACCENT, width=1.3)), row=3, col=1)
    fig2.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], name="Signal", line=dict(color="#ffb84f", width=1.3)), row=3, col=1)
    macd_colors = np.where(data["MACD_Hist"] >= 0, GREEN, RED)
    fig2.add_trace(go.Bar(x=data.index, y=data["MACD_Hist"], name="Histogram", marker_color=macd_colors), row=3, col=1)

    fig2.update_layout(template=PLOTLY_TEMPLATE, height=780, margin=dict(t=40, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- ML Prediction ----------------
with tab_ml:
    st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    model_type = mc1.selectbox("Model", ["Random Forest", "Gradient Boosting"])
    lookback_lags = mc2.slider("Lag features (days)", 2, 10, 5)
    horizon = mc3.slider("Forecast horizon (days)", 1, 10, 5)

    if st.button("🚀 Train model & forecast", use_container_width=True):
        with st.spinner("Engineering features and training..."):
            feat = mlp.build_feature_set(raw, lookback_lags=lookback_lags)
            if len(feat) < 60:
                st.warning("Not enough history for reliable training -- pick a longer history window.")
            else:
                model, metrics, results, importances = mlp.train_model(feat, model_type, lookback_lags)
                forecast = mlp.forecast_forward(model, feat, n_days=horizon, lookback_lags=lookback_lags)

                st.session_state["ml_results"] = (metrics, results, importances, forecast)

    if "ml_results" in st.session_state:
        metrics, results, importances, forecast = st.session_state["ml_results"]

        st.markdown('<div class="section-header">Backtest Performance (held-out test window)</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"{metrics['RMSE']:.2f}")
        m2.metric("MAE", f"{metrics['MAE']:.2f}")
        m3.metric("R²", f"{metrics['R2']:.3f}")
        m4.metric("Directional Accuracy", f"{metrics['Directional Accuracy']*100:.1f}%")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=results.index, y=results["Target"], name="Actual (next-day close)", line=dict(color="#e6e9f0")))
        fig3.add_trace(go.Scatter(x=results.index, y=results["Predicted"], name="Predicted", line=dict(color=ACCENT, dash="dot")))
        fig3.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(t=20, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig3, use_container_width=True)

        colA, colB = st.columns([0.55, 0.45])
        with colA:
            st.markdown('<div class="section-header">Forward Forecast</div>', unsafe_allow_html=True)
            fig4 = go.Figure()
            tail = raw["Close"].tail(30)
            fig4.add_trace(go.Scatter(x=tail.index, y=tail.values, name="Recent Close", line=dict(color="#e6e9f0")))
            fig4.add_trace(go.Scatter(x=forecast.index, y=forecast["Predicted_Close"], name="Forecast",
                                       line=dict(color=GREEN, dash="dot"), mode="lines+markers"))
            fig4.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(t=20, b=10))
            st.plotly_chart(fig4, use_container_width=True)
            st.dataframe(forecast.style.format({"Predicted_Close": "{:.2f}"}), use_container_width=True)

        with colB:
            st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
            fig5 = go.Figure(go.Bar(
                x=importances.values[:10][::-1], y=importances.index[:10][::-1],
                orientation="h", marker_color=ACCENT,
            ))
            fig5.update_layout(template=PLOTLY_TEMPLATE, height=420, margin=dict(t=20, b=10))
            st.plotly_chart(fig5, use_container_width=True)

        st.caption(
            "⚠️ Educational project, not investment advice. Time-based split (no shuffling), "
            "regularized tree ensembles chosen deliberately to control overfitting on limited daily data."
        )

# ---------------- Options Analytics ----------------
with tab_opt:
    st.markdown('<div class="section-header">Black-Scholes Pricing & Greeks</div>', unsafe_allow_html=True)

    oc1, oc2, oc3 = st.columns(3)
    spot_default = float(last_close)
    S = oc1.number_input("Spot price (S)", value=spot_default, step=1.0)
    K = oc2.number_input("Strike price (K)", value=round(spot_default, 0), step=1.0)
    T_days = oc3.slider("Days to expiry", 1, 365, 30)
    T = T_days / 365

    oc4, oc5, oc6 = st.columns(3)
    r = oc4.slider("Risk-free rate (%)", 0.0, 15.0, 6.5) / 100
    sigma = oc5.slider("Implied volatility (%)", 1.0, 150.0, float(min(max((vol_20 or 0.25) * 100, 5), 100))) / 100
    option_type = oc6.selectbox("Option type", ["call", "put"])

    price = opx.black_scholes_price(S, K, T, r, sigma, option_type)
    g = opx.greeks(S, K, T, r, sigma, option_type)

    g1, g2, g3, g4, g5, g6 = st.columns(6)
    g1.metric("Theoretical Price", f"{price:,.2f}")
    g2.metric("Delta", f"{g['delta']:.3f}")
    g3.metric("Gamma", f"{g['gamma']:.4f}")
    g4.metric("Vega", f"{g['vega']:.3f}")
    g5.metric("Theta / day", f"{g['theta']:.3f}")
    g6.metric("Rho", f"{g['rho']:.3f}")

    st.markdown('<div class="section-header">Payoff Diagram</div>', unsafe_allow_html=True)
    price_range = np.linspace(S * 0.6, S * 1.4, 100)
    if option_type == "call":
        payoff = np.maximum(price_range - K, 0) - price
    else:
        payoff = np.maximum(K - price_range, 0) - price
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=price_range, y=payoff, line=dict(color=ACCENT, width=2), fill="tozeroy",
                               fillcolor="rgba(79,124,255,0.1)"))
    fig6.add_hline(y=0, line_color="#8a93a8", line_dash="dot")
    fig6.add_vline(x=S, line_color=GREEN, line_dash="dot", annotation_text="Spot")
    fig6.update_layout(template=PLOTLY_TEMPLATE, height=350, margin=dict(t=20, b=10),
                        xaxis_title="Underlying price at expiry", yaxis_title="P&L per contract")
    st.plotly_chart(fig6, use_container_width=True)

    if asset_class != "Upload CSV (Kaggle dataset)":
        st.markdown('<div class="section-header">Live Options Chain (if available)</div>', unsafe_allow_html=True)
        expiries = df_.get_option_expirations(ticker)
        if expiries:
            expiry = st.selectbox("Expiration date", expiries)
            calls, puts = df_.get_option_chain(ticker, expiry)
            cc, pc = st.columns(2)
            with cc:
                st.caption("Calls")
                st.dataframe(calls[["strike", "lastPrice", "bid", "ask", "impliedVolatility", "volume", "openInterest"]]
                             .sort_values("strike"), use_container_width=True, height=300)
            with pc:
                st.caption("Puts")
                st.dataframe(puts[["strike", "lastPrice", "bid", "ask", "impliedVolatility", "volume", "openInterest"]]
                             .sort_values("strike"), use_container_width=True, height=300)
        else:
            st.info("No listed options chain available for this instrument via the free data feed (common for pure futures contracts).")

# ---------------- Futures & Basis ----------------
with tab_fut:
    st.markdown('<div class="section-header">Cost-of-Carry Basis Check</div>', unsafe_allow_html=True)
    st.caption("Basis = Futures price − Spot price. Positive → contango, negative → backwardation.")

    fc1, fc2, fc3, fc4 = st.columns(4)
    spot_input = fc1.number_input("Spot / index price", value=spot_default, step=1.0)
    futures_input = fc2.number_input("Observed futures price", value=spot_default * 1.01, step=1.0)
    carry_r = fc3.slider("Financing rate (%)", 0.0, 15.0, 6.5, key="carry_r") / 100
    carry_days = fc4.slider("Days to futures expiry", 1, 365, 45, key="carry_days")

    dividend_q = st.slider("Dividend / convenience yield (%)", 0.0, 10.0, 1.2) / 100
    basis_info = fut.classify_basis(spot_input, futures_input)
    carry_check = fut.cost_of_carry_check(spot_input, futures_input, carry_r, dividend_q, carry_days / 365)

    pill_class = "pill-contango" if basis_info["state"] == "Contango" else "pill-backwardation"
    st.markdown(
        f'<span class="pill {pill_class}">{basis_info["state"]}</span> &nbsp; '
        f'Basis: **{basis_info["basis"]:,.2f}** ({basis_info["basis_pct"]*100:+.2f}%)',
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns(3)
    b1.metric("Theoretical Futures Price", f"{carry_check['theoretical_futures']:,.2f}")
    b2.metric("Observed Futures Price", f"{carry_check['observed_futures']:,.2f}")
    b3.metric("Mispricing vs. Fair Value", f"{carry_check['mispricing']:+,.2f} ({carry_check['mispricing_pct']*100:+.2f}%)")

    st.markdown('<div class="section-header">Cross-Asset Correlation (Futures Universe)</div>', unsafe_allow_html=True)
    if st.checkbox("Load correlation matrix across major futures (extra API calls)"):
        with st.spinner("Fetching futures universe..."):
            price_dict = {}
            for label, sym in list(df_.FUTURES_TICKERS.items())[:8]:
                h = df_.get_price_history(sym, period="1y", interval="1d")
                if not h.empty:
                    price_dict[label] = h["Close"]
            if price_dict:
                corr = fut.cross_asset_correlation(price_dict)
                fig7 = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                             colorscale="RdBu", zmid=0, zmin=-1, zmax=1))
                fig7.update_layout(template=PLOTLY_TEMPLATE, height=500, margin=dict(t=20, b=10))
                st.plotly_chart(fig7, use_container_width=True)

# ---------------- Risk ----------------
with tab_risk:
    st.markdown('<div class="section-header">Risk & Capital-Preservation Metrics</div>', unsafe_allow_html=True)

    benchmark_returns = None
    if asset_class != "Upload CSV (Kaggle dataset)":
        bench = df_.get_price_history("^GSPC", period=period, interval=interval)
        if not bench.empty:
            benchmark_returns = bench["Close"].pct_change()

    summary = risk.risk_summary(data, benchmark_returns)

    cols = st.columns(3)
    for i, (label, value) in enumerate(summary.items()):
        with cols[i % 3]:
            if "Ratio" in label:
                display = f"{value:.2f}" if pd.notna(value) else "—"
            elif "Beta" in label:
                display = f"{value:.2f}" if pd.notna(value) else "—"
            else:
                display = f"{value*100:.2f}%" if pd.notna(value) else "—"
            st.metric(label, display)

    st.markdown('<div class="section-header">Drawdown</div>', unsafe_allow_html=True)
    cumulative = data["Close"] / data["Close"].iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max - 1) * 100
    fig8 = go.Figure(go.Scatter(x=data.index, y=drawdown, fill="tozeroy", line=dict(color=RED),
                                 fillcolor="rgba(255,92,114,0.12)"))
    fig8.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(t=20, b=10), yaxis_title="Drawdown (%)")
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown('<div class="section-header">Return Distribution</div>', unsafe_allow_html=True)
    fig9 = go.Figure(go.Histogram(x=data["Daily_Return"].dropna() * 100, nbinsx=60, marker_color=ACCENT))
    fig9.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(t=20, b=10), xaxis_title="Daily return (%)")
    st.plotly_chart(fig9, use_container_width=True)

st.divider()
st.caption("Built with Python, Streamlit, scikit-learn, Plotly & yfinance • Educational project — not financial advice.")
