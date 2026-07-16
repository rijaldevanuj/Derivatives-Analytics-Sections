# Derivatives Analytics Platform

An interactive Streamlit app that blends **machine learning price forecasting**
with the core toolkit of a **futures/derivatives analyst**: technical analysis,
options pricing & Greeks, futures basis (contango/backwardation), and
risk/capital-preservation metrics.

Built as a predictive Modelling Project using Random Forest And Gradient Boosting for
international derivatives markets (equities, index futures, commodities, FX,
rates).


Live Website - https://derivatives-analytics-sections-3ppwxtglj7waftshosfr4x.streamlit.app/

## Why this project, and how it maps to the job description

| Features Provided | Where it shows up in the app |
|---|---|
| Fundamental, technical, statistical analysis | Technical Analysis tab: SMA/EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR — all implemented from first principles |
| Derivatives, futures, exchange mechanics | Futures & Basis tab: cost-of-carry model, contango/backwardation classification, cross-asset correlation across 8 major futures |
| Statistical/quantitative technique, strategy development | ML Prediction tab: engineered feature set, time-based train/test split, RandomForest/GradientBoosting, backtested directional accuracy, feature importances |
| Risk management and capital preservation | Risk tab: historical & parametric VaR, Sharpe/Sortino, max drawdown, beta vs. benchmark, return distribution |
| Options-specific analytical work | Options Analytics tab: Black-Scholes pricing, all five Greeks, implied-volatility solver, live options chain, payoff diagrams |
| Coding / data / quant skills | Entire app: Python, pandas/numpy, scikit-learn, scipy, Plotly, Streamlit, clean modular architecture |

## Architecture

```
futures-analytics-platform/
├── app.py                      # Streamlit UI (6 tabs, sidebar controls, custom dark theme)
├── modules/
│   ├── data_fetcher.py         # yfinance wrappers (equities, indices, futures, options chains) + CSV fallback
│   ├── indicators.py           # SMA/EMA/RSI/MACD/Bollinger/ATR/Stochastic, from scratch
│   ├── ml_predictor.py         # Feature engineering, model training, iterative forecasting
│   ├── options_pricing.py      # Black-Scholes, Greeks, implied volatility, cost-of-carry
│   ├── risk_metrics.py         # VaR, Sharpe, Sortino, drawdown, beta
│   └── futures_curve.py        # Basis classification, cross-asset correlation
└── requirements.txt
```

Each module is independent and unit-testable in isolation — deliberately
structured that way so it reads as production-style code, not a notebook
dumped into a script.

## Data sources (all free, no paid API keys)

- **Live/historical prices**: [`yfinance`](https://pypi.org/project/yfinance/)
  — covers equities, indices (`^GSPC`, `^NSEI`, `^VIX`, ...) and CME/NYMEX/COMEX
  continuous futures contracts (`ES=F`, `CL=F`, `GC=F`, `6E=F`, ...) with no key required.
- **Options chains**: pulled live via `yfinance` for listed equities/ETFs where available.
- **Offline / backtest mode**: upload any Kaggle historical OHLCV dataset
  (e.g. *"Huge Stock Market Dataset"*, *"S&P 500 stock data"*, *"Nifty 50 Stock Market Data"*)
  as a CSV — the app normalizes and runs the full pipeline against it with
  zero network dependency, which is handy for a live demo if internet access
  is unreliable during an interview.

## Modeling notes (things you should be ready to explain)

- **No shuffling in the train/test split.** Financial time series are
  temporally ordered; a random shuffle leaks future information into
  training and silently inflates every metric. The split here is strictly
  chronological.
- **Tree ensembles over an LSTM.** With a few hundred to a few thousand daily
  bars for a single ticker, a deep net is far more likely to overfit than to
  learn signal. RandomForest/GradientBoosting are more data-efficient, give
  interpretable feature importances, and are easier to defend under
  questioning — the same overfitting/regularization trade-off covered in any
  intro ML course, applied to a real, noisy domain.
- **Directional accuracy is reported alongside RMSE/MAE/R².** In trading,
  getting the *sign* of the next move right often matters more than the
  exact price level.
- **The forward forecast is explicitly labeled as a simplification** — it
  iteratively rolls the model forward using synthetic lag features rather
  than fully rebuilt OHLCV bars. That's a deliberate, disclosed limitation
  rather than an overstated claim, which is the right instinct for a market
  research/analyst role.
- **Basis/contango-backwardation uses a documented workaround.** Free data
  providers don't reliably expose a full multi-expiry futures term structure,
  so the app compares the observed futures price against a cost-of-carry
  theoretical price (`F = S · e^(r−q)T`) — the same relationship a desk
  uses, without fabricating data that isn't actually available for free.

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploying it for free (so you can put a live link on your resume)

[Streamlit Community Cloud](https://streamlit.io/cloud) will deploy directly
from a public GitHub repo at no cost:

1. Push this folder to a GitHub repo.
2. Go to share.streamlit.io → "New app" → point it at the repo and `app.py`.
3. Put the live URL in your resume/LinkedIn next to this project.

## Suggested resume line

> **Derivatives Analytics Platform** — Built an end-to-end Streamlit
> application combining ML-based price forecasting (RandomForest/GBM,
> time-series cross-validation), Black-Scholes options pricing & Greeks,
> futures basis/contango analysis, and portfolio risk metrics (VaR, Sharpe,
> max drawdown) across equities, indices, and 10+ global futures contracts,
> using free real-time market data (yfinance) and Kaggle historical datasets.

## Not investment advice

This is an educational/portfolio project. Forecasts, Greeks, and risk metrics
are illustrative and should not be used to make real trading decisions.
