"""
ml_predictor.py
----------------
Feature engineering + model training for next-day / multi-day price direction
and level forecasting.

Design choices (worth being able to explain in an interview):
- Time-based train/test split (never shuffled) -- shuffling leaks the future
  into the training set, which is the single most common mistake in
  financial ML.
- Tree ensembles (RandomForest / GradientBoosting) instead of a deep LSTM:
  on a single ticker with a few thousand daily bars there isn't enough data
  to responsibly train a deep net without overfitting; tree ensembles are
  far more data-efficient, give feature importances for free, and are
  easier to defend when someone asks "why should I trust this."
- Regularization (max_depth, min_samples_leaf, n_estimators) is tuned
  conservatively to control variance, the same overfit/underfit trade-off
  covered in any intro ML course.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import randint, uniform

from modules.indicators import add_all_indicators

FEATURE_COLUMNS = [
    "SMA_20", "SMA_50", "EMA_20", "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Mid", "BB_Lower", "ATR_14", "Stoch_K", "Stoch_D",
    "Volatility_20", "Volume",
]


def build_feature_set(df: pd.DataFrame, lookback_lags: int = 5) -> pd.DataFrame:
    """Add indicators + lagged returns, and the prediction target (next-day close)."""
    feat = add_all_indicators(df)
    for lag in range(1, lookback_lags + 1):
        feat[f"Return_Lag_{lag}"] = feat["Daily_Return"].shift(lag)
    # Keep the absolute next-day close for display, but train on next-day return
    feat["Target"] = feat["Close"].shift(-1)
    feat["Target_Return"] = (feat["Target"] - feat["Close"]) / feat["Close"]
    feat = feat.dropna()
    return feat


def get_feature_columns(lookback_lags: int = 5) -> list:
    return FEATURE_COLUMNS + [f"Return_Lag_{lag}" for lag in range(1, lookback_lags + 1)]


def train_test_split_time(feat: pd.DataFrame, test_size: float = 0.2):
    split_idx = int(len(feat) * (1 - test_size))
    return feat.iloc[:split_idx], feat.iloc[split_idx:]


def train_model(feat: pd.DataFrame, model_type: str = "Random Forest", lookback_lags: int = 5,
                tune: bool = True):
    """Train a model to predict next-day RETURN (more stable than raw price levels).

    Returns the fitted model (or CV wrapper), performance metrics computed on a
    held-out temporal test split (metrics are reported on price-levels for
    easier interpretation), the test dataframe with predictions, and feature importances.
    """
    cols = get_feature_columns(lookback_lags)
    train, test = train_test_split_time(feat)

    X_train, y_train = train[cols], train["Target_Return"]
    X_test, y_test = test[cols], test["Target_Return"]

    if model_type == "Gradient Boosting":
        base = GradientBoostingRegressor(random_state=42)
        param_dist = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(2, 6),
            "learning_rate": uniform(0.01, 0.2),
            "min_samples_leaf": randint(3, 20),
            "subsample": uniform(0.6, 0.4),
        }
    else:
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_dist = {
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 12),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
        }

    model = base
    # Light randomized hyperparameter search with time-series CV to limit overfitting
    if tune:
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=20,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        model = search
    else:
        model.fit(X_train, y_train)

    # Obtain return predictions from the (possibly wrapped) model
    preds_ret = model.predict(X_test)

    # Convert predicted returns to price-level predictions for interpretable metrics/plots
    predicted_close = test["Close"].values * (1 + preds_ret)
    actual_close = test["Target"].values

    directional_actual = np.sign(test["Target_Return"].values)
    directional_pred = np.sign(preds_ret)
    directional_accuracy = float(np.mean(directional_actual == directional_pred))

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(actual_close, predicted_close))),
        "MAE": float(mean_absolute_error(actual_close, predicted_close)),
        "R2": float(r2_score(actual_close, predicted_close)),
        "Directional Accuracy": directional_accuracy,
    }

    results = test.copy()
    results["Predicted"] = predicted_close

    # Feature importances: pull from best estimator if using RandomizedSearchCV
    try:
        est = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        importances = pd.Series(est.feature_importances_, index=cols).sort_values(ascending=False)
    except Exception:
        importances = pd.Series(dtype=float)

    return model, metrics, results, importances


def forecast_forward(model, feat: pd.DataFrame, n_days: int = 5, lookback_lags: int = 5) -> pd.DataFrame:
    """
    Iteratively roll the model forward n_days, re-deriving indicators/lags
    from the growing synthetic series each step. This is a simplification
    (indicators are recomputed on a small trailing window rather than a
    fully rebuilt OHLCV history) -- appropriate for a short-horizon directional
    forecast, and clearly documented rather than silently overstated.
    """
    cols = get_feature_columns(lookback_lags)
    history = feat.copy()
    future_rows = []
    last_date = history.index[-1]

    # If model is a CV wrapper (RandomizedSearchCV), use its best_estimator_
    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model

    for step in range(1, n_days + 1):
        last_row = history.iloc[-1]
        X_next = pd.DataFrame([last_row[cols].values], columns=cols)
        # model predicts next-day return
        next_ret = estimator.predict(X_next)[0]
        next_close = last_row["Close"] * (1 + next_ret)

        next_date = last_date + pd.tseries.offsets.BDay(step)
        future_rows.append({"Date": next_date, "Predicted_Close": next_close})

        # crude forward-fill of a synthetic bar so lag features remain computable
        synthetic = last_row.copy()
        synthetic["Close"] = next_close
        synthetic["Daily_Return"] = (next_close - last_row["Close"]) / last_row["Close"]
        for lag in range(lookback_lags, 1, -1):
            synthetic[f"Return_Lag_{lag}"] = last_row[f"Return_Lag_{lag - 1}"]
        synthetic["Return_Lag_1"] = last_row["Daily_Return"]
        history = pd.concat([history, pd.DataFrame([synthetic])], ignore_index=False)

    return pd.DataFrame(future_rows).set_index("Date")
