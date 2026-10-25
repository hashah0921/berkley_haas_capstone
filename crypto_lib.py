"""
Shared data-loading, feature-engineering and modeling logic for the Cryptocurrency Market
Prediction capstone. Used by both `app.py` (Streamlit) and `api.py` (FastAPI) so the two don't
drift out of sync. No Streamlit or FastAPI imports live here — plain pandas/sklearn/xgboost only.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

DATA_DIR = "data"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def list_coins():
    """{coin_name: csv_path} for every real daily coin file (excludes the 15-min placeholder)."""
    files = sorted(
        f for f in glob.glob(f"{DATA_DIR}/coin_*.csv")
        if os.path.basename(f) != "coin_price_15min.csv"
    )
    return {os.path.basename(f).replace("coin_", "").replace(".csv", ""): f for f in files}


def load_coin(path):
    coin_df = pd.read_csv(path)
    coin_df["Date"] = pd.to_datetime(coin_df["Date"]).dt.normalize()
    coin_df = coin_df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    return coin_df[["Open", "High", "Low", "Close", "Volume", "Marketcap"]].astype(float)


def load_sentiment():
    path = f"{DATA_DIR}/news_sentiment_daily.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


# ---------------------------------------------------------------------------
# Feature engineering — same indicators as 2_crypto_market_prediction.ipynb, adapted to daily bars
# (that notebook predicts 96 bars of 15-min data = 24h ahead; here we predict N trading days ahead).
# ---------------------------------------------------------------------------
def rsi(series, period=14):
    """Relative Strength Index: ratio of average gains to average losses over `period` bars."""
    delta = series.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + avg_gain / (avg_loss + 1e-12))


def make_features(daily_df, sentiment=None):
    feat = pd.DataFrame(index=daily_df.index)
    close = daily_df["Close"]
    # momentum: returns over 1/3/7/14-day lookbacks
    for lookback in (1, 3, 7, 14):
        feat[f"ret_{lookback}"] = close.pct_change(lookback)
    feat["rsi_14"] = rsi(close)
    # trend: MACD (difference of two EMAs) and its signal line
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    feat["macd"] = (ema12 - ema26) / close
    feat["macd_signal"] = feat["macd"].ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]
    # trend: how far price has drifted from each EMA, as a fraction of price
    for span in (10, 20, 50):
        feat[f"ema_gap_{span}"] = close / close.ewm(span=span, adjust=False).mean() - 1
    # volatility: rolling std of 1-day returns over short/long windows
    ret_1day = close.pct_change()
    for window in (7, 30):
        feat[f"vol_{window}"] = ret_1day.rolling(window).std()
    feat["range_hl"] = (daily_df["High"] - daily_df["Low"]) / close
    feat["volume_spike"] = daily_df["Volume"] / daily_df["Volume"].rolling(30).mean()
    if sentiment is not None:
        feat = feat.join(sentiment, how="left")
        feat[sentiment.columns] = feat[sentiment.columns].fillna(0.0)
    return feat


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.02, 0.05, 0.08],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
}


def train_and_evaluate(coin_path, horizon_days, use_sentiment, n_search_iter):
    """Train a fresh, lightly-tuned XGBoost classifier for `horizon_days`-ahead direction on one
    coin's daily data. Returns a plain dict (JSON- and Streamlit-friendly) or {"error": ...}."""
    daily_df = load_coin(coin_path)
    sentiment = load_sentiment() if use_sentiment else None
    features_df = make_features(daily_df, sentiment=sentiment)
    fwd_ret = daily_df["Close"].shift(-horizon_days) / daily_df["Close"] - 1
    data = features_df.assign(fwd_ret=fwd_ret, y_dir=(fwd_ret > 0).astype(int))
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    feature_cols = list(features_df.columns)

    n_rows = len(data)
    if n_rows < 150:
        return {"error": f"Only {n_rows} usable rows after feature/target construction — pick a coin with more history."}

    gap = horizon_days  # embargo: leaves a full horizon between train and test so targets never overlap
    train_end, val_end = int(n_rows * 0.70), int(n_rows * 0.85)
    train, test = data.iloc[: train_end - gap], data.iloc[val_end:]
    if len(train) < 60 or len(test) < 20:
        return {"error": f"Not enough rows per split ({len(train)} train / {len(test)} test) for this coin/horizon."}

    X_train, X_test = train[feature_cols].values, test[feature_cols].values

    time_series_cv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        PARAM_DIST, n_iter=n_search_iter, scoring="roc_auc", cv=time_series_cv, random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, train["y_dir"])
    model = search.best_estimator_

    test_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(test["y_dir"], (test_proba > 0.5).astype(int))
    auc = roc_auc_score(test["y_dir"], test_proba)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    latest_features = features_df[feature_cols].iloc[[-1]].ffill().fillna(0.0)
    latest_proba_up = float(model.predict_proba(latest_features.values)[0, 1])

    return {
        "df": daily_df, "data": data, "test": test, "p_test": test_proba,
        "accuracy": float(accuracy), "auc": float(auc), "best_params": search.best_params_,
        "importances": importances, "latest_proba_up": latest_proba_up,
        "latest_date": features_df.index[-1], "horizon_days": horizon_days,
    }
