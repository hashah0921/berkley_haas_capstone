"""
REST API for the Cryptocurrency Market Prediction capstone.

Serves the same data/feature/model code as app.py (both import crypto_lib.py), so the Streamlit
UI and this API never drift out of sync — they call the exact same train_and_evaluate().

Run with:
    uvicorn api:app --reload --port 8000

Then see interactive docs at http://127.0.0.1:8000/docs, or e.g.:
    curl http://127.0.0.1:8000/coins
    curl "http://127.0.0.1:8000/predict/Bitcoin?horizon_days=1&n_search_iter=15"
"""
from datetime import date
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import crypto_lib as lib

app = FastAPI(
    title="Crypto Market Prediction API",
    description=(
        "Predicts N-day-ahead price direction for 23 real daily-coin datasets, training a "
        "lightly hyperparameter-tuned XGBoost classifier on demand. Same logic as app.py "
        "(Streamlit) and the grid-search section of 2_crypto_market_prediction.ipynb. "
        "⚠️ For demonstration only — not the capstone's final reported result "
        "(see 2_crypto_market_prediction.ipynb for the tuned XGBoost+GRU ensemble)."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class CoinInfo(BaseModel):
    name: str
    rows: int
    start_date: date
    end_date: date
    last_close: float


class PredictResponse(BaseModel):
    coin: str
    horizon_days: int
    used_sentiment: bool
    n_search_iter: int
    test_accuracy: float
    test_auc: float
    latest_date: date
    probability_up: float
    best_params: dict
    top_features: dict[str, float]
    disclaimer: str = (
        "Demo model retrained on the fly with a small hyperparameter search; not the capstone's "
        "final reported result and not financial advice."
    )


class SentimentPoint(BaseModel):
    date: date
    sentiment_mean: float
    sentiment_std: float
    n_headlines: int
    synthetic_placeholder: bool = True


# ---------------------------------------------------------------------------
# Cached training — avoid retraining on every request for the same inputs.
# lru_cache needs hashable args, which (coin, horizon_days, use_sentiment, n_search_iter) are.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=128)
def _cached_train(coin_path: str, horizon_days: int, use_sentiment: bool, n_search_iter: int):
    return lib.train_and_evaluate(coin_path, horizon_days, use_sentiment, n_search_iter)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/coins", response_model=list[CoinInfo])
def get_coins():
    """List every coin with real daily data, its row count, date range and last close."""
    out = []
    for name, path in lib.list_coins().items():
        df = lib.load_coin(path)
        out.append(CoinInfo(
            name=name, rows=len(df),
            start_date=df.index.min().date(), end_date=df.index.max().date(),
            last_close=float(df["Close"].iloc[-1]),
        ))
    return sorted(out, key=lambda c: c.name)


def _resolve_coin_path(coin: str) -> str:
    coins = lib.list_coins()
    # case-insensitive match so /predict/bitcoin works like /predict/Bitcoin
    match = next((name for name in coins if name.lower() == coin.lower()), None)
    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown coin '{coin}'. See GET /coins for valid names.",
        )
    return coins[match], match


@app.get("/coins/{coin}/price")
def get_price(coin: str, limit: int = Query(100, ge=1, le=5000, description="Most recent N rows")):
    """Recent daily OHLCV rows for one coin."""
    path, name = _resolve_coin_path(coin)
    df = lib.load_coin(path).tail(limit).reset_index()
    df["Date"] = df["Date"].dt.date.astype(str)
    return {"coin": name, "rows": df.to_dict(orient="records")}


@app.get("/predict/{coin}", response_model=PredictResponse)
def predict(
    coin: str,
    horizon_days: int = Query(1, ge=1, le=30, description="Predict direction this many days ahead"),
    use_sentiment: bool = Query(False, description="Include the news-sentiment feature (needs data/news_sentiment_daily.csv)"),
    n_search_iter: int = Query(15, ge=1, le=60, description="RandomizedSearchCV iterations — more = slower, better-tuned"),
):
    """
    Train (or reuse a cached run of) a lightly-tuned XGBoost classifier for `coin` and return its
    test-set accuracy/AUC, today's predicted P(price up) at `horizon_days` ahead, and feature
    importances. Demo-only — see the `disclaimer` field.
    """
    path, name = _resolve_coin_path(coin)
    result = _cached_train(path, horizon_days, use_sentiment, n_search_iter)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    top_features = result["importances"].head(10).round(4).to_dict()
    return PredictResponse(
        coin=name,
        horizon_days=result["horizon_days"],
        used_sentiment=use_sentiment,
        n_search_iter=n_search_iter,
        test_accuracy=round(result["accuracy"], 4),
        test_auc=round(result["auc"], 4),
        latest_date=result["latest_date"].date(),
        probability_up=round(result["latest_proba_up"], 4),
        best_params=result["best_params"],
        top_features=top_features,
    )


@app.get("/sentiment/latest", response_model=SentimentPoint)
def sentiment_latest():
    """Most recent daily sentiment score from data/news_sentiment_daily.csv, if it exists."""
    sent = lib.load_sentiment()
    if sent is None:
        raise HTTPException(
            status_code=404,
            detail="No sentiment file found. Run 3_news_sentiment_analysis.ipynb first to generate data/news_sentiment_daily.csv.",
        )
    row = sent.iloc[-1]
    return SentimentPoint(
        date=sent.index[-1].date(),
        sentiment_mean=float(row["sentiment_mean"]),
        sentiment_std=float(row["sentiment_std"]),
        n_headlines=int(row["n_headlines"]),
    )
