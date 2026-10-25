"""
Streamlit demo for the Cryptocurrency Market Prediction capstone.

Ties together the three notebooks in this repo:
  - 1_crypto_exploratory_data_analysis.ipynb : real daily price data (data/coin_*.csv)
  - 3_news_sentiment_analysis.ipynb          : optional data/news_sentiment_daily.csv feature
  - 2_crypto_market_prediction.ipynb           : the feature-engineering + XGBoost approach, retrained
                                                 live here on whichever coin the user picks

Shares its data/feature/model code with api.py via crypto_lib.py.

Run with:
    streamlit run app.py
"""
import os

import matplotlib.pyplot as plt
import streamlit as st

import crypto_lib as lib

st.set_page_config(page_title="Crypto Market Prediction", layout="wide")

list_coins = st.cache_data(lib.list_coins)
load_coin = st.cache_data(lib.load_coin)
load_sentiment = st.cache_data(lib.load_sentiment)
train_and_evaluate = st.cache_data(show_spinner=False)(lib.train_and_evaluate)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("Crypto Market Prediction")
coins = list_coins()
coin_names = sorted(coins.keys())
coin_name = st.sidebar.selectbox("Coin", coin_names, index=coin_names.index("Bitcoin") if "Bitcoin" in coins else 0)
horizon = st.sidebar.slider("Prediction horizon (days ahead)", 1, 14, 1)
use_sentiment = st.sidebar.checkbox(
    "Include news-sentiment feature",
    value=os.path.exists(f"{lib.DATA_DIR}/news_sentiment_daily.csv"),
    help="Needs data/news_sentiment_daily.csv — produced by 3_news_sentiment_analysis.ipynb (synthetic placeholder unless you've supplied real headlines).",
)
n_iter = st.sidebar.slider("Grid-search iterations", 5, 40, 15, help="More = slower but a better-tuned model.")
st.sidebar.caption(
    "⚠️ Daily per-coin files are real Kaggle data. The 15-minute series used in "
    "`2_crypto_market_prediction.ipynb` is still a synthetic placeholder — this app only uses the daily files."
)
st.sidebar.caption("Same model also served as JSON via `api.py` — see README's API section.")

st.title(f"{coin_name} — price, sentiment & directional forecast")

# ---------------------------------------------------------------------------
# Price + sentiment chart
# ---------------------------------------------------------------------------
df = load_coin(coins[coin_name])
sentiment = load_sentiment()

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Price history")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df.index, df["Close"])
    ax.set_yscale("log")
    ax.set_ylabel("Close (log scale)")
    st.pyplot(fig)

with col2:
    st.subheader("Snapshot")
    st.metric("Last close", f"${df['Close'].iloc[-1]:,.2f}")
    st.metric("Rows of history", f"{len(df):,}")
    st.metric("Date range", f"{df.index.min().date()} → {df.index.max().date()}")

if sentiment is not None:
    st.subheader("Daily news sentiment (VADER, from 3_news_sentiment_analysis.ipynb)")
    aligned = sentiment.reindex(df.index).dropna()
    if len(aligned):
        fig2, ax2 = plt.subplots(figsize=(10, 2.5))
        ax2.plot(aligned.index, aligned["sentiment_mean"])
        ax2.axhline(0, color="gray", ls="--")
        st.pyplot(fig2)
    st.caption("Synthetic placeholder unless you've replaced data/news_headlines.csv with real headlines before re-running the sentiment notebook.")
else:
    st.info("No sentiment file found. Run `3_news_sentiment_analysis.ipynb` to generate `data/news_sentiment_daily.csv`, then re-check the sidebar box.")

# ---------------------------------------------------------------------------
# Model: train on the fly (cached per coin/horizon/sentiment/n_iter combo)
# ---------------------------------------------------------------------------
st.subheader(f"XGBoost model — {horizon}-day-ahead direction")
with st.spinner("Tuning hyperparameters and training..."):
    result = train_and_evaluate(coins[coin_name], horizon, use_sentiment, n_iter)

if "error" in result:
    st.error(result["error"])
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{result['accuracy']:.1%}")
    c2.metric("Test ROC AUC", f"{result['auc']:.3f}")
    c3.metric(
        f"P(up) as of {result['latest_date'].date()}",
        f"{result['latest_proba_up']:.1%}",
    )
    st.caption(f"Best params from this run's grid search: {result['best_params']}")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Feature importance**")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        result["importances"].sort_values().plot.barh(ax=ax3)
        st.pyplot(fig3)
    with colB:
        st.markdown("**Predicted probability vs. actual direction (test set)**")
        te = result["test"]
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        ax4.scatter(range(len(te)), result["p_test"], c=te["y_dir"], cmap="coolwarm", s=12, alpha=0.7)
        ax4.axhline(0.5, color="gray", ls="--")
        ax4.set_xlabel("test-set row (chronological)"); ax4.set_ylabel("P(up)")
        st.pyplot(fig4)

    st.warning(
        "This model is retrained from scratch on whatever coin/horizon you pick, with a small hyperparameter "
        "search, purely for interactive demonstration. It is not the same tuned run as "
        "`2_crypto_market_prediction.ipynb` (which also includes a GRU and an ensemble) and should not be read "
        "as the capstone's final reported result."
    )
