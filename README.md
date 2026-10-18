# Capstone: Cryptocurrency Market Prediction

Claude-generated starter based on the capstone proposal (Google Doc).

## Research question
Can 15-minute price sequence data combined with engineered momentum indicators and market metrics predict 24-hour directional price movement and returns for major cryptocurrencies?

## Contents
| File | Purpose |
|---|---|
| `crypto_market_prediction.ipynb` | End-to-end pipeline: load, features, target, split, baselines, XGBoost, GRU, ensemble, interpretability |
| `requirements.txt` | Python dependencies |
| `data/` | Put the Kaggle CSV here (not committed) |

## Data (Kaggle)
- Coin Price Forecasting Challenge: 15-minute OHLCV, 4h to 24h horizons (primary source for the notebook)
- G-Research Crypto Forecasting: minute-level, Bitcoin/Ethereum and others
- Crypto Datasets: 196 Pairs 1-Min Trading Data
- Cryptocurrency Historical Prices (daily): long-term market cap and price

Save the 15-minute file as `data/coin_price_15min.csv`, then adjust `COLUMN_MAP`, `ASSET_COL` and `ASSET` in the notebook config cell to match its schema. 1-minute sources can be resampled to 15 minutes. **If no file is found, the notebook falls back to synthetic data** so it still runs; those results are meaningless for real markets.

## Method
- **Features:** multi-horizon returns, RSI, MACD, EMA gaps, rolling volatility, high-low range, volume spike ratio, hour-of-day.
- **Target:** binary 24h direction (`y_dir`) and 24h forward return (`fwd_ret`), 96 bars ahead.
- **Split:** chronological 70/15/15 with a 96-bar embargo to avoid overlapping-target leakage.
- **Baselines:** random walk / majority class, naive momentum, logistic + ridge.
- **Models:** XGBoost (classifier + regressor), GRU over 32-bar windows (optional, needs PyTorch), averaged ensemble.
- **Interpretability:** permutation importance and SHAP.
- **Success criteria (from proposal):** directional accuracy above 55-60% and lower MAE than baselines.

## Run
```bash
cd berkley_haas_module20_claudegenerated
pip install -r requirements.txt
jupyter notebook crypto_market_prediction.ipynb
```

## Caveats
- The notebook has not been run against real Kaggle data; column names are assumptions to verify.
- Transaction costs, slippage and regime changes are not modeled; directional accuracy alone does not imply a profitable strategy.
