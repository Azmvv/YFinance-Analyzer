#!/usr/bin/env python3
"""
Financial Price Prediction Tool
================================
Uses LSTM neural networks to predict financial asset prices.
Fetches data via yfinance, engineers features (RSI, 7-day MA),
applies a sliding-window approach, and forecasts future prices.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────── helpers ────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data and add RSI + 7-day Moving Average."""
    print(f"\n📥  Downloading {ticker} data from {start} to {end} …")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data returned. Check the ticker symbol and date range.")

    # Flatten MultiIndex columns if present (yfinance >= 0.2)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["RSI_14"] = compute_rsi(df["Close"], period=14)
    df.dropna(inplace=True)
    print(f"✅  Received {len(df)} rows after feature engineering.")
    return df


# ──────────────── sliding-window dataset builder ─────────────────

WINDOW = 60  # look-back window (days)
HORIZON = 1  # predict next N days


def build_datasets(df: pd.DataFrame, feature_cols: list[str], target_col: str = "Close"):
    """
    Scale features, create sliding-window sequences, and split into
    train / test sets (80 / 20).
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)

    # Separate scaler for the target so we can inverse-transform predictions
    target_idx = feature_cols.index(target_col)
    target_scaler = MinMaxScaler()
    target_scaler.fit(df[[target_col]].values)

    X, y = [], []
    for i in range(WINDOW, len(scaled) - HORIZON + 1):
        X.append(scaled[i - WINDOW : i])
        y.append(scaled[i, target_idx])  # next-day close (scaled)

    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Dates aligned to y values
    dates = df.index[WINDOW : WINDOW + len(y)]
    train_dates = dates[:split]
    test_dates = dates[split:]

    return (X_train, y_train, X_test, y_test,
            train_dates, test_dates, scaler, target_scaler, scaled)


# ──────────────────────────── model ──────────────────────────────

def build_model(input_shape: tuple) -> Sequential:
    """Two-layer LSTM with Dropout."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ────────────────────── future forecasting ───────────────────────

FUTURE_DAYS = 7


def forecast_future(model, last_window: np.ndarray, target_idx: int,
                    n_features: int, days: int = FUTURE_DAYS):
    """Iteratively predict the next `days` values."""
    current = last_window.copy()
    preds = []
    for _ in range(days):
        pred = model.predict(current[np.newaxis, :, :], verbose=0)[0, 0]
        preds.append(pred)
        # Shift window: drop oldest row, append new row (fill non-target cols
        # with last known values and set target col to prediction)
        new_row = current[-1].copy()
        new_row[target_idx] = pred
        current = np.vstack([current[1:], new_row])
    return np.array(preds)


# ──────────────────────── visualisation ──────────────────────────

def plot_results(df, train_dates, test_dates, y_test_actual,
                 y_pred, future_dates, future_prices, ticker):
    """Produce a publication-quality chart."""
    fig, ax = plt.subplots(figsize=(16, 7))

    # Full historical close
    ax.plot(df.index, df["Close"], color="#1f77b4", linewidth=1.2,
            label="Actual Price", alpha=0.85)

    # Predicted on test set
    ax.plot(test_dates, y_pred, color="#ff7f0e", linewidth=1.6,
            label="Predicted (Test)", linestyle="--")

    # Future forecast
    ax.plot(future_dates, future_prices, color="#2ca02c", linewidth=2,
            label=f"Future Forecast ({FUTURE_DAYS} days)", linestyle="-.",
            marker="o", markersize=4)

    # Shade test region
    ax.axvspan(test_dates[0], test_dates[-1], alpha=0.06, color="orange",
               label="Test Period")

    # Formatting
    ax.set_title(f"{ticker}  —  LSTM Price Prediction & 7-Day Forecast",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=35)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    out_path = f"{ticker.replace('-', '_')}_prediction.png"
    plt.savefig(out_path, dpi=200)
    print(f"\n📊  Chart saved to {out_path}")
    plt.show()


# ──────────────────────────── main ───────────────────────────────

def main():
    # ── 1. User Inputs ──
    ticker = input("Enter ticker symbol (e.g. BTC-USD, AAPL): ").strip()
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date   (YYYY-MM-DD): ").strip()

    # ── 2. Fetch & engineer features ──
    df = fetch_data(ticker, start_date, end_date)
    feature_cols = ["Close", "Volume", "MA_7", "RSI_14"]

    # ── 3. Build sliding-window datasets ──
    (X_train, y_train, X_test, y_test,
     train_dates, test_dates,
     scaler, target_scaler, scaled) = build_datasets(df, feature_cols)

    print(f"\n🔢  Training samples : {len(X_train)}")
    print(f"🔢  Testing samples  : {len(X_test)}")

    # ── 4. Build & train LSTM ──
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=8,
                               restore_best_weights=True)

    print("\n🚀  Training LSTM model …")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )

    # ── 5. Evaluate on test set ──
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = np.mean(np.abs(y_test_actual - y_pred))
    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
    print(f"\n📈  Test MAE  : {mae:,.2f}")
    print(f"📈  Test MAPE : {mape:.2f}%")

    # ── 6. Future 7-day forecast ──
    target_idx = feature_cols.index("Close")
    last_window = scaled[-WINDOW:]
    future_scaled = forecast_future(model, last_window, target_idx,
                                    n_features=len(feature_cols))
    future_prices = target_scaler.inverse_transform(
        future_scaled.reshape(-1, 1)).flatten()

    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                  periods=FUTURE_DAYS)

    print(f"\n🔮  7-Day Forecast from {future_dates[0].strftime('%Y-%m-%d')}:")
    for d, p in zip(future_dates, future_prices):
        print(f"    {d.strftime('%Y-%m-%d')}  →  ${p:,.2f}")

    # ── 7. Plot ──
    plot_results(df, train_dates, test_dates, y_test_actual,
                 y_pred, future_dates, future_prices, ticker)


if __name__ == "__main__":
    main()
