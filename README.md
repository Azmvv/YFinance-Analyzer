<p align="center">
  <h1 align="center">📈 YFinance Analyzer</h1>
  <p align="center">
    <strong>Predict financial asset prices using deep learning — powered by LSTM neural networks.</strong>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> · <a href="#-features">Features</a> · <a href="#%EF%B8%8F-how-it-works">How It Works</a> · <a href="#-example">Example</a> · <a href="#-license">License</a>
  </p>
</p>

---

YFinance Analyzer is a command-line tool that fetches real-time market data, engineers technical indicators, trains an LSTM model on historical prices, and produces a clean forecast chart — all in a single script. Just pick a ticker, give it a date range, and let the model do the rest.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/Azmvv/YFinance-Analyzer.git
cd YFinance-Analyzer

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
python3 predictor.py
```

You'll be prompted for three inputs:

| Prompt | Example | Description |
|--------|---------|-------------|
| **Ticker symbol** | `BTC-USD`, `AAPL`, `TSLA` | Any valid Yahoo Finance ticker |
| **Start date** | `2020-01-01` | Beginning of the historical window |
| **End date** | `2025-12-31` | End of the historical window |

The script handles everything from there — data download, feature engineering, model training, evaluation, and chart generation.

## ✨ Features

- **Any Yahoo Finance asset** — stocks, crypto, ETFs, indices, forex pairs, and more.
- **Technical indicators** — automatically computes a 14-day RSI and 7-day Moving Average to give the model richer context.
- **Sliding-window sequences** — uses the past 60 days of multi-feature data to predict the next day's closing price.
- **Two-layer LSTM with Dropout** — deep enough to capture temporal patterns, regularized to avoid overfitting.
- **Early stopping** — training halts automatically when validation loss stops improving, saving time and preventing overfit.
- **7-day future forecast** — after evaluation, the model rolls forward to predict the next 7 business days beyond your dataset.
- **Publication-quality chart** — a single PNG with actual prices, test-set predictions, and the forecast line, complete with legend, grid, and formatted dates.

## ⚙️ How It Works

```
User Input           Data Pipeline              Model                  Output
───────────     ─────────────────────     ─────────────────     ─────────────────
 Ticker    ──▶  yfinance download    ──▶  LSTM (128 → 64)  ──▶  Test predictions
 Start date     + RSI, 7-day MA            + Dropout 0.3         Future forecast
 End date       + MinMaxScaler             + EarlyStopping        Saved PNG chart
                + 60-day windows           + Adam / MSE           MAE & MAPE
```

### Step-by-step

1. **Fetch** — Historical OHLCV data is downloaded from Yahoo Finance via the `yfinance` library.
2. **Engineer** — Two technical indicators are computed and appended as new columns:
   - **7-day Moving Average** — smooths short-term price noise.
   - **14-day RSI** — measures momentum (overbought/oversold conditions).
3. **Scale** — All features (`Close`, `Volume`, `MA_7`, `RSI_14`) are normalized to [0, 1] using `MinMaxScaler`.
4. **Window** — Data is reshaped into overlapping 60-day sequences, each paired with the next day's closing price as the label.
5. **Split** — 80% of sequences go to training, 20% to testing (chronological, no shuffling).
6. **Train** — A two-layer LSTM network trains for up to 50 epochs with early stopping (patience = 8).
7. **Evaluate** — The model predicts on the held-out test set. **MAE** and **MAPE** are printed to the console.
8. **Forecast** — The last 60-day window is fed back into the model iteratively to generate 7 future price points.
9. **Plot** — Everything is visualized on a single matplotlib chart and saved as a high-resolution PNG.

### Model Architecture

```
Layer               Output Shape         Parameters
─────────────────────────────────────────────────────
LSTM (128 units)    (batch, 60, 128)     68,096
Dropout (0.3)       (batch, 60, 128)     0
LSTM (64 units)     (batch, 64)          49,408
Dropout (0.3)       (batch, 64)          0
Dense (32, ReLU)    (batch, 32)          2,080
Dense (1, linear)   (batch, 1)           33
─────────────────────────────────────────────────────
Total trainable parameters: ~119,617
```

## 📊 Example

```
$ python3 predictor.py
Enter ticker symbol (e.g. BTC-USD, AAPL): AAPL
Enter start date (YYYY-MM-DD): 2020-01-01
Enter end date   (YYYY-MM-DD): 2025-12-31

📥  Downloading AAPL data from 2020-01-01 to 2025-12-31 …
✅  Received 1,487 rows after feature engineering.

🔢  Training samples : 1,142
🔢  Testing samples  : 286

🚀  Training LSTM model …
Epoch 1/50 ━━━━━━━━━━━━━━━━━━ 33/33 - loss: 0.0042 - val_loss: 0.0018
...
Epoch 19/50 ━━━━━━━━━━━━━━━━━━ 33/33 - loss: 0.0005 - val_loss: 0.0004

📈  Test MAE  : 4.32
📈  Test MAPE : 1.87%

🔮  7-Day Forecast from 2026-01-02:
    2026-01-02  →  $248.71
    2026-01-03  →  $249.15
    ...
```

A chart like this is saved automatically:

> **`AAPL_prediction.png`** — Actual vs. Predicted prices with a 7-day forecast extension.

## 🗂 Project Structure

```
YFinance-Analyzer/
├── predictor.py        # Main script — data, model, training, plotting
├── requirements.txt    # Python dependencies
├── LICENSE             # AGPL-3.0
└── README.md           # You are here
```

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Data manipulation & time series |
| `yfinance` | Yahoo Finance market data API |
| `scikit-learn` | `MinMaxScaler` for feature normalization |
| `tensorflow` | LSTM model (Keras API) |
| `matplotlib` | Chart generation |

## ⚠️ Disclaimer

This tool is built for **educational and research purposes only**. Financial markets are inherently unpredictable — no model can guarantee future returns. Do not use these predictions as the sole basis for real trading or investment decisions. Always do your own research and consult a qualified financial advisor.

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0** — see the [LICENSE](LICENSE) file for details.
