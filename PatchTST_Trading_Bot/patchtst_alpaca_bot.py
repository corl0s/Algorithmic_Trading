import os
import math
import logging
import datetime as dt
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Alpaca
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

@dataclass
class Config:
    TICKERS = ["AAPL", "MSFT", "AMZN", "GOOG"]

    MODEL_PATH = "models/patchtst_final.pth"

    PRED_QUANTILE = 0.75
    RSI_MAX = 70

    DAILY_ALLOC_PCT = 0.10
    MAX_TOTAL_EXPOSURE = 0.50

    STOP_LOSS_PCT = 0.03
    TAKE_PROFIT_PCT = 0.06

    DRY_RUN = False
    TIMEZONE = "America/New_York"
    LOG_LEVEL = logging.INFO

cfg = Config()

logging.basicConfig(level=cfg.LOG_LEVEL,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("patchtst_alpaca_bot")

def load_keys_from_file(path="keys/alpaca_keys.txt"):
    if not os.path.exists(path):
        return False
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k] = v
    return True


if not os.getenv("APCA_API_KEY_ID") or not os.getenv("APCA_API_SECRET_KEY"):
    if load_keys_from_file():
        logger.info("Loaded Alpaca keys from keys/alpaca_keys.txt")
    else:
        logger.warning("Alpaca keys not set—expecting environment variables.")

class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, n_features, patch_len, model_dim=128):
        super().__init__()
        assert seq_len % patch_len == 0
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.input_dim = patch_len * n_features
        self.proj = nn.Linear(self.input_dim, model_dim)

    def forward(self, x):
        # x: (B, seq_len, features)
        patches = x.unfold(1, self.patch_len, self.patch_len)
        B, P, PL, F = patches.shape
        patches = patches.reshape(B, P, PL * F)
        return self.proj(patches)


class PatchTST(nn.Module):
    def __init__(self, seq_len, n_features, patch_len,
                 model_dim=128, n_heads=8, n_layers=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(seq_len, n_features, patch_len, model_dim)
        n_patches = seq_len // patch_len

        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_emb
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

def to_series(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return pd.Series(x)

def normalize_df(df):
    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = to_series(df[col]).astype(float)
    return df

def SMA(s, n): return to_series(s).rolling(n).mean()
def EMA(s, n): return to_series(s).ewm(span=n, adjust=False).mean()

def RSI(s, n=14):
    s = to_series(s).astype(float)
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    down = -d.clip(upper=0).rolling(n).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def ATR(df, n=14):
    high = to_series(df["High"]); low = to_series(df["Low"]); close = to_series(df["Close"])
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def add_indicators(df):
    df = normalize_df(df)
    df["returns"] = df["Close"].pct_change()
    df["sma10"] = SMA(df["Close"], 10)
    df["sma50"] = SMA(df["Close"], 50)
    df["ema12"] = EMA(df["Close"], 12)
    df["ema26"] = EMA(df["Close"], 26)
    df["rsi14"] = RSI(df["Close"], 14)
    df["atr14"] = ATR(df, 14)
    df["vol20"] = df["returns"].rolling(20).std()
    df["close_sma10_ratio"] = df["Close"] / df["sma10"] - 1
    df["close_sma50_ratio"] = df["Close"] / df["sma50"] - 1
    return df.fillna(0)

def load_checkpoint(path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    import sklearn
    from torch.serialization import add_safe_globals
    add_safe_globals([sklearn.preprocessing.StandardScaler])

    ckpt = torch.load(path, map_location=device, weights_only=False)

    feature_cols = ckpt["feature_cols"]
    scaler = ckpt["scaler"]
    cfg_dict = ckpt["config"]

    seq_len = cfg_dict["seq_len"]
    patch_len = cfg_dict["patch_len"]
    n_features = cfg_dict["n_features"]

    model = PatchTST(seq_len, n_features, patch_len)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return dict(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        seq_len=seq_len,
        patch_len=patch_len
    )

def get_alpaca_clients():
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Alpaca API keys not set.")

    trading = TradingClient(key, secret, paper=True)
    data = StockHistoricalDataClient(key, secret)
    return trading, data

def fetch_daily_bars(data_client, symbol, lookback_days=365*3):
    end = dt.datetime.now(pytz.UTC)
    start = end - dt.timedelta(days=lookback_days)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed="iex"
    )
    bars = data_client.get_stock_bars(req).df

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol)

    bars = bars.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"
    })

    return bars[["Open","High","Low","Close","Volume"]]

def prepare_and_predict(data, model_bundle, ticker):

    df = fetch_daily_bars(data, ticker)
    df = add_indicators(df).dropna()

    seq_len = model_bundle["seq_len"]
    feature_cols = model_bundle["feature_cols"]

    if len(df) < seq_len + 1:
        raise RuntimeError(f"Not enough data for {ticker}")

    scaler = model_bundle["scaler"]
    last_window = df[feature_cols].iloc[-seq_len:].values

    scaled = scaler.transform(last_window)
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    model = model_bundle["model"]
    with torch.no_grad():
        pred = model(x).item()

    last_close = float(df["Close"].iloc[-1])
    last_ind = df.iloc[-1]

    return pred, last_close, last_ind

def compute_signals(preds: Dict[str, float], indicators):
    s = pd.Series(preds)
    thresh = s.quantile(cfg.PRED_QUANTILE)

    signals = {}
    for t, p in preds.items():
        rsi = indicators[t]["rsi14"]
        signals[t] = 1 if (p > thresh and rsi < cfg.RSI_MAX) else 0

    return signals

def size_shares(trading_client, price):
    acct = trading_client.get_account()
    equity = float(acct.equity)
    alloc = equity * cfg.DAILY_ALLOC_PCT
    return max(1, int(alloc // price))

def place_bracket_buy(trading_client, ticker, qty, price):
    order = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        stop_loss={"stop_price": f"{price*(1-cfg.STOP_LOSS_PCT):.2f}"},
        take_profit={"limit_price": f"{price*(1+cfg.TAKE_PROFIT_PCT):.2f}"}
    )
    return trading_client.submit_order(order)

def run_once(dry_run=True):
    logger.info("=== Starting daily PatchTST run ===")

    mb = load_checkpoint(cfg.MODEL_PATH)
    model = mb["model"]

    trading, data = get_alpaca_clients()

    predictions = {}
    closes = {}
    inds = {}

    for t in cfg.TICKERS:
        try:
            pred, last_close, ind = prepare_and_predict(data, mb, t)
            predictions[t] = pred
            closes[t] = last_close
            inds[t] = ind
            logger.info(f"{t}: pred={pred:.5f}, close={last_close:.2f}, rsi={ind['rsi14']:.2f}")
        except Exception as e:
            logger.error(f"Error predicting {t}: {e}")

    if not predictions:
        logger.error("No predictions generated — aborting.")
        return

    signals = compute_signals(predictions, inds)
    logger.info("Signals: %s", signals)

    max_pos = int(cfg.MAX_TOTAL_EXPOSURE / cfg.DAILY_ALLOC_PCT)
    buys = [t for t, s in signals.items() if s == 1]

    if len(buys) > max_pos:
        buys = sorted(buys, key=lambda x: predictions[x], reverse=True)[:max_pos]

    for t in cfg.TICKERS:

        if t in buys:
            price = closes[t]
            qty = size_shares(trading, price)

            if dry_run:
                logger.info(f"[DRY] BUY {t} qty={qty} @ {price:.2f}")
            else:
                try:
                    resp = place_bracket_buy(trading, t, qty, price)
                    logger.info(f"Placed BUY {t} -> id={resp.id}")
                except Exception as e:
                    logger.error(f"Order failed for {t}: {e}")

        else:
            if dry_run:
                logger.info(f"[DRY] No position for {t}")
            else:
                try:
                    trading.close_position(t)
                    logger.info(f"Closed {t}")
                except Exception:
                    pass

    logger.info("=== Daily run complete ===")
    return dict(predictions=predictions, signals=signals)

if __name__ == "__main__":
    out = run_once(dry_run=cfg.DRY_RUN)
    print(out)