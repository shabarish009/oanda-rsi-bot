# oanda_live_trader.py - Updated with full live trade execution support

import logging
import pandas as pd
import numpy as np
import requests
import time
import schedule
import smtplib
import concurrent.futures
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ==== CONFIGURATION ====
OANDA_API_KEY = "93b6806b94a587128ad4e7be3542d775-bee30c02602ff4fe553d252b079c3562"
OANDA_ACCOUNT_ID = "101-004-27216569-001"
OANDA_URL = "https://api-fxpractice.oanda.com"
EMAIL_USER = "shammy09102000@gmail.com"
EMAIL_PASS = "swadpgblucbnahiy"
EMAIL_TO = "shammy09102000@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

RSI_PERIOD = 2
RSI_OVERBOUGHT = 95
RSI_OVERSOLD = 5
TREND_SMA = 200
ATR_PERIOD = 14
RISK_PERCENT = 0.01
ATR_MULTIPLIER_LOW = 2.0
ATR_MULTIPLIER_HIGH = 3.5
MAX_HOLD_DAYS = 5
SCAN_INTERVAL_MINUTES = 5
MAX_THREADS = 20

headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}
open_trades = []
closed_trades = []

# === EMAIL ===
def send_email(subject, body):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        logging.info(f"📧 Email sent: {subject}")
    except Exception as e:
        logging.error(f"❌ Email error: {e}")

# === STRATEGY HELPERS ===
def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_chandelier_exit(df, atr_multiplier):
    highest_high = df['high'].rolling(window=22).max()
    return highest_high - atr_multiplier * df['ATR']

def is_bullish_engulfing(df):
    return df['close'].iloc[-1] > df['open'].iloc[-1] and \
           df['open'].iloc[-2] > df['close'].iloc[-2] and \
           df['close'].iloc[-1] > df['open'].iloc[-2]

def is_bearish_engulfing(df):
    return df['close'].iloc[-1] < df['open'].iloc[-1] and \
           df['open'].iloc[-2] < df['close'].iloc[-2] and \
           df['close'].iloc[-1] < df['open'].iloc[-2]

def passes_volume_filter(df):
    if 'volume' not in df.columns:
        return True
    df['rolling_volume_median'] = df['volume'].rolling(window=20).median()
    if df['rolling_volume_median'].iloc[-1] < 5:
        return True
    return df['volume'].iloc[-1] >= 0.5 * df['rolling_volume_median'].iloc[-1]

# === API HELPERS ===
def get_candles(instrument, count=400, granularity="H1"):
    url = f"{OANDA_URL}/v3/instruments/{instrument}/candles"
    params = {"count": count, "granularity": granularity, "price": "M"}
    try:
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        candles = r.json().get("candles", [])
        data = [{
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": c.get("volume", 0)
        } for c in candles if c["complete"]]
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        logging.warning(f"Candle fetch error for {instrument}: {e}")
        return None

def get_tradeable_instruments():
    url = f"{OANDA_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/instruments"
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return [i["name"] for i in r.json()["instruments"] if i["type"] in ["CFD", "CURRENCY", "METAL", "BOND"]]
    except Exception as e:
        logging.critical(f"Instrument fetch failed: {e}")
        return []

def get_account_balance():
    url = f"{OANDA_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/summary"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return float(r.json()['account']['balance'])

def place_oanda_trade(symbol, units, stop_loss, take_profit):
    url = f"{OANDA_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/orders"
    order = {
        "order": {
            "instrument": symbol,
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": f"{stop_loss:.5f}"},
            "takeProfitOnFill": {"price": f"{take_profit:.5f}"}
        }
    }
    try:
        r = requests.post(url, headers=headers, json=order)
        r.raise_for_status()
        logging.info(f"🚀 Trade placed: {symbol}, units: {units}, SL: {stop_loss}, TP: {take_profit}")
    except Exception as e:
        logging.error(f"❌ Trade error: {symbol}: {e}")

# === MARKET SCAN ===
def scan_symbol(symbol, balance):
    df = get_candles(symbol)
    if df is None or len(df) < TREND_SMA:
        return

    df['SMA200'] = df['close'].rolling(TREND_SMA).mean()
    df['RSI2'] = compute_rsi(df['close'], RSI_PERIOD)
    df['ATR'] = compute_atr(df, ATR_PERIOD)

    latest = df.iloc[-1]
    price, sma, rsi, atr = latest['close'], latest['SMA200'], latest['RSI2'], latest['ATR']
    if pd.isna(sma) or pd.isna(atr):
        return

    atr_median = df['ATR'].median()
    atr_multiplier = ATR_MULTIPLIER_HIGH if atr >= atr_median else ATR_MULTIPLIER_LOW

    chandelier_stop = calculate_chandelier_exit(df, atr_multiplier).iloc[-1]
    risk = abs(price - chandelier_stop)
    if risk <= 0:
        return

    units = int((balance * RISK_PERCENT) / risk)
    if units == 0:
        return

    if price > sma and rsi < RSI_OVERSOLD and is_bullish_engulfing(df) and passes_volume_filter(df):
        place_oanda_trade(symbol, units, chandelier_stop, price + 2*risk)
        send_email(f"LONG - {symbol}", f"Entry: {price:.2f} | SL: {chandelier_stop:.2f} | TP: {price + 2*risk:.2f} | Units: {units}")

    elif price < sma and rsi > RSI_OVERBOUGHT and is_bearish_engulfing(df) and passes_volume_filter(df):
        stop = df['high'].rolling(22).max().iloc[-1] + atr_multiplier * atr
        risk = abs(price - stop)
        if risk <= 0:
            return
        units = -int((balance * RISK_PERCENT) / risk)
        place_oanda_trade(symbol, units, stop, price - 2*risk)
        send_email(f"SHORT - {symbol}", f"Entry: {price:.2f} | SL: {stop:.2f} | TP: {price - 2*risk:.2f} | Units: {units}")

# === MAIN ===
def scan_market():
    logging.info("🔁 Market scan started")
    balance = get_account_balance()
    tradables = get_tradeable_instruments()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        executor.map(lambda sym: scan_symbol(sym, balance), tradables)

def end_of_day_report():
    send_email("Daily Bot Status", "End of day check completed.")

def main():
    logging.info("🚀 Bot starting")
    scan_market()
    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(scan_market)
    schedule.every().day.at("23:59").do(end_of_day_report)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
