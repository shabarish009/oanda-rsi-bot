import pandas as pd
import numpy as np
import requests
import time
import schedule
import smtplib
import logging
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# === CONFIGURATION ===
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
RISK_PERCENT = 0.01  # 1% risk per trade
ATR_MULTIPLIER_LOW = 2.0
ATR_MULTIPLIER_HIGH = 3.5
MAX_HOLD_DAYS = 5
SCAN_INTERVAL_MINUTES = 15

open_trades = []
closed_trades = []
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
headers = {"Authorization": f"Bearer {OANDA_API_KEY}"}

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
        logging.info(f"Email sent: {subject}")
    except Exception as e:
        logging.error(f"Email error: {e}")

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
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def get_candles(instrument, count=400, granularity="H1"):
    url = f"{OANDA_URL}/v3/instruments/{instrument}/candles"
    params = {
        "count": count,
        "granularity": granularity,
        "price": "M"
    }
    response = requests.get(url, headers=headers, params=params)
    candles = response.json().get("candles", [])
    if not candles:
        return None
    data = [{
        "time": c["time"],
        "open": float(c["mid"]["o"]),
        "high": float(c["mid"]["h"]),
        "low": float(c["mid"]["l"]),
        "close": float(c["mid"]["c"]),
    } for c in candles if c["complete"]]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df

def get_tradeable_instruments():
    url = f"{OANDA_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/instruments"
    response = requests.get(url, headers=headers)
    instruments = response.json().get("instruments", [])
    return [inst["name"] for inst in instruments if inst["type"] in ["CFD", "CURRENCY", "METAL", "BOND"]]

def get_account_balance():
    url = f"{OANDA_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/summary"
    response = requests.get(url, headers=headers)
    return float(response.json()['account']['balance'])

def calculate_chandelier_exit(df, atr_multiplier):
    highest_high = df['high'].rolling(window=22).max()
    return highest_high - atr_multiplier * df['ATR']

# === MAIN SCANNER ===
def scan_market():
    global open_trades, closed_trades
    tradables = get_tradeable_instruments()
    balance = get_account_balance()

    for symbol in tradables:
        try:
            df = get_candles(symbol)
            if df is None or len(df) < TREND_SMA:
                continue
            df['SMA200'] = df['close'].rolling(TREND_SMA).mean()
            df['RSI2'] = compute_rsi(df['close'], RSI_PERIOD)
            df['ATR'] = compute_atr(df, ATR_PERIOD)
            latest = df.iloc[-1]
            price, sma, rsi2, atr = latest['close'], latest['SMA200'], latest['RSI2'], latest['ATR']
            if pd.isna(sma) or pd.isna(atr):
                continue
            trend = "up" if price > sma else "down"
            active = next((t for t in open_trades if t['symbol'] == symbol), None)

            # === EXIT ===
            if active:
                pnl = ((price - active['entry'])/active['entry']*100) if active['type']=="long" else ((active['entry'] - price)/active['entry']*100)
                if datetime.utcnow() - active['entry_time'] > timedelta(days=MAX_HOLD_DAYS) or \
                   (active['type']=="long" and price <= active['stop']) or \
                   (active['type']=="short" and price >= active['stop']):
                    closed_trades.append({"symbol": symbol, "type": active['type'], "entry": active['entry'],
                                          "exit": price, "pnl": round(pnl, 2), "result": "exit",
                                          "entry_time": active['entry_time'].isoformat(), "exit_time": datetime.utcnow().isoformat()})
                    open_trades.remove(active)
                    send_email(f"{symbol} EXIT", f"{active['type'].upper()} {symbol} at {price:.2f} | PnL: {pnl:.2f}%")
                continue

            # === ENTRY ===
            atr_multiplier = ATR_MULTIPLIER_LOW if atr < df['ATR'].median() else ATR_MULTIPLIER_HIGH
            chandelier_stop = calculate_chandelier_exit(df, atr_multiplier).iloc[-1]
            risk = abs(price - chandelier_stop)
            if risk <= 0:
                continue
            units = int((balance * RISK_PERCENT) / risk)

            if trend == "up" and rsi2 < RSI_OVERSOLD:
                open_trades.append({"symbol": symbol, "type": "long", "entry": price,
                                    "stop": chandelier_stop, "target": price + 2 * risk,
                                    "entry_time": datetime.utcnow(), "units": units})
                send_email(f"LONG Signal - {symbol}", f"LONG {symbol} at {price:.2f} | Stop: {chandelier_stop:.2f} | Units: {units}")
            elif trend == "down" and rsi2 > RSI_OVERBOUGHT:
                chandelier_stop = df['high'].rolling(window=22).max().iloc[-1] + atr_multiplier * df['ATR'].iloc[-1]
                risk = abs(price - chandelier_stop)
                if risk <= 0:
                    continue
                units = int((balance * RISK_PERCENT) / risk)
                open_trades.append({"symbol": symbol, "type": "short", "entry": price,
                                    "stop": chandelier_stop, "target": price - 2 * risk,
                                    "entry_time": datetime.utcnow(), "units": units})
                send_email(f"SHORT Signal - {symbol}", f"SHORT {symbol} at {price:.2f} | Stop: {chandelier_stop:.2f} | Units: {units}")

        except Exception as e:
            logging.error(f"Error with {symbol}: {e}")

# === REPORT ===
def end_of_day_report():
    if not closed_trades:
        return
    summary = f"Total Trades: {len(closed_trades)}\n"
    wins = [t for t in closed_trades if t['pnl'] > 0]
    losses = [t for t in closed_trades if t['pnl'] <= 0]
    summary += f"Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {len(wins)/len(closed_trades)*100:.2f}%\n"
    for t in closed_trades:
        summary += f"{t['symbol']} - {t['type'].upper()} | Entry: {t['entry']} | Exit: {t['exit']} | PnL: {t['pnl']}%\n"
    send_email("Daily Trade Summary", summary)
    closed_trades.clear()

# === SCHEDULER ===
schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(scan_market)
schedule.every().day.at("23:59").do(end_of_day_report)

logging.info("OANDA RSI(2) Bot with Smart Chandelier Exit started...")
while True:
    schedule.run_pending()
    time.sleep(1)
