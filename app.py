import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crypto Signal Scanner", page_icon="📊", layout="wide")

BASE_URL = "https://api.binance.com/api/v3/klines"
COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]

st.title("📊 Crypto 1-Minute Signal Scanner")
st.caption("Signals only • No broker connection • No automatic trades")

st.warning(
    "Important: This app uses Binance spot candles, NOT Quotex quotes. "
    "It cannot verify or generate reliable Quotex OTC signals. OTC markets may use different prices. "
    "Signals are experimental and are not financial advice; no win rate is guaranteed."
)

def get_candles(symbol: str, limit: int = 500) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": "1m", "limit": min(max(int(limit), 50), 1000)}
    response = requests.get(BASE_URL, params=params, timeout=15)
    response.raise_for_status()
    raw = response.json()
    if not raw:
        raise ValueError("No candles returned by the data provider.")
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    # The final row may be an unfinished candle. Exclude it.
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] < now].copy()
    return df.reset_index(drop=True)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ema9"] = x["close"].ewm(span=9, adjust=False).mean()
    x["ema21"] = x["close"].ewm(span=21, adjust=False).mean()
    delta = x["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    x["rsi"] = x["rsi"].fillna(50)
    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]
    x["bull_points"] = (
        (x["ema9"] > x["ema21"]).astype(int)
        + (x["rsi"] > 52).astype(int)
        + (x["macd_hist"] > 0).astype(int)
        + (x["close"] > x["ema9"]).astype(int)
    )
    x["bear_points"] = (
        (x["ema9"] < x["ema21"]).astype(int)
        + (x["rsi"] < 48).astype(int)
        + (x["macd_hist"] < 0).astype(int)
        + (x["close"] < x["ema9"]).astype(int)
    )
    x["signal"] = "NO TRADE"
    x.loc[(x["bull_points"] >= 3) & (x["bull_points"] - x["bear_points"] >= 2), "signal"] = "UP / CALL"
    x.loc[(x["bear_points"] >= 3) & (x["bear_points"] - x["bull_points"] >= 2), "signal"] = "DOWN / PUT"
    return x

def backtest(df: pd.DataFrame):
    # Signal on a completed candle; outcome measured from that candle's close
    # to the following candle's close. This is only a simple proxy for binary expiry.
    x = add_indicators(df).copy()
    x["next_close"] = x["close"].shift(-1)
    x["outcome"] = np.where(
        x["next_close"] > x["close"], "UP",
        np.where(x["next_close"] < x["close"], "DOWN", "DRAW")
    )
    trades = x[x["signal"].isin(["UP / CALL", "DOWN / PUT"]) & x["next_close"].notna()].copy()
    trades["result"] = np.where(
        ((trades["signal"] == "UP / CALL") & (trades["outcome"] == "UP"))
        | ((trades["signal"] == "DOWN / PUT") & (trades["outcome"] == "DOWN")),
        "WIN",
        np.where(trades["outcome"] == "DRAW", "DRAW", "LOSS")
    )
    decided = trades[trades["result"].isin(["WIN", "LOSS"])]
    wins = int((decided["result"] == "WIN").sum())
    losses = int((decided["result"] == "LOSS").sum())
    draws = int((trades["result"] == "DRAW").sum())
    total = wins + losses
    accuracy = (100 * wins / total) if total else np.nan
    return x, trades, wins, losses, draws, accuracy

with st.sidebar:
    st.header("Scanner settings")
    selected_coins = st.multiselect("Crypto pairs", COINS, default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    candle_limit = st.slider("Historical candles", min_value=100, max_value=1000, value=500, step=100)
    st.caption("One candle = one minute. Data is fetched from Binance public market data.")
    st.caption("OTC is deliberately excluded because a reliable matching data feed has not been configured.")

tab_live, tab_test, tab_help = st.tabs(["Live scanner", "Backtest", "How to use"])

with tab_live:
    st.write("Click **Refresh scan** to fetch the latest completed 1-minute candles.")
    if st.button("🔄 Refresh scan", type="primary"):
        if not selected_coins:
            st.info("Select at least one pair in the sidebar.")
        else:
            rows = []
            errors = []
            progress = st.progress(0)
            for i, symbol in enumerate(selected_coins):
                try:
                    candles = get_candles(symbol, limit=150)
                    if len(candles) < 40:
                        raise ValueError("Not enough completed candles.")
                    calc = add_indicators(candles)
                    last = calc.iloc[-1]
                    rows.append({
                        "Pair": symbol,
                        "Signal": last["signal"],
                        "Last close (USDT)": round(float(last["close"]), 8),
                        "RSI": round(float(last["rsi"]), 2),
                        "EMA 9": round(float(last["ema9"]), 8),
                        "EMA 21": round(float(last["ema21"]), 8),
                        "Bull points / 4": int(last["bull_points"]),
                        "Bear points / 4": int(last["bear_points"]),
                        "Last completed candle (UTC)": last["close_time"].strftime("%Y-%m-%d %H:%M:%S"),
                    })
                except Exception as e:
                    errors.append(f"{symbol}: {e}")
                progress.progress((i + 1) / max(len(selected_coins), 1))
                time.sleep(0.15)
            if rows:
                result = pd.DataFrame(rows)
                st.dataframe(result, use_container_width=True, hide_index=True)
                st.caption("A signal is a rule-based indicator, not a prediction guarantee. Confirm that the asset and price feed match your intended platform before considering any trade.")
            for error in errors:
                st.error(error)
            st.session_state["last_scan"] = rows
    elif "last_scan" in st.session_state and st.session_state["last_scan"]:
        st.dataframe(pd.DataFrame(st.session_state["last_scan"]), use_container_width=True, hide_index=True)
        st.caption("Showing the last scan saved in this session. Click Refresh scan for new data.")

with tab_test:
    st.subheader("Historical backtest")
    test_symbol = st.selectbox("Pair to test", COINS, index=0, key="test_symbol")
    test_limit = st.slider("Candles to test", 100, 1000, 500, 100, key="test_limit")
    if st.button("Run backtest"):
        try:
            with st.spinner("Fetching candles and testing rules…"):
                hist = get_candles(test_symbol, test_limit)
                if len(hist) < 50:
                    st.error("Not enough completed candles to run a useful test.")
                else:
                    calc, trades, wins, losses, draws, accuracy = backtest(hist)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Signals", len(trades))
                    c2.metric("Wins", wins)
                    c3.metric("Losses", losses)
                    c4.metric("Win rate", f"{accuracy:.2f}%" if pd.notna(accuracy) else "N/A")
                    st.metric("Draws (excluded from win-rate denominator)", draws)
                    st.caption(
                        f"Tested {len(hist)} completed candles for {test_symbol}. "
                        "The win rate is historical and may not predict future results. "
                        "This simplified test assumes the entry reference is the signal candle close and the outcome is the next candle close."
                    )
                    if not trades.empty:
                        view = trades[["open_time", "close", "signal", "next_close", "outcome", "result"]].tail(100).copy()
                        view["open_time"] = view["open_time"].dt.strftime("%Y-%m-%d %H:%M UTC")
                        st.dataframe(view.rename(columns={
                            "open_time": "Signal candle (UTC)",
                            "close": "Reference close",
                            "signal": "Signal",
                            "next_close": "Next candle close",
                            "outcome": "Price direction",
                            "result": "Backtest result"
                        }), use_container_width=True, hide_index=True)
                    else:
                        st.info("No qualifying signals in this sample. This is valid; do not loosen rules only to force more trades.")
        except Exception as e:
            st.error(f"Backtest failed: {e}")

with tab_help:
    st.subheader("What this starter does")
    st.markdown("""
- Scans selected **Binance spot** crypto pairs using completed 1-minute candles.
- Uses EMA 9/21, RSI 14, MACD histogram, and price-vs-EMA confirmation.
- Emits **UP / CALL**, **DOWN / PUT**, or **NO TRADE**.
- Backtests the same rules against historical candles and calculates measured wins/losses.
- Does **not** log into Quotex, read Quotex prices, place trades, or send orders.
- Does **not** support OTC unless you separately establish a reliable, permitted data feed matching the OTC quote stream.
""")
    st.subheader("Important limits")
    st.markdown("""
1. Binance prices may differ from Quotex prices; OTC quotes can differ substantially.
2. A next-candle close is only a rough proxy for a binary option expiry. Real entry timing, payout, ties, spreads, and platform rules can change results.
3. The app currently uses an in-session display; it is not a guaranteed always-on alert service.
4. Do not interpret historical win rate as a guaranteed future win rate. Test on unseen periods and use demo mode first.
""")
    st.subheader("Data source")
    st.markdown("[Binance Spot API documentation](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)")

st.divider()
st.caption("Educational prototype • No auto-trading • No guaranteed accuracy • Use demo testing first")
