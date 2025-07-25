import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import streamlit as st
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from scipy.signal import find_peaks
import plotly.graph_objects as go
# --- LSTM Forecast Integration ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# ==== NEUE FUNKTIONEN: Exakter Haupt-Channel & RSI-Trendlinie ====
def add_precise_channel(fig, series, start_idx=None, end_idx=None, color="blue", name_prefix="Main Channel"):
    sub_series = series
    if start_idx and end_idx:
        sub_series = series[start_idx:end_idx]
    upper_quantile = sub_series.quantile(0.95)
    lower_quantile = sub_series.quantile(0.05)
    upper_idx = sub_series[sub_series >= upper_quantile].index
    lower_idx = sub_series[sub_series <= lower_quantile].index
    if len(upper_idx) > 1 and len(lower_idx) > 1:
        import matplotlib.dates as mdates
        from scipy.stats import linregress
        x_upper = mdates.date2num(upper_idx.to_pydatetime())
        y_upper = sub_series[upper_idx]
        slope_up, intercept_up, _, _, _ = linregress(x_upper, y_upper)
        x_lower = mdates.date2num(lower_idx.to_pydatetime())
        y_lower = sub_series[lower_idx]
        slope_lo, intercept_lo, _, _, _ = linregress(x_lower, y_lower)
        x_plot = mdates.date2num(sub_series.index.to_pydatetime())
        upper_line = slope_up * x_plot + intercept_up
        lower_line = slope_lo * x_plot + intercept_lo
        mid_line = (upper_line + lower_line) / 2
        fig.add_trace(go.Scatter(x=sub_series.index, y=upper_line,
                                 mode='lines', line=dict(color=color, width=2),
                                 name=f"{name_prefix} Top"))
        fig.add_trace(go.Scatter(x=sub_series.index, y=lower_line,
                                 mode='lines', line=dict(color=color, width=2),
                                 name=f"{name_prefix} Bottom"))
        fig.add_trace(go.Scatter(x=sub_series.index, y=mid_line,
                                 mode='lines', line=dict(color=color, dash='dot', width=1),
                                 name=f"{name_prefix} Mid"))
        fig.add_shape(type="rect",
                      x0=sub_series.index[0], x1=sub_series.index[-1],
                      y0=min(lower_line), y1=max(upper_line),
                      fillcolor="rgba(0, 0, 255, 0.05)", line=dict(width=0),
                      layer="below")


# --- Inserted function: plot_spx_monthly_ma_chart() ---
def plot_spx_monthly_ma_chart():

    # SPX laden (Monthly)
    df = yf.download("^GSPC", start="2015-01-01", interval="1mo")
    df.dropna(inplace=True)
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], label='SPX Monthly Close', color='black')
    ax.plot(df.index, df['EMA5'], label='EMA5', linestyle='--', alpha=0.7)
    ax.plot(df.index, df['EMA14'], label='EMA14', linestyle='--', alpha=0.7)
    ax.plot(df.index, df['EMA20'], label='EMA20', linestyle='--', alpha=0.7)
    ax.plot(df.index, df['MA20'], label='MA20', linestyle=':', alpha=0.7)
    ax.plot(df.index, df['MA50'], label='MA50', linestyle=':', alpha=0.7)
    ax.set_title("SPX Monthly Close + MAs")
    ax.set_ylabel("Index Level")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    st.pyplot(fig)


st.set_page_config(layout="wide")
st.title("📊 Marktanalyse-Dashboard: Buy-/Test-Zonen & Sektorrotation")

ticker = None  # move definition down
st.sidebar.title("🔧 Einstellungen")
interval = st.sidebar.selectbox("⏱️ Datenintervall", options=["1wk", "1d", "1h"], index=0)
# Intervall-Notiz unterhalb des Intervall-Selectbox
resolution_note = {
    "1h": "⏰ Intraday (Scalping/Daytrading)",
    "1d": "🔎 Daily (Swingtrading)",
    "1wk": "📆 Weekly (Makro-Trends)"
}
st.sidebar.markdown(f"**Ausgewähltes Intervall:** {resolution_note.get(interval, '')}")

vereinfachte_trading = st.sidebar.checkbox("🎯 Vereinfachte Trading-Ansicht", value=False)

# ==== Sidebar-Checkboxen für neue Overlays ====
show_precise_channel = st.sidebar.checkbox("🎯 Exakten Haupt-Channel anzeigen", value=True, disabled=vereinfachte_trading)

# Sidebar: Anzeigeoptionen für Indikatoren und Signale
with st.sidebar.expander("🔍 Anzeigen"):
    show_indicators = st.checkbox("Indikatoren anzeigen", value=True, disabled=vereinfachte_trading)
    show_signals = st.checkbox("Buy/Test Signale anzeigen", value=True, disabled=vereinfachte_trading)
    show_fib_extensions = st.checkbox("Fibonacci Extensions anzeigen", value=True, disabled=vereinfachte_trading)

# Neu: Auswahlfeld für Trendrichtung
trend_direction = st.sidebar.radio("Trendrichtung für Fibonacci", options=["Uptrend", "Downtrend"], index=0)

# Dynamische Standardwerte für RSI/MA je nach Intervall
if interval == "1h":
    default_rsi_buy = 35
    default_rsi_test = 70
    default_ma_buy_distance = 2
elif interval == "1wk":
    default_rsi_buy = 45
    default_rsi_test = 60
    default_ma_buy_distance = 5
else:
    default_rsi_buy = 40
    default_rsi_test = 65
    default_ma_buy_distance = 3

ticker = st.sidebar.text_input("📈 Ticker", value="^GSPC")
with st.sidebar.expander("📘 Tickerliste (Beispiele)"):
    st.markdown("""
    **Indizes**
    - ^GSPC → S&P 500  
    - ^NDX → Nasdaq 100  
    - ^DJI → Dow Jones  
    - ^RUT → Russell 2000  
    - ^GDAXI → Dax 40

    **Einzelaktien**
    - AAPL → Apple  
    - MSFT → Microsoft  
    - NVDA → Nvidia  
    - TSLA → Tesla  
    - AMZN → Amazon
    - AMD → AMD
    - MO.PA → LVMH
    
    **Einzelaktien**
    - GC=F → Gold Future  

    **ETFs**
    - SPY → S&P 500 ETF  
    - QQQ → Nasdaq 100 ETF  
    - IWM → Russell 2000 ETF  
    - DIA → Dow Jones ETF  
    """)



start_date = st.sidebar.date_input("📅 Startdatum", value=pd.to_datetime("2024-01-01"))
# Set default end date to tomorrow (today + 1 day), but only as default; if the user selects another date, use that.
default_end_date = pd.to_datetime("today") + pd.Timedelta(days=1)
end_date = st.sidebar.date_input("📅 Enddatum", value=default_end_date)
## Remove sliders for RSI/MA/Volume thresholds, use fixed defaults
rsi_buy_threshold = 30
rsi_test_threshold = 50
ma_buy_distance = 3
price_bins = 50


zone_prominence = st.sidebar.slider("Prominenz für Zonenfindung", 10, 1000, 300, step=50)
with st.sidebar.expander("ℹ️ Erklärung zur Zonenprominenz"):
    st.markdown("""
    Die **Prominenz** bestimmt, wie **ausgeprägt** ein lokales Hoch oder Tief sein muss, um als Zone erkannt zu werden.

    - **Niedrige Prominenz** (z. B. 100): erkennt viele kleinere Zonen – ideal für **Intraday-Setups**
    - **Hohe Prominenz** (z. B. 600–1000): erkennt nur markante, längerfristige Zonen – geeignet für **Swing- oder Positionstrading**

    **Technischer Hintergrund:** Eine Spitze zählt nur dann als relevant, wenn sie sich um mindestens die gewählte Prominenz **von benachbarten Kurswerten abhebt** (basierend auf `scipy.signal.find_peaks`).
    """)

with st.sidebar.expander("🤖 Automatisches Multimarkt-LSTM-Training"):
    st.markdown("""
    Trainiere dein LSTM-Modell **automatisch** mit mehreren Märkten (z. B. S&P 500, Nasdaq, Dow, Russell, AAPL, MSFT, NVDA, TSLA).

    Dadurch wird das Modell robuster und erkennt Muster über verschiedene Indizes und große Aktien hinweg.
    """)

    if st.button("🔄 Modell mit mehreren Märkten trainieren"):
        st.info("📥 Lade kombinierte Daten (mehrere Indizes & Aktien)...")

        tickers = ["^GSPC", "^NDX", "^DJI", "^RUT", "AAPL", "MSFT", "NVDA", "TSLA"]
        frames = []

        for ticker_symbol in tickers:
            df = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1d", auto_adjust=True)
            if df.empty or 'Close' not in df.columns:
                continue

            df['Close_Series'] = df['Close'].squeeze()
            df['EMA5'] = df['Close_Series'].ewm(span=5, adjust=False).mean()
            df['MA20'] = df['Close_Series'].rolling(window=20).mean()
            df['MA50'] = df['Close_Series'].rolling(window=50).mean()
            df['RSI'] = RSIIndicator(close=df['Close_Series'], window=14).rsi()

            vix_df = yf.download("^VIX", start=start_date, end=end_date, interval="1d", auto_adjust=True)
            vix_df['VIX_SMA5'] = vix_df['Close'].rolling(window=5).mean()
            vix_df['VIX_RSI'] = RSIIndicator(close=vix_df['Close'].squeeze(), window=14).rsi()
            vix_df['VIX_Change'] = vix_df['Close'].pct_change()
            vix_df['Month'] = vix_df.index.month / 12.0
            vix_df = vix_df[['Close', 'VIX_SMA5', 'VIX_RSI', 'VIX_Change', 'Month']]
            vix_df.rename(columns={'Close': 'VIX_Close'}, inplace=True)

            df = df.join(vix_df, how='left')
            df['RSI_Change'] = df['RSI'].diff()
            df['Close_MA20_Pct'] = (df['Close_Series'] - df['MA20']) / df['MA20']
            df['Close_EMA5_Pct'] = (df['Close_Series'] - df['EMA5']) / df['EMA5']
            df.dropna(inplace=True)

            features_df = df[['Close_Series', 'RSI', 'MA50', 'VIX_Close', 'VIX_SMA5', 'VIX_RSI',
                              'VIX_Change', 'Month', 'RSI_Change', 'Close_MA20_Pct', 'Close_EMA5_Pct']]
            frames.append(features_df)

        if not frames:
            st.error("❌ Keine gültigen Daten gefunden.")
        else:
            combined_df = pd.concat(frames, axis=0)
            combined_df.dropna(inplace=True)

            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(combined_df)

            def create_sequences(data, seq_len=30):
                X, y = [], []
                for i in range(len(data) - seq_len):
                    X.append(data[i:i + seq_len])
                    y.append(data[i + seq_len, 0])
                return np.array(X), np.array(y)

            sequence_length = 30
            X_seq, y_seq = create_sequences(scaled_features, sequence_length)
            expected_shape = (sequence_length, scaled_features.shape[1])

            model = Sequential()
            model.add(LSTM(64, activation='relu', input_shape=expected_shape))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse', run_eagerly=True)

            checkpoint = ModelCheckpoint("lstm_model.keras", monitor='loss', save_best_only=True, verbose=0)
            import tensorflow as tf
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

            epochs = 50
            batch_size = 16
            progress_bar = st.progress(0)
            status_text = st.empty()

            for epoch in range(epochs):
                model.fit(X_seq, y_seq, epochs=1, batch_size=batch_size, verbose=0, callbacks=[checkpoint, early_stop])
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Training... Epoche {epoch + 1}/{epochs}")

            progress_bar.progress(1.0)
            status_text.text("✅ Multimarkt-Training abgeschlossen.")



# Statischer Chart
show_static = st.sidebar.checkbox("📷 Statischen Chart anzeigen", value=False)

@st.cache_data(ttl=60)  # cache expires after 10 minutes
def load_data(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)
    df.dropna(inplace=True)
    # Debug-Ausgaben für df['Close']
    print(df['Close'].head())
    print(type(df['Close']))
    if isinstance(df['Close'], pd.DataFrame):
        df['MA50'] = df['Close'].iloc[:, 0].rolling(window=50).mean()
    else:
        df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Close_Series'] = df['Close'].squeeze()
    df['RSI'] = RSIIndicator(close=df['Close_Series'], window=14).rsi()

    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['EMA69'] = df['Close'].ewm(span=69, adjust=False).mean()
    df['EMA_5W'] = df['Close'].ewm(span=5 * 5, adjust=False).mean()  # 5 Wochen EMA auf Tagesbasis
    df['EMA_5Y'] = df['Close'].ewm(span=5 * 252, adjust=False).mean()  # 5 Jahres EMA auf Tagesbasis
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()

    bb = BollingerBands(close=df['Close'].squeeze(), window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_mid'] = bb.bollinger_mavg()
    return df

if st.button("🔄 Daten neu laden"):
    st.cache_data.clear()
data = load_data(ticker, start_date, end_date, interval)
data.index = pd.to_datetime(data.index)
close_series = data['Close_Series']


def identify_zone_ranges(series, prominence=0.5):
    # Buy-Zonen: lokale Tiefs
    lows_idx, _ = find_peaks(-series, prominence=prominence)
    low_levels = sorted(set(round(series[i], -1) for i in lows_idx))  # gerundet für Clustering
    # Test-Zonen: lokale Hochs
    highs_idx, _ = find_peaks(series, prominence=prominence)
    high_levels = sorted(set(round(series[i], -1) for i in highs_idx))  # gerundet für Clustering
    return low_levels, high_levels


raw_buy_levels, raw_test_levels = identify_zone_ranges(close_series, prominence=zone_prominence)
buy_levels = raw_buy_levels
test_levels = raw_test_levels

buy_zone_df = pd.DataFrame({'Level': buy_levels})
test_zone_df = pd.DataFrame({'Level': test_levels})

# Buy-/Test-Zonen (manuell, für Signalpunkte)
buy_zone = data[(close_series < data['MA200'] * (1 + ma_buy_distance / 100)) & (data['RSI'] < rsi_buy_threshold)]
test_zone = data[(close_series > data['MA50'] * 1.05) & (data['RSI'] > rsi_test_threshold)]

# Fibonacci-Level
low = close_series.min()
high = close_series.max()
fib = {
    "0.0": high,
    "0.236": high - 0.236 * (high - low),
    "0.382": high - 0.382 * (high - low),
    "0.5": high - 0.5 * (high - low),
    "0.618": high - 0.618 * (high - low),
    "0.786": high - 0.786 * (high - low),
    "1.0": low,
}

# Trendrichtung erkennen
if close_series[-1] > close_series[0]:
    trend = "up"
else:
    trend = "down"

# Trend-Info in der Sidebar anzeigen (nach Definition von trend)
st.markdown(f"**Aktueller Trend:** {'Aufwärts (Uptrend)' if trend == 'up' else 'Abwärts (Downtrend)'}")

# Fibonacci-Extensions berechnen
if trend_direction == "Uptrend":
    fib_ext = {
        "1.236": high + 0.236 * (high - low),
        "1.382": high + 0.382 * (high - low),
        "1.618": high + 0.618 * (high - low),
        "2.0": high + 1.0 * (high - low),
        "2.618": high + 1.618 * (high - low),
    }
else:  # Downtrend
    fib_ext = {
        "1.236": low - 0.236 * (high - low),
        "1.382": low - 0.382 * (high - low),
        "1.618": low - 0.618 * (high - low),
        "2.0": low - 1.0 * (high - low),
        "2.618": low - 1.618 * (high - low),
    }

if trend_direction == "Uptrend":
    fib = {
        "0.0": low,
        "0.236": low + 0.236 * (high - low),
        "0.382": low + 0.382 * (high - low),
        "0.5": low + 0.5 * (high - low),
        "0.618": low + 0.618 * (high - low),
        "0.786": low + 0.786 * (high - low),
        "1.0": high,
    }
else:  # Downtrend
    fib = {
        "0.0": high,
        "0.236": high - 0.236 * (high - low),
        "0.382": high - 0.382 * (high - low),
        "0.5": high - 0.5 * (high - low),
        "0.618": high - 0.618 * (high - low),
        "0.786": high - 0.786 * (high - low),
        "1.0": low,
    }
# Volumenprofil
hist_vals, bin_edges = np.histogram(close_series, bins=price_bins)
max_volume = max(hist_vals)

# Plot: Matplotlib-Chart
fig, ax = plt.subplots(figsize=(14, 8))
# Y-Achsen-Skalierung optimieren: Skalenabstand auf 100 Punkte

ax.yaxis.set_major_locator(MultipleLocator(100))  # Skalenabstand auf 100 Punkte setzen
ax.plot(close_series.index, close_series.values, label='Close', linewidth=2.5, color='#00bfff')
ax.plot(data['MA50'], label='MA50', linestyle='--', color='#ffaa00')
ax.plot(data['MA100'], label='MA100', linestyle='--', color='brown')
ax.plot(data['MA200'], label='MA200', linestyle='--', color='#ff0000')

ax.plot(data['EMA5'], label='EMA5', linestyle='--', color='#cc00cc')
ax.plot(data['EMA9'], label='EMA9', linestyle='--', color='#ffff00')
ax.plot(data['EMA14'], label='EMA14', linestyle='--', color='#00cc00')
ax.plot(data['EMA69'], label='EMA69', linestyle='--', color='#9966ff')
ax.plot(data['MA20'], label='MA20', linestyle='--', color='red')

ax.plot(data['BB_upper'], label='BB Upper', linestyle='--', color='purple', alpha=0.4)
ax.plot(data['BB_lower'], label='BB Lower', linestyle='--', color='purple', alpha=0.4)
ax.plot(data['BB_mid'], label='BB Mid', linestyle='--', color='purple', alpha=0.3)

# Signalpunkte
ax.scatter(buy_zone.index, close_series.loc[buy_zone.index], label='Buy Zone (Signal)', marker='o', color='green', s=80)
ax.scatter(test_zone.index, close_series.loc[test_zone.index], label='Test Zone (Signal)', marker='x', color='red', s=80)


## Remove all sliders for RSI, MA-Nähe, Y-Achsen-Zoom, Clustering-Schwelle und Volume-Bins
# (No explicit code for these; ensure only zone_prominence and min_score slider remain)

# Replace previous slider for minimal zone score threshold with new "Konfluenz-Schwelle"
selected_min_score = st.sidebar.slider("Konfluenz-Schwelle", 1, 3, 2)

# --- Neue Confluence Zone Logik ---
def is_near_fibonacci_level(price, fibs=None, tolerance=0.01):
    """True, wenn Preis nahe an einem Fibonacci-Level liegt."""
    if fibs is None:
        return False
    for val in fibs.values():
        if abs(float(price) - float(val)) / float(val) < tolerance:
            return True
    return False

def is_near_ma(price, ma_values, tolerance=0.015):
    """True, wenn Preis nahe an einem der angegebenen MA-Werte liegt."""
    for ma in ma_values:
        if pd.notna(ma) and abs(price - ma) / ma < tolerance:
            return True
    return False

def find_confluence_zones(series, prominence=300, fibs=None, ma_series_dict=None):
    """
    Identifiziert Confluence Zones (ehemals Buy/Test-Zonen) mit 3-Punkte-Score.
    Score: +1 lokale Preisreaktion (Prominenz), +1 Nähe Fibonacci, +1 Nähe MA.
    """
    # 1. Lokale Extrema (Preisreaktion)
    lows_idx, _ = find_peaks(-series, prominence=prominence)
    highs_idx, _ = find_peaks(series, prominence=prominence)
    zone_idxs = sorted(set(list(lows_idx) + list(highs_idx)))
    zone_levels = [series[i] for i in zone_idxs]
    zones = []
    for i, lvl in zip(zone_idxs, zone_levels):
        zone_score = 0
        # 1. Lokale Preisreaktion (immer gegeben, da durch Prominenz identifiziert)
        zone_score += 1
        # 2. Nähe Fibonacci-Level
        fib_hit = is_near_fibonacci_level(lvl, fibs=fibs, tolerance=0.015)
        if fib_hit:
            zone_score += 1
        # 3. Nähe zu gleitendem Durchschnitt (MA200 oder EMA50)
        ma_hit = False
        if ma_series_dict is not None:
            for ma_name, ma_ser in ma_series_dict.items():
                # Nächstliegender Index für diesen Level
                nearest_idx = np.abs(series.values - lvl).argmin()
                ma_val = ma_ser.iloc[nearest_idx] if nearest_idx < len(ma_ser) else np.nan
                if pd.notna(ma_val) and abs(lvl - ma_val) / ma_val < 0.015:
                    ma_hit = True
                    break
        if ma_hit:
            zone_score += 1
        # Compute low/high/mid for zone labeling
        band = lvl * 0.015  # 1.5% band
        zone_low = lvl - band
        zone_high = lvl + band
        zone_mid = lvl
        zones.append({
            'level': lvl,
            'score': zone_score,
            'low': zone_low,
            'high': zone_high,
            'mid': zone_mid
        })
    return zones

# Buy-/Test-Zonen als Flächen (je 1 Rechteck pro Zone mit 1.5% Bandbreite)
valid_ma200 = data['MA200'].dropna()
if not valid_ma200.empty:
    buy_center = valid_ma200.mean()
    buy_lower = buy_center * (1 - 0.015)
    buy_upper = buy_center * (1 + 0.015)
    ax.axhspan(buy_lower, buy_upper, color='#00ff00', alpha=0.1, label='Buy-Zone (MA200±1.5%) [manuell]')

valid_ma50 = data['MA50'].dropna()
if not valid_ma50.empty:
    test_center = valid_ma50.mean()
    test_lower = test_center * 1.03
    test_upper = test_center * 1.08
    # Adjust test zone to 1.5% band around mean of test zone range for consistency
    test_mid = (test_lower + test_upper) / 2
    test_lower = test_mid * (1 - 0.015)
    test_upper = test_mid * (1 + 0.015)
    ax.axhspan(test_lower, test_upper, color='#ff6600', alpha=0.1, label='Test-Zone (MA50+1.5%) [manuell]')

# Automatisch erkannte Buy-/Test-Zonen als Rechtecke (je 1 Rechteck pro Zone mit 1.5% Bandbreite)
if buy_levels:
    buy_min = min(buy_levels)
    buy_max = max(buy_levels)
    buy_lower_auto = buy_min * (1 - 0.015)
    buy_upper_auto = buy_max * (1 + 0.015)
    ax.axhspan(buy_lower_auto, buy_upper_auto, color='#00ff00', alpha=0.1, label='Buy-Zone automatisch')

if test_levels:
    test_min = min(test_levels)
    test_max = max(test_levels)
    test_lower_auto = test_min * (1 - 0.015)
    test_upper_auto = test_max * (1 + 0.015)
    ax.axhspan(test_lower_auto, test_upper_auto, color='#ff6600', alpha=0.1, label='Test-Zone automatisch')

## --- Entferne Buy-/Test-Zonen-Flächen und zeichne nur noch Confluence Zones ---
# Bestimme Confluence Zones mit Score
ma_series_dict = {'MA200': data['MA200'], 'EMA50': data['EMA14']}  # EMA14 als EMA50-Ersatz, falls EMA50 nicht vorhanden
confluence_zones = find_confluence_zones(
    close_series, prominence=zone_prominence, fibs=fib, ma_series_dict=ma_series_dict
)
# Filtere nach Score
confluence_zones = [z for z in confluence_zones if z['score'] >= selected_min_score]
# Zeichne Confluence Zones als horizontale Linien mit neuem Label-Stil
#
# --- Neue Beschriftung der Confluence Zones weiter rechts, mit Preisbereich, automatischer Versatz ---
used_y_positions = []
min_vsep = 0.01  # minimaler vertikaler Abstand (relativ zum Preis)

for i, zone in enumerate(confluence_zones):
    color = {3: 'darkgreen', 2: 'orange', 1: 'gray'}.get(zone['score'], 'gray')
    ax.axhline(y=zone['level'], color=color, linestyle='--', linewidth=2, alpha=0.8)
    # Preisbereich (Mitte, Low, High)
    zone_bottom = zone.get('low', zone['level'])
    zone_top = zone.get('high', zone['level'])
    price_level = (zone_top + zone_bottom) / 2
    match_count = zone['score']
    total_indicators = 3
    # Calculate price_min and price_max for annotation
    price_min = min(zone_top, zone_bottom)
    price_max = max(zone_top, zone_bottom)
    # X-Position: deutlich weiter rechts, um Überlappung mit Candles zu vermeiden
    x_pos = data.index[-1] + pd.Timedelta(days=30)
    # Automatischer Versatz bei Überlappung
    y_pos = price_level
    for prev_y in used_y_positions:
        if abs(prev_y - y_pos) / max(1, y_pos) < min_vsep:
            y_pos += (zone_top - zone_bottom) * 0.3 if (i % 2 == 0) else -(zone_top - zone_bottom) * 0.3
    used_y_positions.append(y_pos)
    # Updated label: show score and price range (rounded, upper–lower)
    label = f"Confluence Zone: {match_count}/{total_indicators}\n{zone['low']:.0f}–{zone['high']:.0f}"
    ax.annotate(
        label,
        xy=(x_pos, y_pos),
        xytext=(x_pos, y_pos + (zone_top-zone_bottom)*0.1),
        ha="left",
        va='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
        fontsize=12,
        arrowprops=None
    )
    # --- Kursziel unterhalb der aktuellen Zone anzeigen ---
    # Berechnung des ATR (14 Perioden)
    atr = data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()
    atr_value = atr.iloc[-1] if isinstance(atr, pd.Series) else atr
    # Kursziel: Unterkante der Zone - ATR * 1.5
    kursziel = zone_bottom - (atr_value * 1.5)
    # Kursziel anzeigen (z. B. als Text rechts im Chart)
    ax.text(
        data.index[-1] + pd.Timedelta(days=20),  # Position rechts neben letztem Kerzenstand
        kursziel,
        f"Zielbereich: {kursziel:.0f}" if isinstance(kursziel, (int, float)) else f"Zielbereich: {float(kursziel.iloc[-1]):.0f}",
        verticalalignment='center',
        bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round,pad=0.4'),
        fontsize=12,
        color='white'
    )

custom_lines = [
    Line2D([0], [0], color='darkgreen', lw=2, linestyle='--', label='Confluence Zone (3/3)'),
    Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Confluence Zone (2/3)'),
    Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Confluence Zone (1/3)'),
]

# Fibonacci farbig in grau (#cccccc), Label oben links, kleinere Schrift
for lvl, val in fib.items():
    ax.axhline(val, linestyle='--', alpha=0.7, label=f'Fib {lvl} → {val:.0f}', color='#cccccc')
for lvl, val in fib.items():
    ax.text(data.index.min(), val, f'Fib {lvl}', color='#666666', fontsize=10, verticalalignment='bottom', horizontalalignment='left')

# Volumenprofil
for count, edge in zip(hist_vals, bin_edges[:-1]):
    ax.barh(y=edge, width=(count / max_volume) * close_series.max() * 0.1, height=(bin_edges[1] - bin_edges[0]), alpha=0.2, color='gray')

# Layout
if show_static:
    st.subheader("📊 Statischer Chart (für Export oder Snapshot)")
    # Y-Achsen-Grenzen anhand der Confluence-Zonen setzen
    if confluence_zones:
        min_zone_price = min([zone['low'] for zone in confluence_zones])
        max_zone_price = max([zone['high'] for zone in confluence_zones])
        ax.set_ylim(min_zone_price - 100, max_zone_price + 100)
    st.pyplot(fig)
    ax.set_xlim([data.index.min(), data.index.max()])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_title(f"{ticker} – Buy-/Test-Zonen mit Volumenprofil & Fibonacci", fontsize=14)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Kurs")
    ax.grid(True)
    # Legende um Confluence ergänzen
    handles, labels = ax.get_legend_handles_labels()
    handles += custom_lines
    labels += ['High Confluence', 'Medium Confluence', 'Low Confluence']
    ax.legend(handles, labels)
    fig.autofmt_xdate()
    st.pyplot(fig)



# --- Zusätzliche Makro-Charts ---

# JNK vs SPX Chart mit RSI
def plot_jnk_spx_chart():
    import matplotlib.pyplot as plt
    import yfinance as yf
    import pandas as pd
    from ta.momentum import RSIIndicator
    import streamlit as st

    # Daten abrufen
    jnk = yf.download("JNK", start="2023-06-01", end=pd.to_datetime("today"), interval="1d")
    spx = yf.download("^GSPC", start="2023-06-01", end=pd.to_datetime("today"), interval="1d")

    # RSI berechnen
    jnk['RSI'] = RSIIndicator(close=jnk['Close'].squeeze(), window=14).rsi()

    # Plot
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 2.5, 1]}, sharex=True)

    # RSI
    axs[0].plot(jnk.index, jnk['RSI'], color='red', label='RSI (14)')
    axs[0].axhline(70, color='gray', linestyle='--', linewidth=1)
    axs[0].axhline(30, color='gray', linestyle='--', linewidth=1)
    axs[0].set_ylabel('RSI')
    axs[0].legend(loc='upper left')

    # Candlestick (vereinfacht als Linienchart)
    axs[1].plot(jnk.index, jnk['Close'], color='green', label='JNK Close')
    axs[1].set_ylabel('JNK')
    axs[1].legend(loc='upper left')

    # SPX
    axs[2].plot(spx.index, spx['Close'], color='cyan', label='SPX Close')
    axs[2].set_ylabel('SPX')
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    st.pyplot(fig)

def plot_hyg_chart():
    import yfinance as yf
    import pandas as pd
    import streamlit as st
    from ta.momentum import RSIIndicator
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Zeitraum festlegen
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=2)

    # Daten laden
    hyg = yf.download("HYG", start=start, end=end)
    spx = yf.download("^GSPC", start=start, end=end)

    # RSI für HYG berechnen
    rsi = RSIIndicator(close=hyg["Close"].squeeze()).rsi()
    hyg["RSI"] = rsi

    # Plot erstellen
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Preisplot: HYG (linke Achse), SPX (rechte Achse)
    ax1.plot(hyg.index, hyg["Close"], label="HYG", color="green")
    ax1.set_ylabel("HYG", color="green")
    ax1.tick_params(axis="y", labelcolor="green")

    ax1b = ax1.twinx()
    ax1b.plot(spx.index, spx["Close"], label="SPX", color="blue", alpha=0.6)
    ax1b.set_ylabel("SPX", color="blue")
    ax1b.tick_params(axis="y", labelcolor="blue")

    ax1.set_title("HYG vs SPX (2 Jahre)")
    ax1.grid(True)

    # RSI-Plot
    ax2.plot(hyg.index, hyg["RSI"], label="RSI (HYG)", color="red")
    ax2.axhline(70, color="gray", linestyle="--", linewidth=1)
    ax2.axhline(30, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("RSI")
    ax2.set_title("HYG RSI")
    ax2.grid(True)

    fig.tight_layout()
    st.pyplot(fig)




# 🟢 Marktampel
st.subheader("🚦Marktampel – Überblick")
# Sicheres Auslesen des letzten RSI-Werts
if not data.empty and 'RSI' in data.columns and not data['RSI'].dropna().empty:
    last_rsi = round(data['RSI'].dropna().iloc[-1], 1)
else:
    last_rsi = None
# Sicheres Auslesen der letzten MA50-Werte für die Steigungsberechnung
if not data.empty and 'MA50' in data.columns and len(data['MA50'].dropna()) >= 5:
    ma_slope = data['MA50'].dropna().iloc[-1] - data['MA50'].dropna().iloc[-5]
else:
    ma_slope = 0

# 5-stufige Ampellogik mit klarer Differenzierung
if last_rsi is not None:
    if last_rsi > 65 and ma_slope > 0.5:
        ampel = "🟢 Sehr bullisch"
    elif last_rsi > 55 and ma_slope > 0:
        ampel = "🟢 Bullisch"
    elif last_rsi > 45:
        ampel = "🟡 Neutral"
    elif last_rsi > 35 or ma_slope < 0:
        ampel = "🟠 Schwach"
    else:
        ampel = "🔴 Sehr schwach"
else:
    ampel = "⚫ Kein RSI verfügbar"

# Metriken anzeigen
st.metric(label="RSI (Letzte Woche)", value=f"{last_rsi}")
st.metric(label="MA50 Trend (5 Wochen)", value=f"{ma_slope:.1f}")


# Ampelbeschreibung
st.markdown(f"**Marktampel:** {ampel}")
with st.expander("ℹ️ Erläuterung zur Marktampel"):
    st.markdown("""
    Die Marktampel bewertet die aktuelle Marktlage basierend auf dem RSI (Relative Strength Index) sowie dem Trendverlauf des MA50:

    - 🟢 **Sehr bullisch**: RSI &gt; 65 und MA50-Trend deutlich steigend
    - 🟢 **Bullisch**: RSI &gt; 55 und MA50-Trend positiv
    - 🟡 **Neutral**: RSI zwischen 45 und 55
    - 🟠 **Schwach**: RSI unter 45 oder fallender MA50-Trend
    - 🔴 **Sehr schwach**: RSI unter 35 und klar negativer MA50-Trend

    Diese Einschätzung hilft bei der groben Einordnung des Marktumfelds, ersetzt aber keine eigene Analyse.
    """)

# 📥 CSV-Export
export_df = pd.DataFrame({
    'Date': data.index,
    'Close': close_series,
    'RSI': data['RSI'],
    'MA50': data['MA50'],
    'MA200': data['MA200'],
    'Buy_Zone': close_series.index.isin(buy_zone.index),
    'Test_Zone': close_series.index.isin(test_zone.index)
})
csv = export_df.to_csv(index=False)
#st.download_button("📥 Exportiere Buy-/Test-Zonen als CSV", data=csv, file_name=f'{ticker}_zones.csv', mime='text/csv')

# Debug-Check: Sind Daten vollständig?
st.write(data[['Open', 'High', 'Low', 'Close']].dropna().tail())  # Zeigt letzte 5 Zeilen mit Kursdaten
st.write(f"Datapoints: {len(data)}")  # Zeigt Anzahl der Zeilen im DataFrame


st.subheader("📊 Interaktiver Chart")
# Prepare buy_signals and test_signals for plotting
plot_df = data.copy()
plot_df['Buy Signal'] = np.where(plot_df.index.isin(buy_zone.index), plot_df['Close_Series'], np.nan)
plot_df['Test Signal'] = np.where(plot_df.index.isin(test_zone.index), plot_df['Close_Series'], np.nan)
buy_signals = plot_df['Buy Signal'].dropna()
test_signals = plot_df['Test Signal'].dropna()

from plotly.subplots import make_subplots
fig3 = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    row_heights=[0.75, 0.25],
    subplot_titles=(f"{ticker} – Preis (Candlestick, MA50, MA200, Zonen, Fibonacci)", "RSI (14)")
)
fig3.update_layout(height=1200)

# Bedingte Anzeige der Indikatoren (alle in row=1, col=1)
if show_indicators:
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA50'], name='MA50', line=dict(dash='dot', color='orange')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA200'], name='MA200', line=dict(dash='dot', color='orange')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA5'], name='EMA5', line=dict(dash='dot', color='blueviolet')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA9'], name='EMA9', line=dict(dash='dot', color='yellow')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA14'], name='EMA14', line=dict(dash='dot', color='green')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA69'], name='EMA69', line=dict(dash='dot', color='magenta')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_5W'], name='Weekly EMA(5)', line=dict(dash='dot', color='gray')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_5Y'], name='Yearly EMA(5)', line=dict(dash='dash', color='gray')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], name='MA20', line=dict(dash='dot', color='red')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA100'], name='MA100', line=dict(dash='dot', color='brown')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_upper'], name='BB Upper', line=dict(dash='dot', color='purple'), opacity=0.6), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_lower'], name='BB Lower', line=dict(dash='dot', color='purple'), opacity=0.6), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_mid'], name='BB Mid', line=dict(dash='dot', color='violet'), opacity=0.4), row=1, col=1)

# Bedingte Anzeige der Buy/Test Signale (row=1, col=1)
if show_signals:
    if not buy_signals.empty:
        fig3.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal',
            marker=dict(symbol='circle', size=10, color='green')), row=1, col=1)
    if not test_signals.empty:
        fig3.add_trace(go.Scatter(
            x=test_signals.index, y=test_signals, mode='markers', name='Test Signal',
            marker=dict(symbol='x', size=10, color='red')), row=1, col=1)

# ==== Neue Overlays: Exakter Haupt-Channel & RSI-Trendlinie (row=1, col=1) ====
if show_precise_channel:
    add_precise_channel(fig3, close_series, color="blue", name_prefix="Precise Channel")

# Sidebar-Expander für EMA(5)-Kontext
with st.sidebar.expander("EMA(5) – Kontext"):
    st.markdown("""
    **Weekly EMA(5):** Zeigt kurzfristige Trendrichtung im Wochenkontext.  
    **Yearly EMA(5):** Extrem langfristiger Trend, Orientierung bei Makrotrends.  
    Beide Linien helfen bei der Einordnung, ob Buy-/Testzonen im Trend liegen oder konträr sind.
    """)
# Ensure OHLC columns in plot_df for Candlestick
plot_df['Open'] = data['Open']
plot_df['High'] = data['High']
plot_df['Low'] = data['Low']
plot_df['Close'] = data['Close']
# OHLC-Spur ans Ende, damit sie oben liegt

# Falls Spalten ein MultiIndex sind (z. B. durch yfinance bei mehreren Tickers)
if isinstance(plot_df.columns, pd.MultiIndex):
    plot_df.columns = plot_df.columns.get_level_values(0)  # Nur die erste Ebene behalten

plot_df_reset = plot_df.reset_index().rename(columns={plot_df.index.name or 'index': 'Date'})

# Candlestick-Plot in row=1, col=1
fig3.add_trace(go.Candlestick(
    x=plot_df_reset['Date'],
    open=plot_df_reset['Open'],
    high=plot_df_reset['High'],
    low=plot_df_reset['Low'],
    close=plot_df_reset['Close'],
    increasing_line_color='lime',
    decreasing_line_color='red',
    name='Candlestick'
), row=1, col=1)

# --- TradingView-Style Layout & Zonen/Annotationen ---
from scipy.stats import linregress
import matplotlib.dates as mdates

def add_wedge_overlay(fig, series, window=60, name_prefix="Wedge"):
    sub_series = series[-window:]
    upper_idx = sub_series[sub_series >= sub_series.quantile(0.9)].index
    lower_idx = sub_series[sub_series <= sub_series.quantile(0.1)].index

    if len(upper_idx) > 1 and len(lower_idx) > 1:
        x_upper = mdates.date2num(upper_idx.to_pydatetime())
        y_upper = sub_series[upper_idx]
        slope_up, intercept_up, _, _, _ = linregress(x_upper, y_upper)

        x_lower = mdates.date2num(lower_idx.to_pydatetime())
        y_lower = sub_series[lower_idx]
        slope_lo, intercept_lo, _, _, _ = linregress(x_lower, y_lower)

        x_plot = mdates.date2num(sub_series.index.to_pydatetime())
        upper_line = slope_up * x_plot + intercept_up
        lower_line = slope_lo * x_plot + intercept_lo

        fig.add_trace(go.Scatter(x=sub_series.index, y=upper_line,
                                 mode='lines', line=dict(color='yellow', width=2),
                                 name=f"{name_prefix} Top"))
        fig.add_trace(go.Scatter(x=sub_series.index, y=lower_line,
                                 mode='lines', line=dict(color='yellow', width=2),
                                 name=f"{name_prefix} Bottom"))
        fig.add_shape(type="rect",
                      x0=sub_series.index[0], x1=sub_series.index[-1],
                      y0=min(lower_line), y1=max(upper_line),
                      fillcolor="rgba(255, 255, 0, 0.05)", line=dict(width=0),
                      layer="below")

def add_broadening_overlay(fig, series, window=80, name_prefix="Broadening"):
    sub_series = series[-window:]
    upper_idx = sub_series[sub_series >= sub_series.quantile(0.9)].index
    lower_idx = sub_series[sub_series <= sub_series.quantile(0.1)].index

    if len(upper_idx) > 1 and len(lower_idx) > 1:
        x_upper = mdates.date2num(upper_idx.to_pydatetime())
        y_upper = sub_series[upper_idx]
        slope_up, intercept_up, _, _, _ = linregress(x_upper, y_upper)

        x_lower = mdates.date2num(lower_idx.to_pydatetime())
        y_lower = sub_series[lower_idx]
        slope_lo, intercept_lo, _, _, _ = linregress(x_lower, y_lower)

        if slope_up - slope_lo > 0.00001:
            x_plot = mdates.date2num(sub_series.index.to_pydatetime())
            upper_line = slope_up * x_plot + intercept_up
            lower_line = slope_lo * x_plot + intercept_lo

            fig.add_trace(go.Scatter(x=sub_series.index, y=upper_line,
                                     mode='lines', line=dict(color='cyan', width=2),
                                     name=f"{name_prefix} Top"))
            fig.add_trace(go.Scatter(x=sub_series.index, y=lower_line,
                                     mode='lines', line=dict(color='cyan', width=2),
                                     name=f"{name_prefix} Bottom"))
            fig.add_shape(type="rect",
                          x0=sub_series.index[0], x1=sub_series.index[-1],
                          y0=min(lower_line), y1=max(upper_line),
                          fillcolor="rgba(0, 255, 255, 0.05)", line=dict(width=0),
                          layer="below")

def add_flag_channel(fig, series, window=50, name_prefix="Flag"):
    sub_series = series[-window:]
    mid = sub_series.median()
    offset = sub_series.std() * 0.5

    x_vals = np.arange(len(sub_series))
    upper_line = np.full_like(x_vals, mid + offset, dtype=np.float64)
    lower_line = np.full_like(x_vals, mid - offset, dtype=np.float64)

    fig.add_trace(go.Scatter(x=sub_series.index, y=upper_line,
                             mode='lines', line=dict(color='magenta', dash='dash', width=2),
                             name=f"{name_prefix} Top"))
    fig.add_trace(go.Scatter(x=sub_series.index, y=lower_line,
                             mode='lines', line=dict(color='magenta', dash='dash', width=2),
                             name=f"{name_prefix} Bottom"))
    fig.add_shape(type="rect",
                  x0=sub_series.index[0], x1=sub_series.index[-1],
                  y0=lower_line[0], y1=upper_line[0],
                  fillcolor="rgba(255, 0, 255, 0.05)", line=dict(width=0),
                  layer="below")

def add_bb_breakouts(fig, df, name_prefix="BB Breakout"):
    if "BB_upper" in df.columns and "BB_lower" in df.columns:
        breakouts_above = df[df["Close_Series"] > df["BB_upper"]]
        breakouts_below = df[df["Close_Series"] < df["BB_lower"]]

        if not breakouts_above.empty:
            fig.add_trace(go.Scatter(x=breakouts_above.index, y=breakouts_above["Close_Series"],
                                     mode='markers', marker=dict(color='lime', size=8, symbol='triangle-up'),
                                     name=f"{name_prefix} Above"))
        if not breakouts_below.empty:
            fig.add_trace(go.Scatter(x=breakouts_below.index, y=breakouts_below["Close_Series"],
                                     mode='markers', marker=dict(color='red', size=8, symbol='triangle-down'),
                                     name=f"{name_prefix} Below"))
# Hintergrund & Grid anpassen (TradingView-Style) für beide Subplots
fig3.update_layout(
    plot_bgcolor='#131722',
    paper_bgcolor='#131722',
    font=dict(color='#dedede'),
    hovermode="x unified",
    title=dict(
        text=f"{ticker} – Interaktiver Chart",
        x=0.5, xanchor='center',
        font=dict(size=16, color='#ffffff')
    ),
    height=1200
)
fig3.update_xaxes(
    gridcolor='#2a2e39',
    showline=True, linewidth=1, linecolor='#666666',
    showspikes=True, spikecolor="white", spikethickness=1, spikedash='dot',
    rangeslider_visible=False,
    row=1, col=1
)
fig3.update_xaxes(
    gridcolor='#2a2e39',
    showline=True, linewidth=1, linecolor='#666666',
    showspikes=True, spikecolor="white", spikethickness=1, spikedash='dot',
    rangeslider_visible=False,
    row=2, col=1
)
fig3.update_yaxes(
    gridcolor='#2a2e39',
    showline=True, linewidth=1, linecolor='#666666',
    showspikes=True, spikecolor="white", spikethickness=1, spikedash='dot',
    title_text="Preis",
    row=1, col=1
)
fig3.update_yaxes(
    gridcolor='#2a2e39',
    showline=True, linewidth=1, linecolor='#666666',
    title_text="RSI",
    range=[0, 100],
    row=2, col=1
)

# Confluence-Zonen als Bänder (row=1, col=1)
for zone in confluence_zones:
    band_color = "rgba(0, 255, 0, 0.07)" if zone['score'] == 3 else \
                 "rgba(255, 165, 0, 0.07)" if zone['score'] == 2 else \
                 "rgba(128, 128, 128, 0.05)"

    fig3.add_shape(
        type="rect",
        x0=plot_df.index[0],
        x1=plot_df.index[-1],
        y0=zone['low'],
        y1=zone['high'],
        fillcolor=band_color,
        line=dict(width=0),
        layer='below',
        row=1, col=1
    )
    fig3.add_annotation(
        x=plot_df.index[-1],
        y=zone['level'],
        text=f"{zone['score']}/3",
        showarrow=False,
        font=dict(size=10, color='white'),
        bgcolor='rgba(0,0,0,0.6)',
        bordercolor='gray',
        borderwidth=1,
        xanchor='left',
        row=1, col=1
    )

# ==== RSI Subplot in row=2, col=1 ====
rsi_series = data['RSI'].dropna()
fig3.add_trace(
    go.Scatter(x=rsi_series.index, y=rsi_series, name='RSI (14)', line=dict(color='deepskyblue', width=2)),
    row=2, col=1
)
fig3.add_hline(y=float(70), line=dict(color='gray', dash='dash'), row=2, col=1)
fig3.add_hline(y=float(30), line=dict(color='gray', dash='dash'), row=2, col=1)

# --- Chartmuster: Channel Overlay (TradingView-Style) ---
# Diese Overlays werden in der vereinfachten Ansicht komplett deaktiviert
if not vereinfachte_trading:
    show_channels = st.sidebar.checkbox("📐 Channel Overlay anzeigen", value=False)
    show_wedge = st.sidebar.checkbox("📐 Wedge Overlay", value=False)
    show_broadening = st.sidebar.checkbox("📈 Broadening Overlay", value=False)
    show_flag = st.sidebar.checkbox("🏁 Flag/Trendkanal Overlay", value=False)
    show_bb_breakouts = st.sidebar.checkbox("💥 BB Breakouts", value=False)
else:
    show_channels = False
    show_wedge = False
    show_broadening = False
    show_flag = False
    show_bb_breakouts = False

if vereinfachte_trading:
    # Vereinfachte Trading-Ansicht: RSI-Subplot direkt in den Hauptchart integrieren
    from plotly.subplots import make_subplots
    fig3 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} – Preis (Candlestick, MA50, MA200, Zonen, Fibonacci)", "RSI (14)")
    )
    fig3.update_layout(height=900)
    # Candlestick
    fig3.add_trace(go.Candlestick(
        x=plot_df_reset['Date'],
        open=plot_df_reset['Open'],
        high=plot_df_reset['High'],
        low=plot_df_reset['Low'],
        close=plot_df_reset['Close'],
        increasing_line_color='lime',
        decreasing_line_color='red',
        name='Candlestick'
    ), row=1, col=1)
    # MA50, MA200
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA50'], name='MA50', line=dict(dash='dot', color='orange')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA200'], name='MA200', line=dict(dash='dot', color='red')), row=1, col=1)
    # Confluence Zones als Rechtecke (nur Score >= Schwelle)
    for zone in confluence_zones:
        band_color = "rgba(0, 255, 0, 0.07)" if zone['score'] == 3 else \
                     "rgba(255, 165, 0, 0.07)" if zone['score'] == 2 else \
                     "rgba(128, 128, 128, 0.05)"
        fig3.add_shape(
            type="rect",
            x0=plot_df.index[0], x1=plot_df.index[-1],
            y0=zone['low'], y1=zone['high'],
            fillcolor=band_color,
            line=dict(width=0),
            layer='below',
            row=1, col=1
        )
        fig3.add_annotation(
            x=plot_df.index[-1],
            y=zone['level'],
            text=f"{zone['score']}/3",
            showarrow=False,
            font=dict(size=10, color='white'),
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor='gray',
            borderwidth=1,
            xanchor='left',
            row=1, col=1
        )
    # Fibonacci (nur Basis-Levels)
    basic_fibs = ["0.0", "0.236", "0.382", "0.5", "0.618", "0.786", "1.0"]
    for lvl in basic_fibs:
        if lvl in fib:
            val = fib[lvl]
            fig3.add_hline(
                y=val,
                line=dict(dash='dot', color='#555555'),
                opacity=0.3,
                row=1, col=1
            )
            fig3.add_annotation(
                x=plot_df.index[-1],
                y=val,
                text=f"Fib {lvl}",
                showarrow=False,
                font=dict(size=10, color='#aaaaaa'),
                bgcolor='rgba(0,0,0,0.2)',
                bordercolor='#555555',
                borderwidth=1,
                xanchor='right',
                row=1, col=1
            )
    # RSI Subplot
    rsi_series = data['RSI'].dropna()
    fig3.add_trace(
        go.Scatter(x=rsi_series.index, y=rsi_series, name='RSI (14)', line=dict(color='deepskyblue', width=2)),
        row=2, col=1
    )
    # RSI Schwellen
    fig3.add_hline(y=float(70), line=dict(color='gray', dash='dash'), row=2, col=1)
    fig3.add_hline(y=float(30), line=dict(color='gray', dash='dash'), row=2, col=1)
    # Layout
    fig3.update_layout(
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
        font=dict(color='#dedede'),
        hovermode="x unified",
        title=dict(
            text=f"{ticker} – Vereinfachte Trading-Ansicht",
            x=0.5, xanchor='center',
            font=dict(size=16, color='#ffffff')
        ),
        height=900
    )
    fig3.update_xaxes(
        gridcolor='#2a2e39',
        showline=True, linewidth=1, linecolor='#666666',
        showspikes=True, spikecolor="white", spikethickness=1, spikedash='dot',
        rangeslider_visible=False,
        row=1, col=1
    )
    fig3.update_yaxes(
        gridcolor='#2a2e39',
        showline=True, linewidth=1, linecolor='#666666',
        showspikes=True, spikecolor="white", spikethickness=1, spikedash='dot',
        title_text="Preis",
        row=1, col=1
    )
    fig3.update_yaxes(
        gridcolor='#2a2e39',
        showline=True, linewidth=1, linecolor='#666666',
        title_text="RSI",
        range=[0, 100],
        row=2, col=1
    )
    # st.plotly_chart(fig3, use_container_width=True)
else:
    # --- Original Chartmuster-Overlay-Aufrufe und Fibonacci/Extensions ---
    if show_channels:
        # Für Beispiel: letzten 60 Punkte verwenden
        channel_window = 60
        channel_data = close_series[-channel_window:]
        x_vals = np.arange(len(channel_data))

        # Regressionslinien für obere und untere Extrempunkte
        upper_quantile = channel_data.quantile(0.9)
        lower_quantile = channel_data.quantile(0.1)

        upper_idx = channel_data[channel_data >= upper_quantile].index
        lower_idx = channel_data[channel_data <= lower_quantile].index

        # Falls nicht genug Punkte für Regression, überspringen
        if len(upper_idx) > 1 and len(lower_idx) > 1:
            # Numerische X-Achsen für Regression
            x_upper = mdates.date2num(upper_idx.to_pydatetime())
            y_upper = channel_data[upper_idx]
            from scipy.stats import linregress
            slope_up, intercept_up, _, _, _ = linregress(x_upper, y_upper)

            x_lower = mdates.date2num(lower_idx.to_pydatetime())
            y_lower = channel_data[lower_idx]
            slope_lo, intercept_lo, _, _, _ = linregress(x_lower, y_lower)

            x_plot = mdates.date2num(channel_data.index.to_pydatetime())
            upper_line = slope_up * x_plot + intercept_up
            lower_line = slope_lo * x_plot + intercept_lo

            # Channel-Bänder in Plotly
            fig3.add_trace(go.Scatter(
                x=channel_data.index, y=upper_line,
                mode='lines', line=dict(color='lime', width=2), name='Channel Top'
            ))
            fig3.add_trace(go.Scatter(
                x=channel_data.index, y=lower_line,
                mode='lines', line=dict(color='red', width=2), name='Channel Bottom'
            ))
            fig3.add_trace(go.Scatter(
                x=channel_data.index, y=(upper_line + lower_line) / 2,
                mode='lines', line=dict(color='orange', dash='dash', width=1.5), name='Channel Mid'
            ))

            # Rechteck als Hintergrund (optional)
            fig3.add_shape(
                type="rect",
                x0=channel_data.index[0], x1=channel_data.index[-1],
                y0=min(lower_line), y1=max(upper_line),
                fillcolor="rgba(0, 255, 0, 0.05)", line=dict(width=0),
                layer="below"
            )

    # Fibonacci-Extensions in hellgrau
    for lvl, val in fib_ext.items():
        fig3.add_hline(
            y=val,
            line=dict(dash='dot', color='#666666'),
            opacity=0.4
        )
        fig3.add_annotation(
            x=plot_df.index[-1],
            y=val,
            text=f"Ext {lvl}",
            showarrow=False,
            font=dict(size=10, color='#aaaaaa'),
            bgcolor='rgba(0,0,0,0.3)',
            bordercolor='#666666',
            borderwidth=1,
            xanchor='right'
        )

    # Fibonacci-Level in sehr hellem Grau
    for lvl, val in fib.items():
        fig3.add_hline(
            y=val,
            line=dict(dash='dot', color='#555555'),
            opacity=0.3
        )
        fig3.add_annotation(
            x=plot_df.index[-1],
            y=val,
            text=f"Fib {lvl}",
            showarrow=False,
            font=dict(size=10, color='#aaaaaa'),
            bgcolor='rgba(0,0,0,0.2)',
            bordercolor='#555555',
            borderwidth=1,
            xanchor='right'
        )

    # --- Chartmuster-Overlay-Aufrufe ---
    if show_wedge:
        add_wedge_overlay(fig3, close_series, window=60, name_prefix="Wedge")

    if show_broadening:
        add_broadening_overlay(fig3, close_series, window=80, name_prefix="Broadening")

    if show_flag:
        add_flag_channel(fig3, close_series, window=50, name_prefix="Flag")

    if show_bb_breakouts:
        add_bb_breakouts(fig3, data, name_prefix="BB Breakout")
# --- Verbesserte Confluence-Zonen-Darstellung mit Rechtecken & Tabelle ---

# Tabelle vorbereiten
zones_table_df = pd.DataFrame([{
    "Level (Mid)": round(zone["mid"], 2),
    "Lower Band": round(zone["low"], 2),
    "Upper Band": round(zone["high"], 2),
    "Score": f"{zone['score']}/3"
} for zone in confluence_zones])

# Tabelle anzeigen (automatisch aktualisiert mit Prominenz-Slider)
st.subheader("📄 Übersicht der Confluence Zonen")
st.dataframe(zones_table_df)

# Zonen als Rechtecke (Shapes) in Plotly-Chart
for zone in confluence_zones:
    color = "rgba(0, 255, 0, 0.2)" if zone["score"] == 3 else "rgba(255, 165, 0, 0.2)" if zone["score"] == 2 else "rgba(128, 128, 128, 0.2)"
    fig3.add_shape(
        type="rect",
        x0=plot_df.index[0],  # links über ganzen Chart
        x1=plot_df.index[-1], # rechts
        y0=zone["low"],
        y1=zone["high"],
        line=dict(color=color.replace("0.2", "1.0"), width=1),
        fillcolor=color,
        layer="below"
    )
    # Preis-Annotation
    label = f"{zone['low']:.0f}–{zone['high']:.0f}\n({zone['score']}/3)"
    fig3.add_annotation(
        x=plot_df.index[-1],
        y=zone["mid"],
        text=label,
        showarrow=False,
        font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="gray",
        borderwidth=1,
        xanchor="left"
    )
#
# --- Erweiterung: Profi-Kommentar zur RSI-Interpretation (aus Screenshot) ---
with st.expander("🧠 Profi-Insight: RSI verstehen & Kontext", expanded=False):
    st.markdown("""
    Klassische Parameter wie **RSI**, Stochastic Oscillator oder Williams %R geben wertvolle Hinweise, sind aber oft interpretationsbedürftig.
    Ein hoher RSI muss nicht zwingend zu einer Korrektur führen, er kann auch nur abkühlen, während der Markt konsolidiert oder Seitwärts läuft. 
    Zum Beispiel können andere Sektoren relative Stärke zeigen, sodass der Gesamtindex trotz hohem RSI nicht fällt.

    **Wichtig:** 
    - RSI über 70 ≠ automatisch Short-Signal.
    - RSI sollte immer mit Sektorenrotation, Marktstruktur und Sentiment kombiniert werden.
    - Laut Backtests liefert "immer Short gehen bei RSI > 70" keine profitable Performance ohne korrektes Risiko-Management.

    👉 Nutze RSI lieber als einen Hinweis zur Überhitzung, nicht als alleinigen Trigger.
    """)

# --- Verbesserter LSTM Forecast mit Unsicherheitsband und Reversion ---
st.subheader("🔮 Verbesserter LSTM Forecast mit Unsicherheitsband")
show_lstm = st.checkbox("LSTM Forecast anzeigen", value=False)

if show_lstm:
    st.info("Der Forecast wird mit Unsicherheitsband (±1σ) und Reversionslogik berechnet. Basierend auf Closing, EMA5 und MA20 sowie VIX.")

    # VIX laden
    vix_df = yf.download("^VIX", start=start_date, end=end_date)
    vix_df['VIX_SMA5'] = vix_df['Close'].rolling(window=5).mean()
    vix_df['VIX_RSI'] = RSIIndicator(close=vix_df['Close'].squeeze(), window=14).rsi()
    vix_df['VIX_Change'] = vix_df['Close'].pct_change()
    vix_df['Month'] = vix_df.index.month / 12.0

    vix_df = vix_df[['Close', 'VIX_SMA5', 'VIX_RSI', 'VIX_Change', 'Month']]
    vix_df.rename(columns={'Close': 'VIX_Close'}, inplace=True)

    features_df = data.join(vix_df, how='left')
    features_df = features_df[['Close_Series', 'RSI', 'MA50', 'MA20', 'EMA5',
                               'VIX_Close', 'VIX_SMA5', 'VIX_RSI', 'VIX_Change', 'Month']]
    features_df['RSI_Change'] = features_df['RSI'].diff()
    features_df['Close_MA20_Pct'] = (features_df['Close_Series'] - features_df['MA20']) / features_df['MA20']
    features_df['Close_EMA5_Pct'] = (features_df['Close_Series'] - features_df['EMA5']) / features_df['EMA5']
    features_df.dropna(inplace=True)

    features_df = features_df[['Close_Series', 'RSI', 'MA50', 'VIX_Close', 'VIX_SMA5', 'VIX_RSI',
                               'VIX_Change', 'Month', 'RSI_Change', 'Close_MA20_Pct', 'Close_EMA5_Pct']]

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_df)

    def create_sequences(data, seq_len=30):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len, 0])
        return np.array(X), np.array(y)

    sequence_length = 30
    X_seq, y_seq = create_sequences(scaled_features, sequence_length)

    import tensorflow as tf
    tf.keras.backend.clear_session()

    model_path = "lstm_model.keras"
    expected_shape = (sequence_length, scaled_features.shape[1])

    if os.path.exists(model_path):
        st.success("✅ Modell gefunden – lade Modell.")
        model = load_model(model_path, compile=False)
        model_shape = model.input_shape[1:]
        if model_shape != expected_shape:
            st.warning(f"⚠️ Shape mismatch! Lösche altes Modell. Expected: {expected_shape}, but was: {model_shape}.")
            os.remove(model_path)
            model = None
        else:
            model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    else:
        model = None

    if model is None:
        st.warning("⚠️ Kein Modell gefunden oder neu erstellt wegen Shape-Wechsel.")
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=expected_shape))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', run_eagerly=True)

    checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, verbose=0)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    epochs = 50
    batch_size = 16

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        model.fit(X_seq, y_seq, epochs=1, batch_size=batch_size, verbose=0, callbacks=[checkpoint, early_stop])
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Training... Epoche {epoch + 1}/{epochs}")

    progress_bar.progress(1.0)
    status_text.text("✅ Training abgeschlossen.")

    # Forecast-Loop
    last_seq = scaled_features[-sequence_length:].copy()
    forecast_scaled = []
    current_seq = last_seq

    for _ in range(5):
        pred_scaled = model.predict(current_seq.reshape(1, sequence_length, X_seq.shape[-1]), verbose=0)[0, 0]
        # Reversion Logic
        pred_scaled = current_seq[-1][0] + 0.8 * (pred_scaled - current_seq[-1][0])
        new_row = current_seq[-1].copy()
        new_row[0] = pred_scaled
        current_seq = np.vstack((current_seq[1:], new_row))
        forecast_scaled.append(pred_scaled)

    # Dummy für Inverse Transform
    dummy_zeros = np.zeros((len(forecast_scaled), scaled_features.shape[1]))
    dummy_zeros[:, 0] = forecast_scaled
    forecast_close = scaler.inverse_transform(dummy_zeros)[:, 0]

    residuals = y_seq - model.predict(X_seq, verbose=0).flatten()
    residual_std = np.std(residuals)

    band_upper = forecast_close + residual_std
    band_lower = forecast_close - residual_std

    forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast_close,
                                'Upper': band_upper, 'Lower': band_lower})

    st.subheader("🗒️ Forecast-Tabelle (LSTM)")
    st.dataframe(forecast_df.style.format({"Forecast": "{:.2f}", "Upper": "{:.2f}", "Lower": "{:.2f}"}))

    # Traces
    forecast_trace = go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines+markers',
                                name='LSTM Forecast', line=dict(color='deepskyblue', width=3))

    upper_trace = go.Scatter(x=forecast_df['Date'], y=forecast_df['Upper'], mode='lines',
                             name='Upper Band', line=dict(color='lightblue', dash='dot'), showlegend=False)

    lower_trace = go.Scatter(x=forecast_df['Date'], y=forecast_df['Lower'], mode='lines',
                             name='Lower Band', line=dict(color='lightblue', dash='dot'), fill='tonexty',
                             fillcolor='rgba(30, 144, 255, 0.2)', showlegend=False)

    connection_trace = go.Scatter(x=[data.index[-1], forecast_df['Date'].iloc[0]],
                                  y=[data['Close_Series'].iloc[-1], forecast_df['Forecast'].iloc[0]],
                                  mode='lines', line=dict(color='deepskyblue', dash='dot'), showlegend=False)

    fig3.add_trace(connection_trace)
    fig3.add_trace(upper_trace)
    fig3.add_trace(lower_trace)
    fig3.add_trace(forecast_trace)

st.plotly_chart(fig3, use_container_width=True)


# Legende als Expander statt im Chart
with st.expander("Legende"):
    st.markdown("""
**Linien & Farben**
- **MA50**: dunkelblau (Durchschnitt der letzten 50 Perioden)
- **EMA20**: violett (Exponentieller Durchschnitt, 20 Perioden)
- **Close**: schwarz/grau (Schlusskurs)
- **Bollinger Bands**: mediumpurple
- **Candlesticks**:  
    - **Dunkelgrün**: Bullish (Schlusskurs > Eröffnung)  
    - **Rot**: Bearish (Schlusskurs < Eröffnung)

**Zonen**
- **Confluence Zones**:  
    - **Dunkelgrün**: Score 3/3  
    - **Orange**: Score 2/3  
    - **Grau**: Score 1/3  
  → Je höher der Score, desto mehr Faktoren treffen an dieser Zone zusammen (Preisreaktion, Fibonacci, MA).

**Signale**
- **Grüne Punkte**: Buy-Signal (Kombination aus RSI/MA)
- **Rote Punkte**: Test-Signal (Kombination aus RSI/MA)
    """)

with st.expander("🧠 Erklärung: Confluence Zones"):
    st.markdown("""
    Die **Confluence Zones** markieren Preisbereiche, an denen mehrere wichtige Faktoren zusammentreffen. Je höher der Score (maximal 3), desto mehr Argumente sprechen für die Relevanz dieser Zone.

    **Bewertungskriterien (je 1 Punkt):**
    1. Lokale Preisreaktion (markantes Hoch oder Tief, Prominenz)
    2. Nähe zu einem Fibonacci-Level
    3. Nähe zu einem gleitenden Durchschnitt (MA200 oder EMA50)

    **Interpretation:**
    - **Score 3/3:** Sehr starke Konfluenz – mehrere wichtige Faktoren treffen zusammen.
    - **Score 2/3:** Mittlere Konfluenz – mindestens zwei Faktoren stimmen überein.
    - **Score 1/3:** Leichte Konfluenz – nur ein Faktor spricht für diese Zone.

    Nur Zonen mit Score ≥ der gewählten Konfluenz-Schwelle werden angezeigt.
    """)

# 📊 Zusätzliche Makro-Charts

def plot_bpspx_chart():
    st.subheader("SPXA200R (Prozent der S&P 500 Aktien über dem 200-Tage-MA) mit RSI")
    try:
        bpspx = yf.download("^SPXA200R", start="2023-01-01", interval="1d")
        if bpspx.empty:
            st.warning("Keine SPXA200R-Daten verfügbar.")
            return

        bpspx["RSI"] = RSIIndicator(close=bpspx["Close"]).rsi()

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.set_title("SPXA200R mit RSI", fontsize=16)
        ax1.plot(bpspx.index, bpspx["Close"], label="SPXA200R", color="tab:blue")
        ax1.set_ylabel("SPXA200R", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(bpspx.index, bpspx["RSI"], label="RSI", color="tab:red", alpha=0.5)
        ax2.set_ylabel("RSI", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Fehler beim Laden der SPXA200R-Daten: {e}")

# Expander für Makro-Charts mit neuen Checkboxen
with st.expander("📊 Zusätzliche Makro-Charts"):
    show_jnk_detail_chart = st.checkbox("JNK Detailchart anzeigen", value=True)
    show_hyg_vs_spx = st.checkbox("HYG vs SPX anzeigen", value=True)
    show_vix_vs_spx = st.checkbox("VIX vs SPX anzeigen", value=True)
    show_spx_ma = st.checkbox("SPX Monthly MAs anzeigen", value=True)
    #show_spxa200r = st.checkbox("SPXA200R anzeigen", value=True)
    show_vix_seasonality = st.checkbox("VIX Saisonalität anzeigen", value=True)
    show_SP500_seasonality = st.checkbox("S&P500 Saisonalität anzeigen", value=True)
    show_sp500_pe_ratio = st.checkbox("S&P500 PE Ratio anzeigen", value=True)

    # JNK Detailchart anzeigen
    if show_jnk_detail_chart:
        start_date = "2023-06-01"
        end_date = "2025-06-27"

        jnk = yf.download("JNK", start=start_date, end=end_date)
        spx = yf.download("^GSPC", start=start_date, end=end_date)

        if not jnk.empty and not spx.empty:
            rsi_jnk = RSIIndicator(close=jnk["Close"].squeeze(), window=14).rsi()

            fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [1, 2, 2]})
            fig.suptitle("JNK vs SPX mit RSI (Detailansicht)", fontsize=16)

            axs[0].plot(jnk.index, rsi_jnk, color='red', label='RSI (14)')
            axs[0].axhline(70, color='gray', linestyle='--', linewidth=1)
            axs[0].axhline(30, color='gray', linestyle='--', linewidth=1)
            axs[0].axhline(50, color='gray', linestyle=':', linewidth=1)
            axs[0].set_ylabel("RSI")
            axs[0].legend(loc="upper left")
            axs[0].grid(True)

            axs[1].plot(jnk.index, jnk["Close"], color='green', label='JNK Close')
            axs[1].set_ylabel("JNK")
            axs[1].legend(loc="upper left")
            axs[1].grid(True)

            axs[2].plot(spx.index, spx["Close"], color='deepskyblue', label='SPX Close')
            axs[2].set_ylabel("SPX")
            axs[2].legend(loc="upper left")
            axs[2].grid(True)

            axs[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            st.pyplot(fig)

# Ensure all charts can be shown independently
#if 'show_junk' in locals() and show_jnk_detail_chart:
 #   plot_jnk_spx_chart()
    if 'show_hyg_vs_spx' in locals() and show_hyg_vs_spx:
        plot_hyg_chart()
    # VIX vs SPX Chart mit RSI
    if show_vix_vs_spx:
        vix = yf.download("^VIX", start=start_date, end=end_date)
        spx = yf.download("^GSPC", start=start_date, end=end_date)

        vix_close = vix["Close"]
        spx_close = spx["Close"]

        rsi_vix = RSIIndicator(close=vix_close.squeeze(), window=14).rsi()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(vix_close, label='VIX', color='black')
        ax1.set_ylabel("VIX", color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.legend(loc="upper left")

        ax1_2 = ax1.twinx()
        ax1_2.plot(spx_close, label='SPX', color='blue', alpha=0.6)
        ax1_2.set_ylabel("SPX", color='blue')
        ax1_2.tick_params(axis='y', labelcolor='blue')
        ax1_2.legend(loc="upper right")
        ax1.set_title("VIX vs SPX")

        ax2.plot(rsi_vix, label="VIX RSI", color='red')
        ax2.axhline(70, linestyle='--', color='gray')
        ax2.axhline(30, linestyle='--', color='gray')
        ax2.set_ylabel("RSI")
        ax2.set_title("VIX RSI")
        ax2.legend()

        st.pyplot(fig)
    elif 'show_vix_vs_spx' in locals() and show_vix_vs_spx:
        vix = yf.download("^VIX", start=start_date, end=end_date)
        spx = yf.download("^GSPC", start=start_date, end=end_date)

        vix_close = vix["Close"]
        spx_close = spx["Close"]

        rsi_vix = RSIIndicator(close=vix_close.squeeze(), window=14).rsi()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(vix_close, label='VIX', color='black')
        ax1.set_ylabel("VIX", color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.legend(loc="upper left")

        ax1_2 = ax1.twinx()
        ax1_2.plot(spx_close, label='SPX', color='blue', alpha=0.6)
        ax1_2.set_ylabel("SPX", color='blue')
        ax1_2.tick_params(axis='y', labelcolor='blue')
        ax1_2.legend(loc="upper right")
        ax1.set_title("VIX vs SPX")

        ax2.plot(rsi_vix, label="VIX RSI", color='red')
        ax2.axhline(70, linestyle='--', color='gray')
        ax2.axhline(30, linestyle='--', color='gray')
        ax2.set_ylabel("RSI")
        ax2.set_title("VIX RSI")
        ax2.legend()

        st.pyplot(fig)
    if 'show_vix_seasonality' in locals() and show_vix_seasonality:
        st.markdown("### VIX Saisonalität (20-Jahres-Schnitt)")
        st.image("images/vix_seasonality.png", use_container_width=True)
        st.caption("Durchschnittliche saisonale Volatilität basierend auf 20 Jahren")
    if 'show_SP500_seasonality' in locals() and show_SP500_seasonality:
        st.markdown("### S&P 500 Saisonalität")
        st.image("images/S&P500seasonality.png", use_container_width=True)
        st.caption("Durchschnittliche saisonale Volatilität basierend auf 20 Jahren")
    if 'show_spx_ma' in locals() and show_spx_ma:
        plot_spx_monthly_ma_chart()
    if 'show_sp500_pe_ratio' in locals() and show_sp500_pe_ratio:
        st.markdown("### S&P 500 PE Ratio")
        st.markdown("[📈 Zur Live-Grafik auf multpl.com](https://www.multpl.com/s-p-500-pe-ratio)")
        st.caption("Externe Quelle: multpl.com – aktuelle PE Ratio immer live.")
   # if 'show_spxa200r' in locals() and show_spxa200r:
   #    plot_bpspx_chart()





# 📊 Sektorrotation
st.header("📊 Sektorrotation")
period_map = {"1 Monat": "1mo", "3 Monate": "3mo", "6 Monate": "6mo", "12 Monate": "1y"}
selected_period = st.selectbox("📆 Zeitraum für Performancevergleich", list(period_map.keys()), index=2)

sector_etfs = {
    "SPY": "S&P 500", "XLK": "Technologie", "XLF": "Finanzen", "XLI": "Industrie", "XLV": "Gesundheit",
    "XLP": "Basiskonsum", "XLE": "Energie", "XLU": "Versorger", "XLY": "Zykl. Konsum",
    "XLC": "Kommunikation", "XLB": "Rohstoffe", "XLRE": "Immobilien"
}

@st.cache_data
def load_sector_data(tickers, period):
    df = yf.download(tickers, period=period, interval="1wk")['Close']
    return df.dropna()

sector_data = load_sector_data(list(sector_etfs.keys()), period_map[selected_period])
sector_perf = ((sector_data.iloc[-1] / sector_data.iloc[0]) - 1) * 100
sector_perf = sector_perf.round(2).sort_values(ascending=False)

fig2, ax2 = plt.subplots(figsize=(12, 6))
colors = ['green' if val > sector_perf['SPY'] else 'red' for val in sector_perf]
bars = ax2.bar(sector_perf.index.map(lambda x: sector_etfs[x]), sector_perf.values, color=colors)
ax2.axhline(sector_perf['SPY'], linestyle='--', color='black', label='SPY Benchmark')
for bar, value in zip(bars, sector_perf.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{value:.1f}%', ha='center', va='bottom')
ax2.set_title(f"Sektor-Performance ({selected_period})", fontsize=14)
ax2.set_ylabel("Performance in %")
ax2.set_xticklabels(sector_perf.index.map(lambda x: sector_etfs[x]), rotation=45, ha='right')
ax2.grid(True, axis='y')
ax2.legend()
st.pyplot(fig2)



# Sidebar: Erklärung Confluence Zone
with st.sidebar.expander("ℹ️ Erklärung: Confluence Zone"):
    st.markdown("""
    Eine **Confluence Zone** entsteht, wenn mehrere technische Indikatoren (z. B. Fibonacci, gleitende Durchschnitte, Volumencluster) im selben Preisbereich zusammenfallen. 
    Diese Zonen gelten als besonders relevant für mögliche Umkehrpunkte oder Breakouts.
    """)
