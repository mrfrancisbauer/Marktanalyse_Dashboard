import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator

# --- Inserted function: plot_spx_monthly_ma_chart() ---
def plot_spx_monthly_ma_chart():
    import yfinance as yf
    import pandas as pd
    import matplotlib.dates as mdates

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
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import matplotlib.dates as mdates
import plotly.graph_objects as go
from scipy.signal import find_peaks
import datetime

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

# Automatische Cluster-Toleranz je nach Intervall (neue Logik)
selected_interval = interval
if selected_interval == "1h":
    cluster_threshold = 0.005  # 0.5%
elif selected_interval == "1d":
    cluster_threshold = 0.01  # 1%
elif selected_interval == "1wk":
    cluster_threshold = 0.015  # 1.5%
elif selected_interval == "1mo":
    cluster_threshold = 0.02  # 2%
else:
    cluster_threshold = 0.01  # Standardwert

# Sidebar: Anzeigeoptionen für Indikatoren und Signale
with st.sidebar.expander("🔍 Anzeigen"):
    show_indicators = st.checkbox("Indikatoren anzeigen", value=True)
    show_signals = st.checkbox("Buy/Test Signale anzeigen", value=True)

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

# Zusatzinfo unter Intervallauswahl
st.sidebar.info("Hinweis: RSI- und MA-Schwellenwerte passen sich automatisch an das gewählte Intervall an.")

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

    **ETFs**
    - SPY → S&P 500 ETF  
    - QQQ → Nasdaq 100 ETF  
    - IWM → Russell 2000 ETF  
    - DIA → Dow Jones ETF  
    """)
start_date = st.sidebar.date_input("📅 Startdatum", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("📅 Enddatum", value=pd.to_datetime("today"))
rsi_buy_threshold = st.sidebar.slider(
    "RSI Buy-Zone Schwelle", 10, 50, default_rsi_buy,
    help="Buy-Signal: RSI unter Schwelle **und** Kurs unter MA200 + Toleranz (%)"
)
rsi_test_threshold = st.sidebar.slider(
    "RSI Test-Zone Schwelle", 50, 90, default_rsi_test,
    help="Test-Signal: RSI über Schwelle **und** Kurs über MA50 + 5 %"
)
ma_buy_distance = st.sidebar.slider(
    "Buy-Zone Nähe zu MA200 (%)", 1, 10, default_ma_buy_distance,
    help="Buy-Signal: Kurs liegt weniger als x % über dem MA200"
)
price_bins = st.sidebar.slider("📊 Volumenprofil-Bins", 10, 100, 50)

# Y-Achse Zoom Slider
y_range_pct = st.sidebar.slider("📐 Y-Achse Zoom (%)", 1, 50, 15, help="Definiert den sichtbaren Bereich um den Medianpreis ± x %")

# Zonen-Prominenz Slider für automatische Zonenfindung
zone_prominence = st.sidebar.slider("Prominenz für Zonenfindung", 10, 1000, 300, step=50)
with st.sidebar.expander("ℹ️ Erklärung zur Zonenprominenz"):
    st.markdown("""
    Die **Prominenz** bestimmt, wie **ausgeprägt** ein lokales Hoch oder Tief sein muss, um als Buy-/Test-Zone erkannt zu werden.

    - **Niedrige Prominenz** (z. B. 100): erkennt viele kleinere Zonen – ideal für **Intraday-Setups**
    - **Hohe Prominenz** (z. B. 600–1000): erkennt nur markante, längerfristige Zonen – geeignet für **Swing- oder Positionstrading**

    **Technischer Hintergrund:** Eine Spitze zählt nur dann als relevant, wenn sie sich um mindestens die gewählte Prominenz **von benachbarten Kurswerten abhebt** (basierend auf `scipy.signal.find_peaks`).
    """)

# Clustering-Schwelle Slider
threshold_pct = st.sidebar.slider("📎 Clustering-Schwelle (%)", 0, 10, 3, step=1)
with st.sidebar.expander("ℹ️ Erklärung zur Clustering-Schwelle"):
    st.markdown("""
    Die Clustering-Schwelle bestimmt, ob **nahe beieinanderliegende Kursniveaus** (z. B. zwei Tiefs bei 4100 und 4120 Punkten) **zu einer gemeinsamen Zone zusammengefasst** werden.

    ---
    **Empfohlene Einstellungen:**
    - **Intraday-Setups (1h):** 1–2 % – genauere Zonen
    - **Swingtrading (1d):** 2–4 % – robuste Zonen mit etwas Toleranz
    - **Makro (1w):** 4–6 % – breite Zonen mit starker Signifikanz

    Die Schwelle wirkt **nachträglich** auf automatisch erkannte Hoch- und Tiefpunkte.
    """)


# Statischer Chart
show_static = st.sidebar.checkbox("📷 Statischen Chart anzeigen", value=False)

@st.cache_data(ttl=60)  # cache expires after 10 minutes
def load_data(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)
    df.dropna(inplace=True)
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Close_Series'] = df['Close'].squeeze()
    df['RSI'] = RSIIndicator(close=df['Close_Series'], window=14).rsi()

    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
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

# --- Zonen-Clustering Funktion (Cluster-Zonen zusammenfassen) ---
def cluster_zones(levels, threshold_pct):
    """Fasst nahe beieinanderliegende Zonen gemäß threshold_pct (in %) zusammen."""
    if not levels:
        return []
    levels_sorted = sorted(levels)
    clusters = []
    current_cluster = [levels_sorted[0]]
    for lvl in levels_sorted[1:]:
        if abs(lvl - current_cluster[-1]) / current_cluster[-1] * 100 <= threshold_pct:
            current_cluster.append(lvl)
        else:
            # Cluster-Mittelwert
            clusters.append(round(np.mean(current_cluster), 2))
            current_cluster = [lvl]
    clusters.append(round(np.mean(current_cluster), 2))
    return clusters

# Zonenfindung mit einstellbarer Prominenz
raw_buy_levels, raw_test_levels = identify_zone_ranges(close_series, prominence=zone_prominence)
# Clustering der gefundenen Levels mit automatisch gewählter cluster_threshold
buy_levels = cluster_zones(raw_buy_levels, cluster_threshold * 100)
test_levels = cluster_zones(raw_test_levels, cluster_threshold * 100)

# Buy-/Test-Zonen als DataFrames zur Visualisierung
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

# Volumenprofil
hist_vals, bin_edges = np.histogram(close_series, bins=price_bins)
max_volume = max(hist_vals)

# Plot: Matplotlib-Chart
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(close_series.index, close_series.values, label='Close', linewidth=2.5, color='#00bfff')
ax.plot(data['MA50'], label='MA50', linestyle='--', color='#ffaa00')
ax.plot(data['MA100'], label='MA100', linestyle='--', color='brown')
ax.plot(data['MA200'], label='MA200', linestyle='--', color='#ff0000')

ax.plot(data['EMA5'], label='EMA5', linestyle='--', color='#cc00cc')
ax.plot(data['EMA14'], label='EMA14', linestyle='--', color='#00cc00')
ax.plot(data['EMA69'], label='EMA69', linestyle='--', color='#9966ff')
ax.plot(data['MA20'], label='MA20', linestyle='--', color='red')

ax.plot(data['BB_upper'], label='BB Upper', linestyle='--', color='purple', alpha=0.4)
ax.plot(data['BB_lower'], label='BB Lower', linestyle='--', color='purple', alpha=0.4)
ax.plot(data['BB_mid'], label='BB Mid', linestyle='--', color='purple', alpha=0.3)

# Signalpunkte
ax.scatter(buy_zone.index, close_series.loc[buy_zone.index], label='Buy Zone (Signal)', marker='o', color='green', s=80)
ax.scatter(test_zone.index, close_series.loc[test_zone.index], label='Test Zone (Signal)', marker='x', color='red', s=80)


# --- Confluent Zones (Mehrfach-Konfluenz-Zonen) ---
def find_confluent_zones(df, prominence_threshold=300, volume_bins=50):
    zones = []
    price = df['Close']
    volume = df['Volume']
    high = df['High']
    low = df['Low']

    # Rolling highs/lows (swing levels)
    rolling_highs = price.rolling(window=10).max()
    rolling_lows = price.rolling(window=10).min()

    for i in range(10, len(df)):
        score = 0
        level = price.iloc[i]

        # Criteria 1: price reacted before (support/resistance)
        if i < len(rolling_highs) and i < len(rolling_lows):
            if (abs(level - rolling_highs.iloc[i]) / level).item() < 0.005 or (abs(level - rolling_lows.iloc[i]) / level).item() < 0.005:
                score += 1

        # Criteria 2: high volume at this price level
        volume_window = volume[i-5:i+5].mean()
        if volume.iloc[i].item() > volume_window.mean() * 1.5:
            score += 1

        # Criteria 3: FVG detection (gap in 3-candle structure)
        if i >= 2:
            if (low.iloc[i - 2].item() > high.iloc[i].item()):
                score += 1
            try:
                high_val = high.iloc[i - 2]
                low_val = low.iloc[i]
                try:
                    if (
                        pd.notna(high_val).item()
                        and pd.notna(low_val).item()
                        and high_val.item() < low_val.item()
                    ):
                        score += 1
                except Exception:
                    pass
            except (IndexError, KeyError):
                continue  # überspringt ungültige Indizes

        # Criteria 4: price cluster via KDE peak (approximated)
        bin_index = int((level - df['Low'].min()) / (df['High'].max() - df['Low'].min()) * volume_bins)
        if bin_index >= 0 and bin_index < volume_bins:
            score += 1

        # Add zone if score >= 1
        if score >= 1:
            zones.append({'level': level, 'score': score})

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

# --- Einzeichnen der neuen Confluent Zones ---
confluent_zones = find_confluent_zones(data, prominence_threshold=zone_prominence, volume_bins=price_bins)
for zone in confluent_zones:
    color = 'gray'
    if zone['score'] >= 4:
        color = 'green'
    elif zone['score'] >= 2:
        color = 'orange'
    ax.axhline(y=zone['level'].item() if isinstance(zone['level'], pd.Series) else zone['level'],
               color=color, linestyle='--', linewidth=1, alpha=0.8)

    # Legende für Confluent Zones ergänzen
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='green', lw=2, linestyle='--', label='High Confluence'),
    Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Medium Confluence'),
    Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Low Confluence'),
]

# Fibonacci farbig in grau (#cccccc), Label oben links, kleinere Schrift
for lvl, val in fib.items():
    ax.axhline(val, linestyle='--', alpha=0.7, label=f'Fib {lvl} → {val:.0f}', color='#cccccc')
for lvl, val in fib.items():
    ax.text(data.index.min(), val, f'Fib {lvl}', color='#666666', fontsize=8, verticalalignment='bottom', horizontalalignment='left')

# Volumenprofil
for count, edge in zip(hist_vals, bin_edges[:-1]):
    ax.barh(y=edge, width=(count / max_volume) * close_series.max() * 0.1, height=(bin_edges[1] - bin_edges[0]), alpha=0.2, color='gray')

# Layout
if show_static:
    st.subheader("📊 Statischer Chart (für Export oder Snapshot)")
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
st.download_button("📥 Exportiere Buy-/Test-Zonen als CSV", data=csv, file_name=f'{ticker}_zones.csv', mime='text/csv')

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

fig3 = go.Figure()
fig3.update_layout(height=1200)
# Y-Achse Bereich um Medianpreis ± x %
mid_price = plot_df['Close'].median()
spread = mid_price * (y_range_pct / 100)
y_min = mid_price - spread
y_max = mid_price + spread

# Bedingte Anzeige der Indikatoren
if show_indicators:
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA50'], name='MA50', line=dict(dash='dot', color='orange')))
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA200'], name='MA200', line=dict(dash='dot', color='orange')))
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA5'], name='EMA5', line=dict(dash='dot', color='blueviolet')))
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA14'], name='EMA14', line=dict(dash='dot', color='green')))
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA69'], name='EMA69', line=dict(dash='dot', color='magenta')))
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_5W'], name='Weekly EMA(5)', line=dict(dash='dot', color='gray')))
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_5Y'], name='Yearly EMA(5)', line=dict(dash='dash', color='gray')))
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], name='MA20', line=dict(dash='dot', color='red')))
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA100'], name='MA100', line=dict(dash='dot', color='brown')))
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_upper'], name='BB Upper', line=dict(dash='dot', color='purple'), opacity=0.6))
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_lower'], name='BB Lower', line=dict(dash='dot', color='purple'), opacity=0.6))
    fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_mid'], name='BB Mid', line=dict(dash='dot', color='violet'), opacity=0.4))

# Bedingte Anzeige der Buy/Test Signale
if show_signals:
    if not buy_signals.empty:
        fig3.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal',
            marker=dict(symbol='circle', size=10, color='green')))
    if not test_signals.empty:
        fig3.add_trace(go.Scatter(
            x=test_signals.index, y=test_signals, mode='markers', name='Test Signal',
            marker=dict(symbol='x', size=10, color='red')))
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

# Candlestick-Plot (x-Achse als 'Date' aus reset_index, damit Plotly korrekt darstellt)
#plot_df_reset = plot_df.reset_index()
fig3.add_trace(go.Candlestick(
    x=plot_df_reset['Date'],
    open=plot_df_reset['Open'],
    high=plot_df_reset['High'],
    low=plot_df_reset['Low'],
    close=plot_df_reset['Close'],
    increasing_line_color='lime',
    decreasing_line_color='red',
    name='Candlestick'
))
# Buy-Zonen als Rechtecke (±1.5% Bandbreite)
if buy_levels:
    buy_min = min(buy_levels)
    buy_max = max(buy_levels)
    for lvl in buy_levels:
        fig3.add_shape(type='rect',
                       xref='x', yref='y',
                       x0=plot_df.index.min(), x1=plot_df.index.max(),
                       y0=lvl * (1 - 0.015), y1=lvl * (1 + 0.015),
                       fillcolor='rgba(0, 128, 0, 0.15)',
                       line=dict(color='green', width=1),
                       layer='below')

# Buy-Zonen Textbeschriftung
for lvl in buy_levels:
    lvl_low = lvl * (1 - 0.015)
    lvl_high = lvl * (1 + 0.015)
    # Vor der Verwendung von plot_df.index[-1] prüfen, ob plot_df leer ist
    if plot_df.empty:
        st.warning("Keine Daten im ausgewählten Zeitintervall verfügbar. Bitte Intervall oder Zeitraum ändern.")
        st.stop()
    fig3.add_annotation(
        x=plot_df.index[-1],
        y=(lvl_low + lvl_high) / 2,
        text=f"Buy-Zone: {lvl_low:.0f} – {lvl_high:.0f}",
        showarrow=False,
        font=dict(size=12, color='green'),
        bgcolor='rgba(0, 128, 0, 0.2)',
        bordercolor='green',
        borderwidth=1,
        yshift=10
    )


# Test-Zonen als Rechtecke (±1.5% Bandbreite)
if test_levels:
    test_min = min(test_levels)
    test_max = max(test_levels)
    for lvl in test_levels:
        fig3.add_shape(type='rect',
                       xref='x', yref='y',
                       x0=plot_df.index.min(), x1=plot_df.index.max(),
                       y0=lvl * (1 - 0.015), y1=lvl * (1 + 0.015),
                       fillcolor='rgba(255, 102, 0, 0.15)',
                       line=dict(color='orange', width=1),
                       layer='below')

# Test-Zonen Textbeschriftung
for lvl in test_levels:
    lvl_low = lvl * (1 - 0.015)
    lvl_high = lvl * (1 + 0.015)
    if plot_df.empty:
        st.warning("Keine Daten im ausgewählten Zeitintervall verfügbar. Bitte Intervall oder Zeitraum ändern.")
        st.stop()
    fig3.add_annotation(
        x=plot_df.index[-1],
        y=(lvl_low + lvl_high) / 2,
        text=f"Test-Zone: {lvl_low:.0f} – {lvl_high:.0f}",
        showarrow=False,
        font=dict(size=12, color='orange'),
        bgcolor='rgba(255, 140, 0, 0.2)',
        bordercolor='orange',
        borderwidth=1,
        yshift=-10
    )

# Fibonacci-Level als horizontale Linien mit Annotation links oben, grau
for lvl, val in fib.items():
    fig3.add_hline(y=val, line=dict(dash='dot', color='#cccccc'), opacity=0.5)
    if plot_df.empty:
        st.warning("Keine Daten im ausgewählten Zeitintervall verfügbar. Bitte Intervall oder Zeitraum ändern.")
        st.stop()
    fig3.add_annotation(
        x=plot_df.index[-1],
        y=val,
        text=f"Fib {lvl}: {val:.0f}",
        showarrow=False,
        font=dict(size=11, color='#cccccc'),
        bgcolor='rgba(204, 204, 204, 0.2)',
        bordercolor='#999999',
        borderwidth=1,
        xanchor='right',
        yshift=15
    )

fig3.update_layout(
    title=dict(text=f"{ticker} – Interaktiver Chart", x=0.5, xanchor='center', font=dict(size=16, family="Arial", color='#ffffff', weight='bold')),
    xaxis_title=dict(text="Datum", font=dict(color='#ffffff', size=12, family="Arial", weight='bold')),
    yaxis_title=dict(text="Preis", font=dict(color='#ffffff', size=12, family="Arial", weight='bold')),
    plot_bgcolor='#1e1e1e',
    paper_bgcolor='#1e1e1e',
    font=dict(color='#ffffff'),
    xaxis=dict(gridcolor='#444444', rangeslider_visible=False),
    yaxis=dict(gridcolor='#444444', autorange=True)
)
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
- **Buy-Zonen**: grünliche Fläche ('rgba(50,200,100,0.2)') – Bereich mit erhöhtem Kaufinteresse
- **Test-Zonen**: orange-braune Fläche ('rgba(200,100,50,0.2)') – Bereich mit Widerstand/Test

**Signale**
- **Grüne Punkte**: Buy-Signal (Kombination aus RSI/MA)
- **Rote Punkte**: Test-Signal (Kombination aus RSI/MA)
    """)

with st.expander("🧠 Erklärung: Buy- und Test-Zonen"):
    st.markdown("""
    Die **Buy- und Test-Zonen** dienen der Identifikation von markanten Preisbereichen, an denen der Markt typischerweise reagiert. Diese Zonen können sowohl für Einstiege als auch für Risikomanagement genutzt werden.

    ---
    ### ✅ **Buy-Zonen**
    - **Definition:** Bereich mit erhöhtem Kaufinteresse. Typischerweise frühere Tiefs, an denen es zu Umkehrformationen kam.
    - **Bedingungen:** 
      - RSI unter eingestellter Schwelle (z. B. unter 40)
      - Kurs liegt nahe unter dem gleitenden Durchschnitt MA200
    - **Signal:** Grüner Punkt im Chart
    - **Beispiel:** 
        - RSI = 35, Kurs bei 4.200 Punkte (MA200 = 4.250) → Buy-Signal wird aktiviert

    ---
    ### 🧪 **Test-Zonen**
    - **Definition:** Preisbereiche, die als Widerstand fungieren oder „abgeklopft“ werden, bevor der Markt entscheidet.
    - **Bedingungen:** 
      - RSI über eingestellter Schwelle (z. B. über 65)
      - Kurs über MA50 + 5 %
    - **Signal:** Roter Punkt im Chart
    - **Beispiel:** 
        - RSI = 72, Kurs bei 4.600 Punkte (MA50 = 4.300) → Test-Zone aktiviert

    ---
    ### 🧠 **Hintergrund zur automatischen Erkennung**
    Zusätzlich zu den signalbasierten Zonen identifiziert der Algorithmus **automatisch relevante Kurscluster**, z. B. lokale Hochs oder Tiefs, die mehrfach angelaufen wurden. Diese Zonen basieren auf der sog. **Prominenz** des Kursverlaufs (analog zu `find_peaks`).

    Dadurch entstehen:
    - **Buy-Zonen (grüne Flächen):** Mehrfache Unterstützungen
    - **Test-Zonen (orange Flächen):** Widerstandszonen oder Pivot-Level

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
    show_junk = st.checkbox("JUNK vs SPX anzeigen")
    show_hyg = st.checkbox("HYG vs SPX anzeigen")
    show_spx_ma = st.checkbox("SPX Monthly MAs anzeigen")
    show_spxa200r = st.checkbox("SPXA200R anzeigen")

# Ensure all charts can be shown independently
if 'show_junk' in locals() and show_junk:
    plot_jnk_spx_chart()
if 'show_hyg' in locals() and show_hyg:
    plot_hyg_chart()
if 'show_spx_ma' in locals() and show_spx_ma:
    plot_spx_monthly_ma_chart()
if 'show_spxa200r' in locals() and show_spxa200r:
    plot_bpspx_chart()


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


