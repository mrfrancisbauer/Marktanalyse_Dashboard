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

st.set_page_config(layout="wide")
st.title("📊 Marktanalyse-Dashboard: Buy-/Test-Zonen & Sektorrotation")

ticker = None  # move definition down
st.sidebar.title("🔧 Einstellungen")
interval = st.sidebar.selectbox("⏱️ Datenintervall", options=["1d", "1wk", "1h"], index=0)
# Intervall-Notiz unterhalb des Intervall-Selectbox
resolution_note = {
    "1h": "⏰ Intraday (Scalping/Daytrading)",
    "1d": "🔎 Daily (Swingtrading)",
    "1wk": "📆 Weekly (Makro-Trends)"
}
st.sidebar.markdown(f"**Ausgewähltes Intervall:** {resolution_note.get(interval, '')}")

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

    **Einzelaktien**
    - AAPL → Apple  
    - MSFT → Microsoft  
    - NVDA → Nvidia  
    - TSLA → Tesla  
    - AMZN → Amazon  

    **ETFs**
    - SPY → S&P 500 ETF  
    - QQQ → Nasdaq 100 ETF  
    - IWM → Russell 2000 ETF  
    - DIA → Dow Jones ETF  
    """)
start_date = st.sidebar.date_input("📅 Startdatum", value=pd.to_datetime("2025-01-01"))
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

@st.cache_data
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
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()

    bb = BollingerBands(close=df['Close'].squeeze(), window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_mid'] = bb.bollinger_mavg()
    return df

data = load_data(ticker, start_date, end_date, interval)
data.index = pd.to_datetime(data.index)
close_series = data['Close_Series']


# Automatische Zonenidentifikation (anhand Kursstruktur)
def identify_zone_ranges(series, prominence=0.5):

    # Buy-Zonen: lokale Tiefs
    lows_idx, _ = find_peaks(-series, prominence=prominence)
    low_levels = sorted(set(round(series[i], -1) for i in lows_idx))  # gerundet für Clustering

    # Test-Zonen: lokale Hochs
    highs_idx, _ = find_peaks(series, prominence=prominence)
    high_levels = sorted(set(round(series[i], -1) for i in highs_idx))  # gerundet für Clustering

    return low_levels, high_levels

# Zonenfindung mit einstellbarer Prominenz
buy_levels, test_levels = identify_zone_ranges(close_series, prominence=zone_prominence)

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

# Fibonacci farbig in grau (#cccccc), Label oben links, kleinere Schrift
for lvl, val in fib.items():
    ax.axhline(val, linestyle='--', alpha=0.7, label=f'Fib {lvl} → {val:.0f}', color='#cccccc')
for lvl, val in fib.items():
    ax.text(data.index.min(), val, f'Fib {lvl}', color='#666666', fontsize=8, verticalalignment='bottom', horizontalalignment='left')

# Volumenprofil
for count, edge in zip(hist_vals, bin_edges[:-1]):
    ax.barh(y=edge, width=(count / max_volume) * close_series.max() * 0.1, height=(bin_edges[1] - bin_edges[0]), alpha=0.2, color='gray')

# Layout
ax.set_xlim([data.index.min(), data.index.max()])
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.set_title(f"{ticker} – Buy-/Test-Zonen mit Volumenprofil & Fibonacci", fontsize=14)
ax.set_xlabel("Datum")
ax.set_ylabel("Kurs")
ax.grid(True)
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)

# 🟢 Marktampel
st.subheader("🟢 Marktampel – Überblick")
last_rsi = round(data['RSI'].dropna().iloc[-1], 1)
ma_slope = data['MA50'].dropna().iloc[-1] - data['MA50'].dropna().iloc[-5] if len(data['MA50'].dropna()) >= 5 else 0
ampel = "🔴 Schwach"
if last_rsi > 60 and ma_slope > 0:
    ampel = "🟢 Bullisch"
elif last_rsi > 45 and ma_slope > 0:
    ampel = "🟡 Neutral ↗"
st.metric(label="RSI (Letzte Woche)", value=f"{last_rsi}")
st.metric(label="MA50 Trend (5 Wochen)", value=f"{ma_slope:.1f}")
st.markdown(f"**Marktampel:** {ampel}")

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


# 📊 Interaktives Plotly-Chart
st.subheader("📊 Interaktiver Chart")
plot_df = data.copy()
plot_df['Buy Signal'] = np.where(plot_df.index.isin(buy_zone.index), plot_df['Close_Series'], np.nan)
plot_df['Test Signal'] = np.where(plot_df.index.isin(test_zone.index), plot_df['Close_Series'], np.nan)

fig3 = go.Figure()
fig3.update_layout(height=1200)
# Y-Achse Bereich um Medianpreis ± x %
mid_price = plot_df['Close'].median()
spread = mid_price * (y_range_pct / 100)
y_min = mid_price - spread
y_max = mid_price + spread
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA50'], name='MA50', line=dict(dash='dot')))
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA200'], name='MA200', line=dict(dash='dot', color='orange')))
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA5'], name='EMA5', line=dict(dash='dot', color='blueviolet')))
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA14'], name='EMA14', line=dict(dash='dot', color='green')))
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA69'], name='EMA69', line=dict(dash='dot', color='magenta')))
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA20'], name='MA20', line=dict(dash='dot', color='red')))
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MA100'], name='MA100', line=dict(dash='dot', color='brown')))
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_upper'], name='BB Upper', line=dict(dash='dot', color='purple'), opacity=0.6))
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_lower'], name='BB Lower', line=dict(dash='dot', color='purple'), opacity=0.6))
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_mid'], name='BB Mid', line=dict(dash='dot', color='violet'), opacity=0.4))
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Buy Signal'], mode='markers', name='Buy Signal', marker=dict(color='green', size=10)))
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Test Signal'], mode='markers', name='Test Signal', marker=dict(color='red', size=10)))
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
    fig3.add_annotation(
        x=plot_df.index[-1],  # Positioniert rechts
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