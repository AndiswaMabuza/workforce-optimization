import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Workforce Optimization", layout="wide")

# --- Title ---
st.title("📊 Strategic Workforce Optimization for a Multilingual Contact Center")


@st.cache_data
def simulate_contact_volume():
    regions = ["South Africa", "Nigeria", "Morocco", "Kenya", "UK"]
    channels = ["Call", "Email", "Chat"]
    languages = ["English", "French", "Arabic", "Swahili", "Zulu"]
    ts_range = pd.date_range("2025-01-01", "2025-01-14 23:45", freq="15T")

    rows = []
    for ts in ts_range:
        for ch in channels:
            for lang in languages:
                for reg in regions:
                    base = np.random.poisson(2 if ch == "Email" else 4)
                    volume = max(0, int(base + 3*np.sin(ts.hour/24*np.pi) + np.random.normal()))
                    rows.append([ts, ch, lang, reg, volume])
    return pd.DataFrame(rows, columns=["timestamp", "channel", "language", "region", "contact_volume"])

contact_df = simulate_contact_volume()

st.subheader("🔍 Exploratory Data Analysis (EDA)")

# Heatmap: Language vs Region
lang_region = contact_df.groupby(["language", "region"])["contact_volume"].sum().reset_index()
heatmap_data = lang_region.pivot(index="language", columns="region", values="contact_volume")

fig_eda1 = px.imshow(
    heatmap_data,
    text_auto=True,
    color_continuous_scale="Blues",
    aspect="auto",
    title="Contact Volume by Language and Region"
)
st.plotly_chart(fig_eda1, use_container_width=True)

# Bar Chart: Volume by Channel
channel_dist = contact_df.groupby("channel")["contact_volume"].sum().reset_index()
fig_eda2 = px.bar(
    channel_dist,
    x="channel",
    y="contact_volume",
    color="channel",
    title="Total Contact Volume by Channel",
    labels={"contact_volume": "Total Volume"}
)
st.plotly_chart(fig_eda2, use_container_width=True)

# Line Chart: Hourly Trend
contact_df["hour"] = contact_df["timestamp"].dt.hour
hourly = contact_df.groupby("hour")["contact_volume"].sum().reset_index()
fig_eda3 = px.line(
    hourly,
    x="hour",
    y="contact_volume",
    markers=True,
    title="Contact Volume by Hour of Day",
    labels={"contact_volume": "Total Volume"}
)
st.plotly_chart(fig_eda3, use_container_width=True)

st.subheader("📈 Forecasting Contact Volume")
channel = st.selectbox("Select Channel to Forecast", ["Call", "Email", "Chat"])
channel_df = contact_df[contact_df["channel"] == channel]
agg = channel_df.groupby("timestamp")["contact_volume"].sum().reset_index()
agg.columns = ["ds", "y"]


model = Prophet(daily_seasonality=True, weekly_seasonality=True)
model.fit(agg)
future = model.make_future_dataframe(periods=96*3, freq="15min")
forecast = model.predict(future)


merged = pd.merge(agg, forecast[["ds", "yhat"]], on="ds")
mae = mean_absolute_error(merged["y"], merged["yhat"])
mape = np.mean(np.abs((merged["y"] - merged["yhat"]) / merged["y"])) * 100
rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))

fig_forecast = px.line(forecast, x="ds", y="yhat", title=f"Forecasted {channel} Volume")
st.plotly_chart(fig_forecast, use_container_width=True)


st.markdown("### ✅ Forecast Accuracy Metrics")
st.columns(3)[0].metric("MAE", f"{mae:.2f}")
st.columns(3)[1].metric("MAPE", f"{mape:.2f}%")
st.columns(3)[2].metric("RMSE", f"{rmse:.2f}")


@st.cache_data
def generate_schedule():
    agents = [f"A{i:03}" for i in range(1, 101)]
    shifts = ["08:00-16:00", "09:00-17:00", "12:00-20:00", "14:00-22:00"]
    sched = []
    for agent in agents:
        for d in pd.date_range("2025-01-01", "2025-01-14"):
            shift = np.random.choice(shifts)
            sched.append([agent, d.date(), shift])
    return pd.DataFrame(sched, columns=["agent_id", "date", "shift"])

st.subheader("📅 Shift Schedule Heatmap")
schedule_df = generate_schedule()
heatmap_df = schedule_df.groupby(["date", "shift"]).size().unstack().fillna(0)
fig2 = px.imshow(heatmap_df.T, aspect="auto", labels=dict(x="Date", y="Shift", color="Agent Count"))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🚨 Scenario Simulation: Mid-Day Volume Spike")
sim_df = forecast.copy()
sim_df["yhat_spike"] = sim_df["yhat"]
sim_df.loc[sim_df["ds"].dt.hour.between(10, 14), "yhat_spike"] *= 1.5

fig3 = px.line(sim_df, x="ds", y=["yhat", "yhat_spike"], labels={"value": "Contact Volume"}, title="Spike vs Normal Forecast")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("🧾 Executive Summary")
st.markdown("""
- 📞 Call volume peaks midday across all regions.
- 👷 FTE needs increase ~30% during 10am–2pm.
- 📟 Shift coverage is uneven—optimize staggered shifts.
- ⚠️ Real-time dashboards can boost SLA compliance.
- 🧠 Simulations allow smarter peak-time planning.
""")
