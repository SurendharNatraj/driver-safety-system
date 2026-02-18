# app.py
# ─────────────────────────────────────────────────────────────
# Driver Health & Drowsiness AI Monitor
# Main Streamlit Dashboard
# Team: DSCET — Surendhar N & Paul Francis
# ─────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Safety AI Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3147;
    }
    .risk-normal   { color: #00ff88; font-size: 1.8rem; font-weight: bold; }
    .risk-warning  { color: #ffaa00; font-size: 1.8rem; font-weight: bold; }
    .risk-critical { color: #ff3344; font-size: 1.8rem; font-weight: bold;
                     animation: blink 0.5s step-start infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    .stMetric { background: #1e2130; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=80)
    st.title("Driver Safety AI")
    st.caption("DSCET Final Year Project\nSurendhar N & Paul Francis")
    st.divider()

    st.subheader("⚙️ Settings")
    simulation_mode = st.toggle("Simulation Mode", value=True,
        help="Use simulated sensor data (no hardware needed)")
    alert_sound     = st.toggle("Alert Sound", value=False)
    refresh_rate    = st.slider("Refresh Rate (sec)", 1, 5, 2)

    st.divider()
    st.subheader("🎯 Thresholds")
    ear_thresh  = st.slider("EAR Threshold", 0.15, 0.35, 0.25, 0.01,
                             help="Eye Aspect Ratio — lower = more sensitive")
    hr_critical = st.slider("Critical Heart Rate", 120, 160, 140,
                             help="BPM above this → CRITICAL alert")
    spo2_warn   = st.slider("SpO2 Warning Level (%)", 90, 97, 95,
                             help="Oxygen below this → WARNING")

# ── Header ────────────────────────────────────────────────────
st.title("🚗 Driver Health & Drowsiness AI Monitor")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()

# ── Simulate sensor data ──────────────────────────────────────
def simulate_sensor_data(scenario="normal"):
    """Simulate real-time sensor readings."""
    if scenario == "normal":
        return {
            "heart_rate":    random.uniform(65, 85),
            "hrv":           random.uniform(40, 65),
            "spo2":          random.uniform(97, 99.5),
            "skin_temp":     random.uniform(33, 35.5),
            "ear":           random.uniform(0.28, 0.38),
            "blink_rate":    random.randint(12, 20),
            "yawn_count":    random.randint(0, 1),
            "activity":      1
        }
    elif scenario == "warning":
        return {
            "heart_rate":    random.uniform(105, 125),
            "hrv":           random.uniform(20, 30),
            "spo2":          random.uniform(94, 96.5),
            "skin_temp":     random.uniform(36, 37.5),
            "ear":           random.uniform(0.20, 0.26),
            "blink_rate":    random.randint(5, 12),
            "yawn_count":    random.randint(2, 4),
            "activity":      0
        }
    else:  # critical
        return {
            "heart_rate":    random.uniform(135, 155),
            "hrv":           random.uniform(10, 18),
            "spo2":          random.uniform(91, 94),
            "skin_temp":     random.uniform(37.5, 39),
            "ear":           random.uniform(0.12, 0.22),
            "blink_rate":    random.randint(1, 5),
            "yawn_count":    random.randint(4, 8),
            "activity":      0
        }


def get_risk_level(data, ear_thresh, hr_critical, spo2_warn):
    """Determine fused risk level from all sensor data."""
    hr   = data["heart_rate"]
    spo2 = data["spo2"]
    ear  = data["ear"]
    hrv  = data["hrv"]

    if hr > hr_critical or spo2 < (spo2_warn - 3) or ear < (ear_thresh - 0.05) or hrv < 15:
        return "CRITICAL"
    elif hr > (hr_critical - 20) or spo2 < spo2_warn or ear < ear_thresh or data["yawn_count"] >= 3:
        return "WARNING"
    else:
        return "NORMAL"


# ── Scenario Selector ─────────────────────────────────────────
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    if st.button("✅ Normal Scenario", use_container_width=True):
        st.session_state["scenario"] = "normal"
with col_s2:
    if st.button("⚠️ Warning Scenario", use_container_width=True):
        st.session_state["scenario"] = "warning"
with col_s3:
    if st.button("🚨 Critical Scenario", use_container_width=True):
        st.session_state["scenario"] = "critical"

scenario = st.session_state.get("scenario", "normal")

# ── Get current sensor data ───────────────────────────────────
data       = simulate_sensor_data(scenario)
risk_level = get_risk_level(data, ear_thresh, hr_critical, spo2_warn)

# ── Risk Banner ───────────────────────────────────────────────
risk_colors = {"NORMAL": "#00ff88", "WARNING": "#ffaa00", "CRITICAL": "#ff3344"}
risk_icons  = {"NORMAL": "✅", "WARNING": "⚠️", "CRITICAL": "🚨"}
risk_msgs   = {
    "NORMAL":   "Driver is alert and healthy. All readings within safe range.",
    "WARNING":  "Signs of fatigue or health anomaly detected. Recommend rest break.",
    "CRITICAL": "CRITICAL RISK DETECTED! Pull over immediately!"
}

st.markdown(f"""
<div style="background:{risk_colors[risk_level]}22; border:2px solid {risk_colors[risk_level]};
     border-radius:12px; padding:20px; text-align:center; margin-bottom:20px;">
    <h2 style="color:{risk_colors[risk_level]}; margin:0;">
        {risk_icons[risk_level]} {risk_level} RISK
    </h2>
    <p style="color:#ccc; margin-top:8px;">{risk_msgs[risk_level]}</p>
</div>
""", unsafe_allow_html=True)

# ── Metrics Row 1: Vision ─────────────────────────────────────
st.subheader("👁️ Vision-Based Monitoring")
v1, v2, v3, v4 = st.columns(4)

ear_status = "🔴 ALERT" if data["ear"] < ear_thresh else "🟢 Normal"
v1.metric("Eye Aspect Ratio (EAR)", f"{data['ear']:.3f}", ear_status)
v2.metric("Blink Rate (per min)",    f"{data['blink_rate']}", 
          "🔴 Low" if data['blink_rate'] < 8 else "🟢 Normal")
v3.metric("Yawn Count",              f"{data['yawn_count']}",
          "🔴 Frequent" if data['yawn_count'] >= 3 else "🟢 OK")
v4.metric("Drowsiness Status",
          "DROWSY 😴" if data["ear"] < ear_thresh else "ALERT 😊",
          delta=None)

st.divider()

# ── Metrics Row 2: Health ─────────────────────────────────────
st.subheader("❤️ Health Monitoring")
h1, h2, h3, h4 = st.columns(4)

h1.metric("Heart Rate (BPM)",   f"{data['heart_rate']:.0f}",
          "🔴 High" if data['heart_rate'] > hr_critical else "🟢 Normal")
h2.metric("HRV (ms)",           f"{data['hrv']:.1f}",
          "🔴 Low" if data['hrv'] < 20 else "🟢 Normal")
h3.metric("SpO2 (%)",           f"{data['spo2']:.1f}",
          "🔴 Low" if data['spo2'] < spo2_warn else "🟢 Normal")
h4.metric("Skin Temp (°C)",     f"{data['skin_temp']:.1f}",
          "🔴 High" if data['skin_temp'] > 37.5 else "🟢 Normal")

st.divider()

# ── Charts ────────────────────────────────────────────────────
st.subheader("📊 Live Trend (Simulated)")

# Generate fake history
if "history" not in st.session_state:
    st.session_state["history"] = []

st.session_state["history"].append({
    "time":       datetime.now().strftime("%H:%M:%S"),
    "heart_rate": data["heart_rate"],
    "ear":        data["ear"],
    "spo2":       data["spo2"],
})

# Keep only last 20 readings
if len(st.session_state["history"]) > 20:
    st.session_state["history"] = st.session_state["history"][-20:]

hist_df = pd.DataFrame(st.session_state["history"]).set_index("time")

c1, c2 = st.columns(2)
with c1:
    st.line_chart(hist_df[["heart_rate"]], color=["#ff6b6b"], height=200)
    st.caption("Heart Rate (BPM)")
with c2:
    st.line_chart(hist_df[["ear"]], color=["#6bcbff"], height=200)
    st.caption("Eye Aspect Ratio (EAR)")

# ── Alert Log ─────────────────────────────────────────────────
st.divider()
st.subheader("📋 Alert Log")

if "alert_log" not in st.session_state:
    st.session_state["alert_log"] = []

if risk_level != "NORMAL":
    st.session_state["alert_log"].append({
        "Time":     datetime.now().strftime("%H:%M:%S"),
        "Risk":     risk_level,
        "HR (BPM)": round(data["heart_rate"], 1),
        "EAR":      round(data["ear"], 3),
        "SpO2 (%)": round(data["spo2"], 1),
        "Source":   "FUSION MODEL"
    })

if st.session_state["alert_log"]:
    log_df = pd.DataFrame(st.session_state["alert_log"][-10:])
    st.dataframe(log_df, use_container_width=True, hide_index=True)
    if st.button("🗑️ Clear Log"):
        st.session_state["alert_log"] = []
else:
    st.info("No alerts recorded yet in this session.")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("🎓 DSCET Final Year Project 2026 — Surendhar N & Paul Francis | "
           "AI-Based Driver Health & Drowsiness Risk Prediction System")

# ── Auto-refresh ─────────────────────────────────────────────
time.sleep(refresh_rate)
st.rerun()
