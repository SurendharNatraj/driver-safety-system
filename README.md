# 🚗 AI-Based Driver Health & Drowsiness Risk Prediction System

**DSCET Final Year Project 2026**
Team: Surendhar N & Paul Francis

---

## 📌 Project Overview

A real-time AI system that monitors driver alertness and health using:
- **Vision-based analysis** — Eye blink rate, Eye Aspect Ratio (EAR), yawn detection
- **Heart rate monitoring** — Wearable sensor / smartwatch data
- **Fusion AI Model** — Random Forest + Gradient Boosting for risk classification
- **Risk Levels** — NORMAL / WARNING / CRITICAL with instant alerts

---

## 📁 Folder Structure

```
driver-safety-system/
├── drowsiness/
│   ├── detect_drowsiness.py       # Eye & yawn detection (OpenCV + dlib)
│   └── shape_predictor_68.dat     # Download separately (see below)
├── health/
│   ├── heart_rate_model.py        # Fusion ML model (RF + GB)
│   └── dataset.csv                # Auto-generated on first run
├── alerts/
│   └── alert.py                   # Alert system
├── models/
│   └── health_model.pkl           # Saved after training
├── app.py                         # Main Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dlib face landmark model
Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
Extract and place `shape_predictor_68_face_landmarks.dat` inside the `drowsiness/` folder.

### 3. Train the ML model
```bash
python health/heart_rate_model.py
```

### 4. Run the dashboard
```bash
streamlit run app.py
```

---

## 🧠 AI Models Used

| Module | Algorithm | Purpose |
|---|---|---|
| Drowsiness | EAR + dlib landmarks | Eye closure & yawn detection |
| Health Risk | Random Forest | Heart rate classification |
| Health Risk | Gradient Boosting | Heart rate classification |
| Final Output | Weighted Fusion (45% RF + 55% GB) | Risk level prediction |

---

## 📊 Risk Classification

| Level | Condition |
|---|---|
| ✅ NORMAL | All vitals in safe range, eyes open |
| ⚠️ WARNING | Elevated HR / low SpO2 / high blink rate |
| 🚨 CRITICAL | Eyes closed >20 frames / HR >140 BPM / SpO2 <94% |

---

## 🔧 Tech Stack

- **Python** — Core language
- **OpenCV + dlib** — Computer vision
- **scikit-learn** — ML models
- **Streamlit** — Dashboard UI
- **pandas / numpy** — Data processing
- **pygame** — Audio alerts

---

## 👨‍💻 Team

- **Surendhar N** — surendharnatraj123@gmail.com | +91 8015664756
- **Paul Francis** — DSCET, Chennai
- Guided by: Dhanalakshmi Srinivasan College of Engineering and Technology
