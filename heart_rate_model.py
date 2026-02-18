# health/heart_rate_model.py
# ─────────────────────────────────────────────────────────────
# Heart Rate Anomaly Detection using Fusion Model
# Random Forest + Gradient Boosting (as per project spec)
# Risk Classification: Normal / Warning / Critical
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ── Risk Labels ──────────────────────────────────────────────
RISK_LABELS = {0: "NORMAL", 1: "WARNING", 2: "CRITICAL"}
RISK_COLORS = {"NORMAL": "green", "WARNING": "orange", "CRITICAL": "red"}


# ── Generate Synthetic Dataset (use real dataset if available) ─
def generate_dataset(n_samples=2000, save_path="health/dataset.csv"):
    """
    Generate a synthetic heart rate dataset for training.
    Replace with real wearable sensor data for production.

    Features:
    - heart_rate       : BPM (beats per minute)
    - hrv              : Heart Rate Variability (ms)
    - spo2             : Blood oxygen level (%)
    - skin_temp        : Skin temperature (°C)
    - activity_level   : 0=rest, 1=light, 2=moderate, 3=active
    - time_of_day_hr   : Hour of day (0-23)
    """
    np.random.seed(42)

    data = []
    for _ in range(n_samples):
        activity = np.random.randint(0, 4)

        # Normal driver ranges
        if activity == 0:   # Resting
            hr   = np.random.normal(70, 8)
            hrv  = np.random.normal(55, 10)
            spo2 = np.random.normal(98, 1)
        elif activity == 1: # Light
            hr   = np.random.normal(85, 10)
            hrv  = np.random.normal(45, 8)
            spo2 = np.random.normal(97.5, 1)
        elif activity == 2: # Moderate
            hr   = np.random.normal(100, 12)
            hrv  = np.random.normal(35, 7)
            spo2 = np.random.normal(97, 1.2)
        else:               # Active
            hr   = np.random.normal(120, 15)
            hrv  = np.random.normal(25, 6)
            spo2 = np.random.normal(96.5, 1.5)

        skin_temp   = np.random.normal(34, 1.5)
        time_of_day = np.random.randint(0, 24)

        # ── Assign Risk Label ─────────────────────────
        if (hr > 140 or hr < 45 or spo2 < 94 or
                hrv < 15 or skin_temp > 38):
            label = 2  # CRITICAL
        elif (hr > 110 or hr < 55 or spo2 < 96 or
              hrv < 25 or time_of_day in range(1, 5)):
            label = 1  # WARNING (nighttime driving also risky)
        else:
            label = 0  # NORMAL

        data.append([
            round(hr, 1), round(hrv, 1), round(spo2, 1),
            round(skin_temp, 1), activity, time_of_day, label
        ])

    df = pd.DataFrame(data, columns=[
        "heart_rate", "hrv", "spo2",
        "skin_temp", "activity_level", "time_of_day_hr", "risk_label"
    ])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[INFO] Dataset saved to {save_path} — {len(df)} samples")
    return df


# ── Fusion Model Class ────────────────────────────────────────
class DriverHealthModel:
    def __init__(self):
        self.rf_model  = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            random_state=42, class_weight="balanced"
        )
        self.gb_model  = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=5, random_state=42
        )
        self.scaler    = StandardScaler()
        self.is_trained = False
        self.feature_cols = [
            "heart_rate", "hrv", "spo2",
            "skin_temp", "activity_level", "time_of_day_hr"
        ]

    def train(self, df):
        """Train both models on the dataset."""
        X = df[self.feature_cols].values
        y = df["risk_label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        # Train both models
        print("[INFO] Training Random Forest...")
        self.rf_model.fit(X_train, y_train)

        print("[INFO] Training Gradient Boosting...")
        self.gb_model.fit(X_train, y_train)

        # ── Evaluate ─────────────────────────────────
        rf_preds = self.rf_model.predict(X_test)
        gb_preds = self.gb_model.predict(X_test)
        fused    = self._fuse_predictions(
            self.rf_model.predict_proba(X_test),
            self.gb_model.predict_proba(X_test)
        )

        print(f"\n[RESULTS] Random Forest Accuracy : {accuracy_score(y_test, rf_preds):.4f}")
        print(f"[RESULTS] Gradient Boosting Accuracy: {accuracy_score(y_test, gb_preds):.4f}")
        print(f"[RESULTS] Fusion Model Accuracy  : {accuracy_score(y_test, fused):.4f}")
        print("\n[REPORT] Fusion Model:\n", classification_report(
            y_test, fused, target_names=list(RISK_LABELS.values())
        ))

        self.is_trained = True
        return accuracy_score(y_test, fused)

    def _fuse_predictions(self, rf_proba, gb_proba, rf_weight=0.45, gb_weight=0.55):
        """Weighted average fusion of RF + GB probabilities."""
        combined = (rf_weight * rf_proba) + (gb_weight * gb_proba)
        return np.argmax(combined, axis=1)

    def predict(self, heart_rate, hrv, spo2, skin_temp,
                activity_level=0, time_of_day_hr=12):
        """Predict risk level for a single reading."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        features = np.array([[
            heart_rate, hrv, spo2,
            skin_temp, activity_level, time_of_day_hr
        ]])
        features_scaled = self.scaler.transform(features)

        rf_proba = self.rf_model.predict_proba(features_scaled)
        gb_proba = self.gb_model.predict_proba(features_scaled)
        label    = self._fuse_predictions(rf_proba, gb_proba)[0]
        confidence = max(
            ((0.45 * rf_proba) + (0.55 * gb_proba))[0]
        )

        return {
            "risk_label": int(label),
            "risk_level": RISK_LABELS[label],
            "confidence": round(float(confidence) * 100, 1),
            "heart_rate": heart_rate,
            "hrv": hrv,
            "spo2": spo2,
            "skin_temp": skin_temp
        }

    def save(self, path="models/health_model.pkl"):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "rf": self.rf_model,
            "gb": self.gb_model,
            "scaler": self.scaler
        }, path)
        print(f"[INFO] Model saved to {path}")

    def load(self, path="models/health_model.pkl"):
        """Load saved model from disk."""
        data = joblib.load(path)
        self.rf_model  = data["rf"]
        self.gb_model  = data["gb"]
        self.scaler    = data["scaler"]
        self.is_trained = True
        print(f"[INFO] Model loaded from {path}")


# ── Train & Save on first run ────────────────────────────────
if __name__ == "__main__":
    df    = generate_dataset()
    model = DriverHealthModel()
    model.train(df)
    model.save()
    print("\n[TEST] Sample prediction:")
    print(model.predict(heart_rate=135, hrv=18, spo2=93,
                        skin_temp=37.5, activity_level=1, time_of_day_hr=3))
