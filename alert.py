# alerts/alert.py
# ─────────────────────────────────────────────────────────────
# Alert System — Sound + Visual notifications
# Handles: NORMAL / WARNING / CRITICAL risk levels
# ─────────────────────────────────────────────────────────────

import time
import threading

# Try pygame for sound (optional)
try:
    import pygame
    pygame.mixer.init()
    SOUND_AVAILABLE = True
except Exception:
    SOUND_AVAILABLE = False

ALERT_COLORS = {
    "NORMAL":   "\033[92m",   # Green
    "WARNING":  "\033[93m",   # Yellow
    "CRITICAL": "\033[91m",   # Red
}
RESET = "\033[0m"

# ── Alert cooldown tracker ────────────────────────────────────
_last_alert_time  = {}
COOLDOWN_SECONDS  = {"NORMAL": 5, "WARNING": 10, "CRITICAL": 3}


def _play_beep(risk_level):
    """Play alert sound based on risk level (if pygame available)."""
    if not SOUND_AVAILABLE:
        return
    freq_map = {"WARNING": 800, "CRITICAL": 1200}
    freq = freq_map.get(risk_level, 0)
    if freq == 0:
        return
    duration_ms = 500 if risk_level == "WARNING" else 1000
    try:
        sound = pygame.sndarray.make_sound(
            (4096 * (
                __import__('numpy').sin(
                    2 * __import__('numpy').pi *
                    __import__('numpy').arange(44100 * duration_ms // 1000) *
                    freq / 44100
                )
            )).astype(__import__('numpy').int16)
        )
        sound.play()
    except Exception:
        pass  # Silent fallback


def trigger_alert(risk_level: str, source: str = "SYSTEM",
                  details: dict = None) -> dict:
    """
    Trigger an alert for the given risk level.

    Args:
        risk_level : "NORMAL" | "WARNING" | "CRITICAL"
        source     : Where the alert came from ("DROWSINESS" | "HEALTH" | "FUSION")
        details    : Extra info dict (e.g. EAR value, heart rate)

    Returns:
        alert_dict with timestamp, message, and action
    """
    now = time.time()
    cooldown = COOLDOWN_SECONDS.get(risk_level, 5)

    # ── Cooldown check ────────────────────────────────────
    if risk_level in _last_alert_time:
        if now - _last_alert_time[risk_level] < cooldown:
            return {"suppressed": True, "risk_level": risk_level}

    _last_alert_time[risk_level] = now

    # ── Build alert message ───────────────────────────────
    messages = {
        "NORMAL":   "✅ Driver status is NORMAL. All readings within safe range.",
        "WARNING":  "⚠️  WARNING! Signs of fatigue or elevated heart rate detected.",
        "CRITICAL": "🚨 CRITICAL ALERT! Immediate danger — pull over now!"
    }

    actions = {
        "NORMAL":   "Continue monitoring.",
        "WARNING":  "Recommend rest break in 15 minutes.",
        "CRITICAL": "STOP VEHICLE IMMEDIATELY. Sound alarm."
    }

    alert = {
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "risk_level":  risk_level,
        "source":      source,
        "message":     messages.get(risk_level, "Unknown risk level"),
        "action":      actions.get(risk_level, ""),
        "details":     details or {},
        "suppressed":  False
    }

    # ── Console output ────────────────────────────────────
    color = ALERT_COLORS.get(risk_level, "")
    print(f"\n{color}{'='*55}")
    print(f"  [{alert['timestamp']}] {alert['source']} ALERT")
    print(f"  RISK: {risk_level}")
    print(f"  {alert['message']}")
    print(f"  ACTION: {alert['action']}")
    if details:
        for k, v in details.items():
            print(f"  {k}: {v}")
    print(f"{'='*55}{RESET}\n")

    # ── Sound alert in background thread ─────────────────
    if risk_level in ("WARNING", "CRITICAL"):
        t = threading.Thread(target=_play_beep, args=(risk_level,), daemon=True)
        t.start()

    return alert


class AlertLogger:
    """Keeps a session log of all alerts."""

    def __init__(self):
        self.log = []

    def add(self, alert: dict):
        if not alert.get("suppressed"):
            self.log.append(alert)

    def get_summary(self):
        total    = len(self.log)
        critical = sum(1 for a in self.log if a["risk_level"] == "CRITICAL")
        warnings = sum(1 for a in self.log if a["risk_level"] == "WARNING")
        return {
            "total_alerts": total,
            "critical":     critical,
            "warnings":     warnings,
            "normal":       total - critical - warnings
        }

    def clear(self):
        self.log.clear()


# ── Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logger = AlertLogger()
    for level in ["NORMAL", "WARNING", "CRITICAL"]:
        alert = trigger_alert(level, source="TEST",
                              details={"heart_rate": 130, "ear": 0.18})
        logger.add(alert)
        time.sleep(0.5)
    print("Session Summary:", logger.get_summary())
