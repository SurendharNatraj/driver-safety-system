# drowsiness/detect_drowsiness.py
# ─────────────────────────────────────────────────────────────
# Eye Blink Rate & Drowsiness Detection using OpenCV + dlib
# Uses Eye Aspect Ratio (EAR) to detect eye closure duration
# ─────────────────────────────────────────────────────────────

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import time

# ── Constants ──────────────────────────────────────────────
EAR_THRESHOLD    = 0.25   # Below this → eye is considered closed
EAR_CONSEC_FRAMES = 20    # Frames eye must be closed to trigger drowsy alert
YAWN_THRESHOLD   = 0.6    # Mouth aspect ratio threshold for yawn detection

# Eye & mouth landmark indexes (dlib 68-point model)
(L_START, L_END) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_START, R_END) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(M_START, M_END) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR) — drops when eye closes."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    """Calculate Mouth Aspect Ratio (MAR) — rises when yawning."""
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (3.0 * D)


class DrowsinessDetector:
    def __init__(self, predictor_path="drowsiness/shape_predictor_68_face_landmarks.dat"):
        print("[INFO] Loading face detector and landmark predictor...")
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.counter      = 0       # Consecutive closed-eye frame counter
        self.blink_count  = 0       # Total blinks detected
        self.yawn_count   = 0       # Total yawns detected
        self.drowsy       = False
        self.yawning      = False
        self.start_time   = time.time()

    def process_frame(self, frame):
        """
        Process a single frame.
        Returns: (annotated_frame, status_dict)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        status = {
            "face_detected": False,
            "ear": 0.0,
            "mar": 0.0,
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "drowsy": False,
            "yawning": False,
            "risk_level": "NORMAL"
        }

        for rect in rects:
            status["face_detected"] = True
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # ── Eyes ─────────────────────────────────────
            left_eye  = shape[L_START:L_END]
            right_eye = shape[R_START:R_END]
            left_ear  = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            status["ear"] = round(ear, 3)

            # ── Mouth ────────────────────────────────────
            mouth = shape[M_START:M_END]
            mar   = mouth_aspect_ratio(mouth)
            status["mar"] = round(mar, 3)

            # ── Drowsiness Logic ─────────────────────────
            if ear < EAR_THRESHOLD:
                self.counter += 1
                if self.counter >= EAR_CONSEC_FRAMES:
                    self.drowsy = True
                    status["drowsy"] = True
                    cv2.putText(frame, "⚠ DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                if self.counter >= 3:
                    self.blink_count += 1
                    status["blink_count"] = self.blink_count
                self.counter = 0
                self.drowsy  = False

            # ── Yawn Logic ───────────────────────────────
            if mar > YAWN_THRESHOLD:
                if not self.yawning:
                    self.yawn_count += 1
                self.yawning = True
                status["yawning"] = True
                cv2.putText(frame, "😮 YAWN DETECTED", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                self.yawning = False

            # ── Draw Eye & Mouth Contours ─────────────────
            for eye in [left_eye, right_eye]:
                hull = cv2.convexHull(eye)
                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

            mouth_hull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 255), 1)

            # ── EAR & MAR display ────────────────────────
            cv2.putText(frame, f"EAR: {ear:.2f}", (frame.shape[1]-150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {self.blink_count}", (frame.shape[1]-150, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── Risk Level ───────────────────────────────────
        if status["drowsy"] and status["yawning"]:
            status["risk_level"] = "CRITICAL"
        elif status["drowsy"] or status["yawn_count"] >= 3:
            status["risk_level"] = "WARNING"
        else:
            status["risk_level"] = "NORMAL"

        return frame, status


# ── Standalone test ──────────────────────────────────────────
if __name__ == "__main__":
    detector = DrowsinessDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, status = detector.process_frame(frame)
        print(status)
        cv2.imshow("Drowsiness Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
