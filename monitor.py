import cv2
import serial
import time
import csv
import mediapipe as mp
import numpy as np
import os
import sys
import datetime

# ===================== Serial: ESP32 + BH1750 + Vibration Motor =====================

SERIAL_PORT = 'COM3'   # Adjust to your actual port if needed
BAUD_RATE = 115200

ser = None
latest_lux = None  # Most recent Lux value

print("[INFO] Starting program...")

# 1) Open serial port
try:
    print(f"[INFO] Opening serial port {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)  # Give ESP32 some time to reset
    print(f"[INFO] Serial port {SERIAL_PORT} opened successfully.")
except Exception as e:
    print(f"[ERROR] Failed to open serial port {SERIAL_PORT}: {e}")
    print("Possible reasons:")
    print("  1. Arduino Serial Monitor is still open (close it and try again).")
    print("  2. The port is not COM3 (check the actual port in the Arduino IDE).")
    input("Press Enter to exit...")
    sys.exit(1)


def update_lux():
    """
    Non-blocking read from serial to update latest_lux
    with the most recent Lux value.
    """
    global latest_lux
    if ser is None:
        return
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        # For debugging you can uncomment the next line to see raw output:
        # print("RAW:", line)
        if line.startswith("LUX:"):
            try:
                value_str = line.split(":", 1)[1]
                latest_lux = float(value_str)
            except ValueError:
                # For example when we see "LUX:ERR"
                latest_lux = None


def trigger_vibration():
    """
    Send '1' to ESP32 to trigger a vibration.
    The exact duration is handled on the ESP32 side.
    """
    try:
        ser.write(b"1")
        print(">>> Sent vibration command '1'")
    except Exception as e:
        print("[ERROR] Failed to send vibration command:", e)


def stop_vibration():
    """
    Send '0' to ESP32 to stop the vibration motor immediately
    (if the firmware uses this command).
    """
    try:
        ser.write(b"0")
        print(">>> Sent stop command '0'")
    except Exception as e:
        print("[ERROR] Failed to send stop command:", e)


# ===================== MediaPipe: Drowsiness Detection =====================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices (MediaPipe FaceMesh)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [61, 81, 291, 308, 13, 14]


def eye_aspect_ratio(pts):
    """Compute Eye Aspect Ratio (EAR)."""
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(pts):
    """Compute Mouth Aspect Ratio (MAR)."""
    A = np.linalg.norm(pts[3] - pts[4])
    B = np.linalg.norm(pts[2] - pts[5])
    C = np.linalg.norm(pts[0] - pts[1])
    mar = (A + B) / (2.0 * C)
    return mar


class DrowsyState:
    """Store accumulated state used during drowsiness detection."""

    def __init__(self):
        self.closed_frames = 0       # Number of consecutive frames with eyes closed
        self.blink_count = 0         # Total blink count
        self.microsleep_events = 0   # Total microsleep event count

        self.yawn_frames = 0         # Number of consecutive frames with mouth open wide
        self.yawn_count = 0          # Total yawn count (for statistics)

        self.drowsy_flag = False     # Whether the current frame is classified as drowsy


# Thresholds (fine-tune these based on your own data)
EAR_THRESH_BLINK = 0.23       # EAR below this threshold is considered eye closed
EAR_FRAMES_BLINK = 3          # Minimum consecutive closed-eye frames to count as a blink
EAR_FRAMES_MICROSLEEP = 180   # Consecutive closed-eye frames to count as microsleep

MAR_THRESH_YAWN = 1.8         # High MAR threshold for yawns (statistics only)
MAR_FRAMES_YAWN = 20          # Consecutive frames above MAR threshold to count as a yawn


def detect_drowsy(frame, state: DrowsyState):
    """
    Use only long-duration eye closure (microsleep) to set the drowsy flag.
    Yawns are counted for statistics and do not affect drowsy_flag.
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    ear = None
    mar = None

    microsleep_now = False

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        mesh_points = np.array(
            [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
        )

        left_eye_pts = mesh_points[LEFT_EYE_IDX]
        right_eye_pts = mesh_points[RIGHT_EYE_IDX]
        mouth_pts = mesh_points[MOUTH_IDX]

        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        ear = (left_ear + right_ear) / 2.0

        mar = mouth_aspect_ratio(mouth_pts)

        # ---------- Blinks & Microsleeps ----------
        if ear < EAR_THRESH_BLINK:
            state.closed_frames += 1
        else:
            # On the transition from closed -> open, classify the previous closed period
            if EAR_FRAMES_BLINK <= state.closed_frames < EAR_FRAMES_MICROSLEEP:
                state.blink_count += 1
            elif state.closed_frames >= EAR_FRAMES_MICROSLEEP:
                state.microsleep_events += 1
            state.closed_frames = 0

        if state.closed_frames >= EAR_FRAMES_MICROSLEEP:
            microsleep_now = True

        # ---------- Yawns (statistics only, not used for drowsy_flag) ----------
        if mar is not None and mar > MAR_THRESH_YAWN:
            state.yawn_frames += 1
        else:
            if state.yawn_frames >= MAR_FRAMES_YAWN:
                state.yawn_count += 1
            state.yawn_frames = 0

        # ---------- Draw helper points ----------
        for p in left_eye_pts:
            cv2.circle(frame, tuple(p.astype(int)), 1, (0, 255, 0), -1)
        for p in right_eye_pts:
            cv2.circle(frame, tuple(p.astype(int)), 1, (0, 255, 0), -1)
        for p in mouth_pts:
            cv2.circle(frame, tuple(p.astype(int)), 1, (255, 0, 0), -1)

        # Show EAR / MAR values
        if ear is not None:
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if mar is not None:
            cv2.putText(frame, f"MAR: {mar:.3f}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ---------- Final drowsy flag for this frame ----------
        state.drowsy_flag = microsleep_now

    else:
        # If no face is detected, treat the current frame as not drowsy and reset counters
        state.drowsy_flag = False
        state.closed_frames = 0
        state.yawn_frames = 0

    return state.drowsy_flag


# ===================== Camera & CSV Initialisation =====================

print("[INFO] Opening camera index 0 ...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Failed to open camera (device index 0).")
    print("Possible reasons:")
    print("  1. The camera is used by another application (Zoom/Teams/another Python script).")
    print("  2. It is not camera index 0 (try 1 or 2).")
    input("Press Enter to exit...")
    ser.close()
    sys.exit(1)

print("[INFO] Camera opened successfully.")

state = DrowsyState()
prev_drowsy = False  # Used to detect the transition from “not drowsy” to “drowsy”

csv_path = "drowsy_log1.csv"

# Check whether the file exists and whether it is empty to decide if we need a header
file_exists = os.path.exists(csv_path)
file_empty = (not file_exists) or (os.path.getsize(csv_path) == 0)

# Open CSV in append mode ("a") so new runs do not overwrite previous data
csv_file = open(csv_path, "a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

if file_empty:
    csv_writer.writerow([
        "timestamp",          # Local real time in ISO format
        "time_sec",           # Relative time since the start of this run
        "lux",
        "drowsy",
        "blink_count",
        "microsleep_events",
        "yawn_count"
    ])

start_time = time.time()

print("[INFO] Initialisation completed.")
print("[INFO] Press 'q' or 'Q' (or ESC) to quit, press 'd' to trigger a vibration manually for testing.")

# ===================== Main Loop =====================

try:
    while True:
        # 1. Update ambient light value
        update_lux()

        # 2. Read from camera
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera. Exiting.")
            break

        # 3. Drowsiness detection
        drowsy = detect_drowsy(frame, state)

        # ====== Drowsiness → Vibration: trigger once on transition False -> True ======
        if (not prev_drowsy) and drowsy:
            trigger_vibration()

        # Optional: send stop command when going from drowsy -> not drowsy
        if prev_drowsy and (not drowsy):
            stop_vibration()

        prev_drowsy = drowsy

        # 4. Overlay text on frame
        if latest_lux is not None:
            cv2.putText(frame,
                        f"Lux: {latest_lux:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2)

        cv2.putText(frame,
                    f"Drowsy: {drowsy}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255) if drowsy else (255, 255, 255),
                    2)

        cv2.putText(frame,
                    f"Blinks: {state.blink_count}  Yawns: {state.yawn_count}  Micro: {state.microsleep_events}",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2)

        cv2.imshow("Drowsiness + Ambient Light + Vibration", frame)

        # 5. Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        if key == ord('d'):
            print(">>> Key 'd' pressed: manual vibration trigger.")
            trigger_vibration()

        # 6. Append one row to CSV
        t = time.time() - start_time

        # Current local timestamp in ISO format
        now_str = datetime.datetime.now().isoformat(timespec="seconds")

        # lux may be None; handle safely
        if latest_lux is not None:
            lux_str = f"{latest_lux:.2f}"
        else:
            lux_str = ""

        csv_writer.writerow([
            now_str,
            f"{t:.2f}",
            lux_str,
            int(drowsy),
            state.blink_count,
            state.microsleep_events,
            state.yawn_count
        ])
        csv_file.flush()   # Flush immediately to avoid losing data if interrupted

except KeyboardInterrupt:
    print("\n[INFO] KeyboardInterrupt detected. Stopping and saving data...")

finally:
    cap.release()
    ser.close()
    csv_file.close()
    cv2.destroyAllWindows()
    print("[INFO] Data appended to:", os.path.abspath(csv_path))
    print("[INFO] Program terminated.")
