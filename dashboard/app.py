
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import serial
from flask import Flask, jsonify, render_template

ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = ROOT / "edgesentinel_clean_baseline.csv"

PORT = "COM13"
BAUD_RATE = 115200
WINDOW_SIZE = 100
MIN_FEATURES_EXCEEDED = 3
CONSECUTIVE_WINDOWS = 2

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]

app = Flask(__name__)

lock = threading.Lock()
events = deque(maxlen=100)

state = {
    "connected": False,
    "status": "CONNECTING",
    "timestamp": None,
    "readings": {},
    "features_exceeded": 0,
    "error": None,
    "samples_processed": 0,
    "thresholds": {},
    "variability": {},
    "feature_exceeded": {},
    "port": PORT,
    "baud_rate": BAUD_RATE,
}

stop_worker = False

def add_event(status, exceeded):
    event = {
        "time": time.strftime("%H:%M:%S"),
        "status": status,
        "features_exceeded": int(exceeded),
    }
    events.appendleft(event)

def sensor_worker():
    global stop_worker
    last_status = None

    # Step 1: Learn baseline variability thresholds
    try:
        if not BASELINE_FILE.exists():
            raise FileNotFoundError(
                f"Clean baseline file '{BASELINE_FILE.name}' not found. "
                "Ensure edgesentinel_clean_baseline.csv exists in the project root."
            )

        baseline = pd.read_csv(BASELINE_FILE)[FEATURES].dropna()
        baseline_windows = []

        for start in range(0, len(baseline) - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = baseline.iloc[start:start + WINDOW_SIZE]
            baseline_windows.append(window.std().to_numpy())

        if not baseline_windows:
            raise ValueError("Not enough baseline data to calculate thresholds.")

        thresholds = np.quantile(np.asarray(baseline_windows), 0.99, axis=0)

        with lock:
            state["thresholds"] = {
                feat: round(float(t), 5) for feat, t in zip(FEATURES, thresholds)
            }

    except Exception as error:
        with lock:
            state["connected"] = False
            state["status"] = "DISCONNECTED"
            state["error"] = f"Baseline Error: {error}"
            add_event("DISCONNECTED", 0)
        return

    # Step 2: Serial reading loop with auto-reconnect and error guidance
    while not stop_worker:
        buffer = []
        consecutive_flags = 0

        try:
            with serial.Serial(PORT, BAUD_RATE, timeout=2) as arduino:
                # Pulse DTR to reset Arduino Uno ATmega328P and cleanly initialize MPU6050
                try:
                    arduino.dtr = False
                    time.sleep(0.1)
                    arduino.dtr = True
                    time.sleep(2)
                except Exception:
                    time.sleep(2)
                arduino.reset_input_buffer()

                consecutive_zeros = 0
                consecutive_malformed = 0

                with lock:
                    state["connected"] = True
                    state["error"] = None
                    state["status"] = "NORMAL"

                while not stop_worker:
                    line = arduino.readline().decode(
                        "utf-8", errors="ignore"
                    ).strip()

                    if not line:
                        continue

                    values = line.split(",")
                    if len(values) != 7:
                        consecutive_malformed += 1
                        if consecutive_malformed >= 25:
                            with lock:
                                state["error"] = (
                                    f"Malformed serial stream: expected 7 CSV values, received {len(values)}. "
                                    f"Check COM13 baud rate ({BAUD_RATE}) and sensor wiring."
                                )
                        continue

                    try:
                        numbers = [float(value) for value in values]
                    except ValueError:
                        consecutive_malformed += 1
                        if consecutive_malformed >= 25:
                            with lock:
                                state["error"] = (
                                    "Malformed serial stream: non-numeric sensor reading received."
                                )
                        continue

                    consecutive_malformed = 0

                    # Check for MPU6050 sensor-read failure:
                    # A physical MPU6050 in 1g Earth gravity can never have all axes equal to 0.0.
                    # All six features zero indicates an I2C bus freeze or sensor communication fault.
                    is_sensor_failure = all(abs(v) < 1e-6 for v in numbers[1:])

                    if is_sensor_failure:
                        consecutive_zeros += 1
                        if consecutive_zeros >= 5:
                            sensor_error = (
                                "Sensor read failure: MPU6050 returning zero readings (I2C communication fault). "
                                "Check MPU6050 wiring (SDA/SCL) or reconnect Arduino."
                            )
                            with lock:
                                state["connected"] = False
                                state["status"] = "DISCONNECTED"
                                state["error"] = sensor_error
                                state["readings"] = {}

                            if last_status != "DISCONNECTED":
                                with lock:
                                    add_event("DISCONNECTED", 0)
                                last_status = "DISCONNECTED"

                        # If zero readings persist, attempt hardware re-initialization via DTR pulse
                        if consecutive_zeros >= 15:
                            try:
                                arduino.dtr = False
                                time.sleep(0.1)
                                arduino.dtr = True
                                time.sleep(2)
                                arduino.reset_input_buffer()
                            except Exception:
                                pass
                            consecutive_zeros = 0

                        # Discard sensor-failure zeros: do not pollute buffer or count as valid samples
                        continue

                    # Valid sensor readings received
                    consecutive_zeros = 0
                    readings = dict(zip(FEATURES, numbers[1:]))
                    buffer.append(numbers[1:])

                    with lock:
                        state["connected"] = True
                        if state["error"] and "Sensor read failure" in state["error"]:
                            state["error"] = None
                        state["readings"] = readings
                        state["timestamp"] = time.strftime("%H:%M:%S")
                        state["samples_processed"] += 1

                    if len(buffer) < WINDOW_SIZE:
                        continue

                    window = pd.DataFrame(buffer, columns=FEATURES)
                    variability = window.std().to_numpy()
                    exceeded_mask = variability > thresholds
                    exceeded = int(np.sum(exceeded_mask))
                    raw_anomaly = exceeded >= MIN_FEATURES_EXCEEDED

                    if raw_anomaly:
                        consecutive_flags += 1
                    else:
                        consecutive_flags = 0

                    if consecutive_flags >= CONSECUTIVE_WINDOWS:
                        status = "ANOMALY CONFIRMED"
                    elif raw_anomaly:
                        status = "CHECKING MOVEMENT"
                    else:
                        status = "NORMAL"

                    with lock:
                        state["status"] = status
                        state["features_exceeded"] = exceeded
                        state["variability"] = {
                            feat: round(float(v), 5) for feat, v in zip(FEATURES, variability)
                        }
                        state["feature_exceeded"] = {
                            feat: bool(b) for feat, b in zip(FEATURES, exceeded_mask)
                        }

                    if status != last_status:
                        with lock:
                            add_event(status, exceeded)
                        last_status = status

                    buffer = []

        except serial.SerialException as error:
            error_message = (
                f"Serial port {PORT} unavailable: {error}. "
                "Ensure Arduino Uno is plugged in and close live_detector.py or any other serial monitor."
            )
            with lock:
                state["connected"] = False
                state["status"] = "DISCONNECTED"
                state["error"] = error_message
                state["readings"] = {}

            if last_status != "DISCONNECTED":
                with lock:
                    add_event("DISCONNECTED", 0)
                last_status = "DISCONNECTED"

            # Wait before attempting auto-reconnect
            time.sleep(2)

        except Exception as error:
            with lock:
                state["connected"] = False
                state["status"] = "DISCONNECTED"
                state["error"] = f"Hardware error: {error}"
                state["readings"] = {}

            if last_status != "DISCONNECTED":
                with lock:
                    add_event("DISCONNECTED", 0)
                last_status = "DISCONNECTED"

            time.sleep(2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    with lock:
        return jsonify({
            **state,
            "events": list(events),
        })

if __name__ == "__main__":
    worker = threading.Thread(target=sensor_worker, daemon=True)
    worker.start()

    app.run(host="127.0.0.1", port=5000, debug=False)
