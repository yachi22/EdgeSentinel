
import time
import serial
import numpy as np
import pandas as pd

PORT = "COM13"
BAUD_RATE = 115200

BASELINE_FILE = "edgesentinel_clean_baseline.csv"

FEATURES = ["ax", "ay", "az", "gx", "gy", "gz"]
WINDOW_SIZE = 100
MIN_FEATURES_EXCEEDED = 3
CONSECUTIVE_WINDOWS = 2

# Learn variability thresholds from the existing clean baseline.
baseline = pd.read_csv(BASELINE_FILE)[FEATURES].dropna()

baseline_windows = []

for start in range(0, len(baseline) - WINDOW_SIZE + 1, WINDOW_SIZE):
    window = baseline.iloc[start:start + WINDOW_SIZE]
    baseline_windows.append(window.std().to_numpy())

thresholds = np.quantile(
    np.asarray(baseline_windows), 0.99, axis=0
)

print("EdgeSentinel Live Detector")
print("Port:", PORT)
print("Window size:", WINDOW_SIZE, "readings")
print("Keep the sensor still initially.")
print("Press Ctrl+C to stop.\n")

buffer = []
consecutive_flags = 0
last_status = None

try:
    with serial.Serial(PORT, BAUD_RATE, timeout=2) as arduino:
        time.sleep(2)
        arduino.reset_input_buffer()

        while True:
            line = arduino.readline().decode(
                "utf-8", errors="ignore"
            ).strip()

            values = line.split(",")

            if len(values) != 7:
                continue

            try:
                numbers = [float(value) for value in values]
            except ValueError:
                continue

            # Ignore time_ms; keep the six sensor features.
            buffer.append(numbers[1:])

            if len(buffer) < WINDOW_SIZE:
                continue

            window = pd.DataFrame(buffer, columns=FEATURES)
            variability = window.std().to_numpy()

            exceeded = np.sum(variability > thresholds)
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

            if status != last_status:
                print(
                    f"{time.strftime('%H:%M:%S')} | {status} | "
                    f"Features exceeded: {exceeded}/6"
                )
                last_status = status

            # Slide forward by one full window.
            buffer = []

except KeyboardInterrupt:
    print("\nEdgeSentinel stopped.")
except serial.SerialException as error:
    print("\nSerial connection error:", error)
    print("Check that COM13 is correct and no other program is using it.")
