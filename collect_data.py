
import csv
import time
import serial

PORT = "COM13"
BAUD_RATE = 115200
DURATION_SECONDS = 60
OUTPUT_FILE = "edgesentinel_baseline_v2.csv"

phases = [
    ("stationary_before", 20),
    ("gentle_movement", 20),
    ("stationary_after", 20),
]

with serial.Serial(PORT, BAUD_RATE, timeout=2) as arduino:
    time.sleep(2)
    arduino.reset_input_buffer()

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "time_ms", "ax", "ay", "az",
            "gx", "gy", "gz", "phase"
        ])

        for phase, duration in phases:
            print(f"\nStarting: {phase}")

            if phase == "stationary_before":
                print("Keep the sensor still.")
            elif phase == "gentle_movement":
                print("Gently tilt the sensor by hand.")
            else:
                print("Put it down and keep it still.")

            start = time.time()

            while time.time() - start < duration:
                line = arduino.readline().decode(
                    "utf-8", errors="ignore"
                ).strip()

                values = line.split(",")

                if len(values) != 7:
                    continue

                try:
                    numbers = [float(v) for v in values]
                except ValueError:
                    continue

                writer.writerow(numbers + [phase])

    print(f"\nFinished! Saved to {OUTPUT_FILE}")
