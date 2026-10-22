
import csv
import time
import serial

PORT = "COM13"
BAUD_RATE = 115200
OUTPUT_FILE = "edgesentinel_clean_baseline.csv"

print("Place the sensor on a stable, flat surface.")
print("Do not touch or move it during the 60-second recording.")
input("Press Enter when ready...")

with serial.Serial(PORT, BAUD_RATE, timeout=2) as arduino:
    time.sleep(2)
    arduino.reset_input_buffer()

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "time_ms", "ax", "ay", "az",
            "gx", "gy", "gz"
        ])

        start = time.time()

        while time.time() - start < 60:
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

            writer.writerow(numbers)

print(f"Baseline saved to {OUTPUT_FILE}")
