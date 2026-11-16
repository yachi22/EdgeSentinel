import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure root and dashboard are in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

from dashboard.app import app, state, FEATURES, WINDOW_SIZE, BASELINE_FILE, MODEL_FILE

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Verify that index route serves the industrial dashboard HTML with required components."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    # Core branding and shell
    assert "EdgeSentinel" in html
    assert "INDUSTRIAL ANALYTICS" in html

    # Views & navigation
    assert "data-view=\"overview\"" in html
    assert "data-view=\"analytics\"" in html
    assert "data-view=\"ai\"" in html
    assert "data-view=\"history\"" in html
    assert "data-view=\"diagnostics\"" in html

    # Theme toggle
    assert "id=\"themeToggleBtn\"" in html

    # Time series chart canvases
    assert "id=\"accelChart\"" in html
    assert "id=\"gyroChart\"" in html

    # Script tags for offline Chart.js and application JS
    assert "/static/chart.umd.min.js" in html
    assert "/static/dashboard.js" in html

def test_api_status_endpoint(client):
    """Verify that /api/status returns valid JSON adhering to the established API contract."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)

    # Required contract keys
    assert "connected" in data
    assert "status" in data
    assert "readings" in data
    assert "features_exceeded" in data
    assert "samples_processed" in data
    assert "thresholds" in data
    assert "variability" in data
    assert "feature_exceeded" in data
    assert "port" in data
    assert "baud_rate" in data
    assert "events" in data
    assert isinstance(data["events"], list)

    # Enhanced industrial analytics keys
    assert "model_loaded" in data
    assert "model_name" in data
    assert "diagnostics" in data
    assert isinstance(data["diagnostics"], dict)
    assert "recent_readings" in data
    assert isinstance(data["recent_readings"], list)

    # Check host diagnostics structure
    diag = data["diagnostics"]
    assert "sensor_model" in diag
    assert "server_uptime_seconds" in diag
    assert "data_freshness" in diag

def test_features_list():
    """Verify 6-DOF IMU features list."""
    assert FEATURES == ["ax", "ay", "az", "gx", "gy", "gz"]
    assert len(FEATURES) == 6
    assert WINDOW_SIZE == 100

def test_baseline_file_and_threshold_calculation():
    """Verify clean baseline file exists and thresholds can be computed."""
    assert BASELINE_FILE.exists(), f"Baseline file missing at {BASELINE_FILE}"
    df = pd.read_csv(BASELINE_FILE)
    for feat in FEATURES:
        assert feat in df.columns

    clean = df[FEATURES].dropna()
    assert len(clean) >= WINDOW_SIZE

    windows = []
    for start in range(0, len(clean) - WINDOW_SIZE + 1, WINDOW_SIZE):
        w = clean.iloc[start:start + WINDOW_SIZE]
        windows.append(w.std().to_numpy())

    thresholds = np.quantile(np.asarray(windows), 0.99, axis=0)
    assert len(thresholds) == 6
    for t in thresholds:
        assert t > 0
        assert not np.isnan(t)

def test_isolation_forest_model_bundle():
    """Verify IsolationForest model bundle can be loaded and predict samples."""
    if not MODEL_FILE.exists():
        pytest.skip("Model file not present")

    bundle = joblib.load(MODEL_FILE)
    assert isinstance(bundle, dict)
    assert "model" in bundle
    assert "scaler" in bundle
    assert "features" in bundle

    scaler = bundle["scaler"]
    model = bundle["model"]
    features = bundle["features"]

    assert features == FEATURES

    # Test sample inference on synthetic zero-motion reading
    sample = pd.DataFrame([[0.0, 0.0, 9.8, 0.0, 0.0, 0.0]], columns=features)
    scaled = scaler.transform(sample)
    score = float(model.decision_function(scaled)[0])
    pred = int(model.predict(scaled)[0])

    assert isinstance(score, float)
    assert pred in [-1, 1]

def test_css_design_tokens_palette():
    """Verify that style.css contains all user-specified theme palette tokens."""
    css_path = ROOT / "dashboard" / "static" / "style.css"
    assert css_path.exists()
    css_text = css_path.read_text(encoding="utf-8")

    # Light Theme Palette Checks
    assert "#F4F6F8" in css_text  # Page bg
    assert "#FFFFFF" in css_text  # Card bg
    assert "#172B24" in css_text  # Primary text
    assert "#64748B" in css_text  # Secondary text
    assert "#E2E8F0" in css_text  # Borders
    assert "#26734D" in css_text  # Primary accent
    assert "#32966A" in css_text  # Success
    assert "#B7791F" in css_text  # Warning
    assert "#C24141" in css_text  # Critical

    # Dark Theme Palette Checks
    assert "#171A1F" in css_text  # Page bg
    assert "#22272E" in css_text  # Card bg
    assert "#F1F5F9" in css_text  # Primary text
    assert "#A7B0BD" in css_text  # Secondary text
    assert "#39414B" in css_text  # Borders
    assert "#8FD3A9" in css_text  # Primary accent & Success
    assert "#F3C477" in css_text  # Warning
    assert "#F87171" in css_text  # Critical

def test_cross_platform_port_and_baud_rate():
    """Verify that baud rate is 115200 and port is properly handled across platforms."""
    from dashboard.app import BAUD_RATE, PORT
    assert BAUD_RATE == 115200
    # In Windows environment, defaults to COM13; on Linux/Raspberry Pi, defaults to /dev/ttyACM0
    if sys.platform.startswith("win"):
        assert "COM" in PORT or "/" in PORT
    else:
        assert PORT == "/dev/ttyACM0"

def test_offline_chart_js_bundle():
    """Verify Chart.js is served locally from static folder for offline edge capability."""
    chart_js = ROOT / "dashboard" / "static" / "chart.umd.min.js"
    assert chart_js.exists(), "chart.umd.min.js must exist in dashboard/static for offline edge operation"
    assert chart_js.stat().st_size > 100_000, "chart.umd.min.js must be a valid complete Chart.js bundle"

def test_truthful_status_representation(client):
    """Verify that when the sensor is not connected, the API truthfully reports disconnected."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.get_json()
    # If no physical hardware is plugged into COM13 / /dev/ttyACM0, connected must not falsely be True
    if not data["connected"]:
        assert data["status"] in ["DISCONNECTED", "CONNECTING"]
        assert data["diagnostics"]["data_freshness"] == "OFFLINE"

