/**
 * EdgeSentinel Live Monitoring Dashboard
 * Frontend Telemetry & Event Polling
 */

document.addEventListener("DOMContentLoaded", () => {
    // Clock in topbar
    function updateClock() {
        const now = new Date();
        const timeStr = now.toTimeString().split(" ")[0];
        const clockEl = document.getElementById("clock");
        if (clockEl) {
            clockEl.textContent = timeStr;
        }
    }
    setInterval(updateClock, 1000);
    updateClock();

    // DOM Elements
    const connectionDot = document.getElementById("connectionDot");
    const connectionText = document.getElementById("connectionText");
    const statusBanner = document.getElementById("statusBanner");
    const statusSymbol = document.getElementById("statusSymbol");
    const systemStatus = document.getElementById("systemStatus");
    const statusDescription = document.getElementById("statusDescription");
    const featuresExceeded = document.getElementById("featuresExceeded");
    const errorBanner = document.getElementById("errorBanner");
    const errorMessage = document.getElementById("errorMessage");
    const samplesProcessed = document.getElementById("samplesProcessed");
    const lastUpdated = document.getElementById("lastUpdated");
    const featureMeter = document.getElementById("featureMeter");
    const eventList = document.getElementById("eventList");

    const sensorElements = {
        ax: document.getElementById("ax"),
        ay: document.getElementById("ay"),
        az: document.getElementById("az"),
        gx: document.getElementById("gx"),
        gy: document.getElementById("gy"),
        gz: document.getElementById("gz")
    };

    const FEATURE_NAMES = ["ax", "ay", "az", "gx", "gy", "gz"];
    const FEATURE_LABELS = {
        ax: "AX (Acc X)",
        ay: "AY (Acc Y)",
        az: "AZ (Acc Z)",
        gx: "GX (Gyro X)",
        gy: "GY (Gyro Y)",
        gz: "GZ (Gyro Z)"
    };

    /**
     * Update Status Banner & Visuals
     */
    function updateStatusDisplay(status, connected, error, exceededCount) {
        // Remove old status classes
        statusBanner.classList.remove(
            "status-normal",
            "status-checking",
            "status-anomaly",
            "status-disconnected"
        );

        if (!connected || status === "DISCONNECTED") {
            statusBanner.classList.add("status-disconnected");
            statusSymbol.textContent = "✕";
            systemStatus.textContent = "DISCONNECTED";
            statusDescription.textContent = error
                ? error
                : "Waiting for Arduino Uno connection on COM13...";

            if (connectionDot) {
                connectionDot.className = "connection-indicator disconnected";
            }
            if (connectionText) {
                connectionText.textContent = "Disconnected";
            }
        } else if (status === "ANOMALY CONFIRMED") {
            statusBanner.classList.add("status-anomaly");
            statusSymbol.textContent = "▲";
            systemStatus.textContent = "ANOMALY CONFIRMED";
            statusDescription.textContent = "Unusual sensor variation detected across consecutive 100-sample windows.";

            if (connectionDot) {
                connectionDot.className = "connection-indicator connected";
            }
            if (connectionText) {
                connectionText.textContent = "Connected";
            }
        } else if (status === "CHECKING MOVEMENT") {
            statusBanner.classList.add("status-checking");
            statusSymbol.textContent = "⚠";
            systemStatus.textContent = "CHECKING MOVEMENT";
            statusDescription.textContent = "Variation exceeded threshold on this window. Evaluating consecutive window...";

            if (connectionDot) {
                connectionDot.className = "connection-indicator connected";
            }
            if (connectionText) {
                connectionText.textContent = "Connected";
            }
        } else {
            // NORMAL
            statusBanner.classList.add("status-normal");
            statusSymbol.textContent = "✓";
            systemStatus.textContent = "NORMAL";
            statusDescription.textContent = "Sensor baseline variability within expected operating limits.";

            if (connectionDot) {
                connectionDot.className = "connection-indicator connected";
            }
            if (connectionText) {
                connectionText.textContent = "Connected";
            }
        }

        if (featuresExceeded) {
            featuresExceeded.textContent = `${exceededCount !== undefined ? exceededCount : 0} / 6`;
        }

        // Connection error alert banner
        if (error && error.trim().length > 0) {
            errorBanner.style.display = "flex";
            errorMessage.textContent = error;
        } else {
            errorBanner.style.display = "none";
            errorMessage.textContent = "";
        }
    }

    /**
     * Update Live Sensor Readings
     */
    function updateReadings(readings, connected) {
        FEATURE_NAMES.forEach(feat => {
            const el = sensorElements[feat];
            if (!el) return;

            if (connected && readings && readings[feat] !== undefined && readings[feat] !== null) {
                const val = Number(readings[feat]);
                el.textContent = isNaN(val) ? "--" : val.toFixed(4);
            } else {
                // Never fabricate sensor values when disconnected
                el.textContent = "--";
            }
        });
    }

    /**
     * Render the 6-Feature Variability Indicator
     */
    function updateFeatureMeter(thresholds, variability, featureExceeded, connected) {
        if (!featureMeter) return;

        let html = "";
        FEATURE_NAMES.forEach(feat => {
            const label = FEATURE_LABELS[feat] || feat.toUpperCase();
            const threshold = thresholds && thresholds[feat] !== undefined ? thresholds[feat] : null;
            const variance = variability && variability[feat] !== undefined ? variability[feat] : null;
            const isExceeded = featureExceeded && featureExceeded[feat] === true;

            let statusClass = "awaiting";
            let statusText = "Awaiting";
            let percentFill = 0;

            if (connected && variance !== null && threshold !== null) {
                if (isExceeded) {
                    statusClass = "high";
                    statusText = "EXCEEDED";
                    percentFill = 100;
                } else {
                    statusClass = "normal";
                    statusText = "NORMAL";
                    percentFill = Math.min(100, Math.max(5, (variance / (threshold || 1)) * 100));
                }
            }

            const varText = variance !== null ? variance.toFixed(4) : "--";
            const thrText = threshold !== null ? threshold.toFixed(4) : "--";

            html += `
                <div class="meter-card ${isExceeded ? 'exceeded' : ''}">
                    <div class="meter-top">
                        <span class="meter-feature">${label}</span>
                        <span class="meter-status ${statusClass}">${statusText}</span>
                    </div>
                    <div class="meter-bar-container">
                        <div class="meter-bar-fill ${isExceeded ? 'exceeded' : ''}" style="width: ${percentFill}%;"></div>
                    </div>
                    <div class="meter-values">
                        <span>Var: <strong>${varText}</strong></span>
                        <span>Lim: <strong>${thrText}</strong></span>
                    </div>
                </div>
            `;
        });

        featureMeter.innerHTML = html;
    }

    /**
     * Render Recent Events History
     */
    function updateEventList(events) {
        if (!eventList) return;

        if (!events || events.length === 0) {
            eventList.innerHTML = `<p class="empty-state">Waiting for the first event...</p>`;
            return;
        }

        let html = "";
        events.forEach(ev => {
            let statusClass = "status-normal";
            const rawStatus = (ev.status || "").toUpperCase();

            if (rawStatus.includes("ANOMALY")) {
                statusClass = "status-anomaly-confirmed";
            } else if (rawStatus.includes("CHECKING")) {
                statusClass = "status-checking-movement";
            } else if (rawStatus.includes("DISCONNECTED")) {
                statusClass = "status-disconnected";
            }

            html += `
                <div class="event-item">
                    <span class="event-time">${ev.time || "--:--:--"}</span>
                    <span class="event-status ${statusClass}">${ev.status || "UNKNOWN"}</span>
                    <span class="event-features">${ev.features_exceeded !== undefined ? ev.features_exceeded : 0}/6 exceeded</span>
                </div>
            `;
        });

        eventList.innerHTML = html;
    }

    /**
     * Fetch status from /api/status (Approximately 1/sec)
     */
    async function fetchStatus() {
        try {
            const response = await fetch("/api/status", { cache: "no-store" });
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            const data = await response.json();

            // 1. Connection & Banner Status
            updateStatusDisplay(
                data.status,
                data.connected,
                data.error,
                data.features_exceeded
            );

            // 2. Telemetry Readings
            updateReadings(data.readings, data.connected);

            // 3. 6-Feature Variability Indicator
            updateFeatureMeter(
                data.thresholds,
                data.variability,
                data.feature_exceeded,
                data.connected
            );

            // 4. Recent Events
            updateEventList(data.events);

            // 5. Samples Processed Count
            if (samplesProcessed) {
                samplesProcessed.textContent = (data.samples_processed || 0).toLocaleString();
            }

            // 6. Last Updated
            if (lastUpdated) {
                if (data.timestamp) {
                    lastUpdated.textContent = `Last sensor update: ${data.timestamp}`;
                } else if (!data.connected) {
                    lastUpdated.textContent = "Offline (no data stream)";
                } else {
                    lastUpdated.textContent = `Awaiting samples (${data.samples_processed || 0} received)`;
                }
            }

        } catch (err) {
            // Web server unreachable or temporary fetch issue
            if (connectionDot) {
                connectionDot.className = "connection-indicator disconnected";
            }
            if (connectionText) {
                connectionText.textContent = "Server Offline";
            }
            if (errorBanner) {
                errorBanner.style.display = "flex";
                if (errorMessage) {
                    errorMessage.textContent = `Dashboard API communication error: ${err.message}. Retrying...`;
                }
            }
        }
    }

    // Initial fetch and poll every 1000ms
    fetchStatus();
    setInterval(fetchStatus, 1000);
});
