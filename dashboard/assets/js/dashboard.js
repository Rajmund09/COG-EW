// COG-EW Dashboard Client Script

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    predictSample();
});

async function checkHealth() {
    const statusEl = document.getElementById("systemStatus");
    if (!statusEl) return;

    try {
        const response = await fetch("/api/health");
        if (response.ok) {
            const data = await response.json();
            statusEl.className = "status online";
            statusEl.textContent = "● Online";
            statusEl.title = `Device: ${data.device} | CUDA: ${data.cuda_available}`;
        } else {
            statusEl.className = "status";
            statusEl.textContent = "● Offline";
        }
    } catch (err) {
        statusEl.className = "status";
        statusEl.textContent = "● Disconnected";
    }
}

async function predictSample() {
    const inputEl = document.getElementById("sampleIndex");
    const resultEl = document.getElementById("predictionResult");
    if (!inputEl || !resultEl) return;

    const sampleIndex = parseInt(inputEl.value, 10);
    if (isNaN(sampleIndex) || sampleIndex < 0) {
        resultEl.innerHTML = `<div class="error">Please enter a valid sample index (&ge; 0).</div>`;
        return;
    }

    resultEl.innerHTML = `<div class="loading">Running RF CNN inference on sample #${sampleIndex}...</div>`;

    try {
        const response = await fetch("/api/predict-test", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ index: sampleIndex }),
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Request failed with status ${response.status}`);
        }

        const data = await response.json();
        const pred = data.prediction || {};
        const quality = data.quality || {};
        const isMatch = data.correct;

        // Probabilities breakdown list
        let probBars = "";
        if (pred.probabilities) {
            probBars = Object.entries(pred.probabilities)
                .map(([cls, pct]) => {
                    const highlight = cls === pred.predicted_class ? 'style="color: var(--accent-cyan); font-weight: bold;"' : "";
                    return `<span ${highlight}>${cls}: ${pct}%</span>`;
                })
                .join(" &bull; ");
        }

        resultEl.innerHTML = `
            <div class="prediction-main">
                <span class="eyebrow">${isMatch ? "CORRECT CLASSIFICATION" : "PREDICTION NOTICE"}</span>
                <h1>${pred.predicted_class || "UNKNOWN"}</h1>
                <p style="color: ${isMatch ? 'var(--accent-green)' : 'var(--accent-red)'}; font-weight: 600; margin-bottom: 8px;">
                    ${isMatch ? "✓ MATCHES GROUND TRUTH" : "✗ MISMATCH"}: True Class is ${data.true_class}
                </p>
                <p style="font-size: 14px; color: var(--text-secondary); margin-bottom: 12px;">
                    Model Confidence: <strong>${pred.confidence}%</strong>
                </p>
                <div style="font-size: 12px; color: var(--text-secondary);">${probBars}</div>
            </div>

            <div class="metrics">
                <div>
                    <span>SNR</span>
                    <strong>${data.snr !== undefined ? data.snr + " dB" : "N/A"}</strong>
                </div>
                <div>
                    <span>Dataset Source</span>
                    <strong>${data.source_name || "N/A"}</strong>
                </div>
                <div>
                    <span>Signal Quality</span>
                    <strong style="color: ${quality.signal_quality === 'GOOD' ? 'var(--accent-green)' : quality.signal_quality === 'MODERATE' ? 'var(--accent-cyan)' : 'var(--accent-red)'}">
                        ${quality.signal_quality || "N/A"}
                    </strong>
                </div>
                <div>
                    <span>Signal Power</span>
                    <strong>${quality.signal_power !== undefined ? quality.signal_power : "N/A"}</strong>
                </div>
            </div>

            <div class="decision">
                <div>
                    <span>Sample ID: #${data.index} (Dataset Index: ${data.original_index})</span>
                    <p style="font-size: 13px; margin-top: 4px; color: var(--text-primary);">
                        DC Offset: ${quality.dc_offset} | I/Q Imbalance: ${quality.iq_imbalance}
                    </p>
                </div>
                <strong>${quality.signal_quality === 'GOOD' ? 'HIGH CONFIDENCE' : 'ADAPTIVE FILTERING ADVISED'}</strong>
            </div>
        `;
    } catch (err) {
        resultEl.innerHTML = `<div class="error">Error: ${err.message}</div>`;
    }
}
