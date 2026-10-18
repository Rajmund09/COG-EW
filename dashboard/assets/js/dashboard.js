
"use strict";

/*
 * COG-EW Dashboard Client
 *
 * Existing endpoints used:
 *   GET  /api/health
 *   POST /api/predict-test
 *   GET  /api/signal-quality
 *   GET  /api/decision
 *
 * This script does not load the HDF5 dataset or the CNN in the browser.
 */

const COGEW_CLASSES = ["8PSK", "BPSK", "QPSK", "16QAM", "64QAM"];

document.addEventListener("DOMContentLoaded", async () => {
    await checkHealth();

    const page = window.location.pathname.toLowerCase();

    if (page.endsWith("/signal_analysis.html")) {
        await Promise.all([
            predictSample(),
            loadSavedQualityReport()
        ]);
        return;
    }

    if (page.endsWith("/decision_support.html")) {
        await loadDecisionReport();
        return;
    }

    // Overview and RF Classification retain interactive sample prediction.
    if (
        document.getElementById("sampleIndex") &&
        document.getElementById("predictionResult")
    ) {
        await predictSample();
    }
});


/* ============================================================
   SHARED HELPERS
   ============================================================ */

async function fetchJSON(url, options = {}) {
    const response = await fetch(url, {
        cache: "no-store",
        ...options
    });

    let payload;

    try {
        payload = await response.json();
    } catch {
        payload = {};
    }

    if (!response.ok) {
        throw new Error(
            payload.detail || `Request failed with HTTP ${response.status}`
        );
    }

    if (
        payload &&
        typeof payload === "object" &&
        (payload.status === "missing" || payload.status === "error")
    ) {
        throw new Error(
            payload.error ||
            `The report is ${payload.status}: ${payload.file || url}`
        );
    }

    return payload;
}


function createElement(tag, className, text) {
    const element = document.createElement(tag);

    if (className) {
        element.className = className;
    }

    if (text !== undefined && text !== null) {
        element.textContent = String(text);
    }

    return element;
}


function formatNumber(value, digits = 4) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return value === undefined || value === null ? "N/A" : String(value);
    }

    return value.toFixed(digits);
}


function formatPercent(value, alreadyPercent = false) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        return "N/A";
    }

    const percent = alreadyPercent ? value : value * 100;
    return `${percent.toFixed(2)}%`;
}


function prettyLabel(value) {
    return String(value)
        .replace(/[_-]+/g, " ")
        .replace(/([a-z])([A-Z])/g, "$1 $2")
        .replace(/\b\w/g, character => character.toUpperCase());
}


function setStatus(element, message, state = "normal") {
    if (!element) return;

    element.textContent = message;
    element.className = "status";

    if (state === "success") {
        element.classList.add("online");
    }

    if (state === "error") {
        element.classList.add("offline");
    }
}


function setPanelMessage(element, message, className = "loading") {
    if (!element) return;

    element.replaceChildren(
        createElement("div", className, message)
    );
}


/* ============================================================
   BACKEND HEALTH
   ============================================================ */

async function checkHealth() {
    const statusElement = document.getElementById("systemStatus");

    if (!statusElement) return;

    try {
        const health = await fetchJSON("/api/health");

        if (health.status === "ok") {
            setStatus(statusElement, "● Online", "success");
        } else {
            setStatus(statusElement, "● Backend response received", "normal");
        }

        statusElement.title =
            `Device: ${health.device || "Unknown"} | ` +
            `GPU: ${health.gpu || "Unknown"} | ` +
            `CUDA: ${health.cuda_available ?? "Not reported"}`;

    } catch (error) {
        setStatus(statusElement, "● Disconnected", "error");
        statusElement.title = error.message;
    }
}


/* ============================================================
   RF CLASSIFICATION — EXISTING PREDICTION ENDPOINT
   ============================================================ */

async function predictSample() {
    const input = document.getElementById("sampleIndex");
    const result = document.getElementById("predictionResult");

    if (!input || !result) return;

    const index = Number(input.value);

    if (!Number.isInteger(index) || index < 0) {
        setPanelMessage(
            result,
            "Enter a non-negative integer test index.",
            "error"
        );
        return;
    }

    setPanelMessage(
        result,
        `Running CNN inference on held-out test sample #${index}...`
    );

    try {
        const data = await fetchJSON("/api/predict-test", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ index })
        });

        renderPrediction(data, result);

    } catch (error) {
        setPanelMessage(result, `Prediction failed: ${error.message}`, "error");
    }
}


function renderPrediction(data, host) {
    host.replaceChildren();

    const prediction = data.prediction || data;
    const quality = data.quality || {};

    const predictedClass =
        prediction.predicted_class || data.predicted_class || "Unknown";

    const trueClass = data.true_class || "Unknown";

    const confidence =
        typeof prediction.confidence === "number"
            ? prediction.confidence
            : data.confidence_percent;

    const isCorrect =
        typeof data.correct === "boolean"
            ? data.correct
            : predictedClass === trueClass;

    const main = createElement("div", "prediction-main");

    main.appendChild(
        createElement(
            "span",
            "eyebrow",
            isCorrect ? "MATCHES GROUND TRUTH" : "PREDICTION RESULT"
        )
    );

    main.appendChild(createElement("h1", "", predictedClass));

    main.appendChild(
        createElement(
            "p",
            "",
            `Ground truth: ${trueClass} · ${isCorrect ? "Correct on this sample" : "Incorrect on this sample"
            }`
        )
    );

    main.appendChild(
        createElement(
            "p",
            "",
            `Model confidence: ${confidence === undefined
                ? "N/A"
                : `${Number(confidence).toFixed(2)}%`
            }`
        )
    );

    if (data.source) {
        main.appendChild(
            createElement(
                "p",
                "",
                `Dataset Source: ${data.source} · Signal SNR: ${data.snr_db !== undefined ? data.snr_db : 'N/A'} dB`
            )
        );
    }

    host.appendChild(main);

    const probabilityMap =
        prediction.probabilities ||
        data.probabilities ||
        {};

    if (Object.keys(probabilityMap).length) {
        const probabilityPanel = createElement("div", "panel-box");
        probabilityPanel.appendChild(
            createElement("h3", "panel-title", "Class probability distribution")
        );

        // Display top predicted classes dynamically
        const classesToDisplay = Object.keys(probabilityMap).length > 0
            ? Object.entries(probabilityMap).sort((a, b) => b[1] - a[1]).slice(0, 8).map(x => x[0])
            : COGEW_CLASSES;

        for (const className of classesToDisplay) {
            const raw = probabilityMap[className];
            if (typeof raw !== "number" || !Number.isFinite(raw)) continue;

            const isFraction = raw >= 0 && raw <= 1;
            const pct = isFraction ? raw * 100 : raw;
            const clamped = Math.max(0, Math.min(100, pct));

            const row = createElement("div", "probability-row");
            const label = createElement("span", "probability-label", className);
            const track = createElement("div", "probability-track");
            const fill = createElement("div", "probability-fill");
            const value = createElement(
                "span",
                "probability-value",
                `${pct.toFixed(2)}%`
            );

            fill.style.width = `${clamped}%`;

            if (className === predictedClass) {
                fill.classList.add("is-predicted");
                label.classList.add("is-predicted");
            }

            track.appendChild(fill);
            row.append(label, track, value);
            probabilityPanel.appendChild(row);
        }

        host.appendChild(probabilityPanel);
    }

    const metrics = createElement("div", "metrics");

    const fields = [
        ["Test index", data.test_index ?? data.index],
        ["Threat Category", data.threat_category || "Early Warning / Track"],
        ["EW Countermeasure", data.recommended_countermeasure || "Active Spot Noise Jamming"],
        ["Dataset SNR", data.snr_db ?? data.snr],
        ["Signal power", quality.signal_power || quality.total_power],
        ["IQ imbalance", quality.iq_imbalance],
        ["DC offset", quality.dc_offset],
        ["Inference time", data.inference_ms]
    ];

    for (const [label, value] of fields) {
        const item = createElement("div", "");
        item.appendChild(createElement("span", "", label));

        let display = value;

        if (label === "Dataset SNR" && value !== undefined) {
            display = `${value} dB`;
        } else if (label === "Inference time" && value !== undefined) {
            display = `${value} ms`;
        } else if (typeof value === "number") {
            display = formatNumber(value);
        }

        item.appendChild(createElement("strong", "", display ?? "N/A"));
        metrics.appendChild(item);
    }

    host.appendChild(metrics);

    host.appendChild(
        createElement(
            "p",
            "muted",
            "This is a benchmark test-set sample. Confidence is not a calibrated "
            + "probability of real-world threat or emitter identity."
        )
    );
}


/* ============================================================
   SIGNAL ANALYSIS — SELECTED SAMPLE
   ============================================================ */

async function loadSavedQualityReport() {
    const host = document.getElementById("savedQualityReport");
    const status = document.getElementById("qualityReportStatus");

    if (!host) return;

    setPanelMessage(host, "Loading saved signal-quality report...");

    try {
        const report = await fetchJSON("/api/signal-quality");

        if (!report || typeof report !== "object") {
            throw new Error("Unexpected signal-quality report format.");
        }

        host.replaceChildren();

        const sample = report.sample || {};
        const classification = report.classification || {};
        const quality = report.quality_assessment || {};
        const signalMetrics = report.signal_metrics || {};
        const recommendation = report.simulation_decision_support || {};

        const summary = createElement("div", "metrics");

        const summaryFields = [
            ["Analyzed test index", sample.test_index],
            ["Original dataset index", sample.original_index],
            ["True modulation", sample.true_class],
            ["Predicted modulation", classification.predicted_class],
            [
                "CNN confidence",
                typeof classification.confidence === "number"
                    ? formatPercent(classification.confidence)
                    : undefined
            ],
            ["Dataset SNR", sample.dataset_snr_db === undefined
                ? undefined
                : `${sample.dataset_snr_db} dB`],
            ["Signal quality", quality.signal_quality],
            ["Classification reliability", quality.classification_reliability]
        ];

        for (const [label, value] of summaryFields) {
            const card = createElement("div", "");
            card.appendChild(createElement("span", "", label));
            card.appendChild(createElement("strong", "", value ?? "Not reported"));
            summary.appendChild(card);
        }

        host.appendChild(summary);

        const metricsPanel = createElement("div", "report-section");
        metricsPanel.appendChild(
            createElement("h3", "panel-title", "Signal measurements")
        );

        appendObjectFields(metricsPanel, signalMetrics);
        host.appendChild(metricsPanel);

        const qualityPanel = createElement("div", "report-section");
        qualityPanel.appendChild(
            createElement("h3", "panel-title", "Quality assessment")
        );

        appendObjectFields(qualityPanel, quality);
        host.appendChild(qualityPanel);

        if (Object.keys(recommendation).length) {
            const recommendationPanel = createElement("div", "report-section");
            recommendationPanel.appendChild(
                createElement("h3", "panel-title", "Simulation recommendation")
            );

            appendObjectFields(recommendationPanel, recommendation);
            host.appendChild(recommendationPanel);
        }

        if (status) {
            setStatus(status, "Saved report loaded", "success");
        }

    } catch (error) {
        setPanelMessage(
            host,
            `Could not load the saved report: ${error.message}`,
            "error"
        );

        if (status) {
            setStatus(status, "Saved report unavailable", "error");
        }
    }
}


/* ============================================================
   DECISION SUPPORT — SAVED DECISION REPORT
   ============================================================ */

async function loadDecisionReport() {
    const host = document.getElementById("decisionResult");
    const status = document.getElementById("decisionStatus");

    if (!host) return;

    setPanelMessage(host, "Loading decision-engine report...");

    try {
        const report = await fetchJSON("/api/decision");

        const classification = report.classification || {};
        const assessment = report.signal_assessment || {};
        const decision = report.decision || {};
        const safety = report.safety_boundary || {};

        host.replaceChildren();

        const summary = createElement("div", "metrics");

        const cards = [
            ["Modulation", classification.modulation],
            ["Modulation family", classification.modulation_family],
            [
                "Model confidence",
                typeof classification.confidence_percent === "number"
                    ? `${classification.confidence_percent.toFixed(2)}%`
                    : undefined
            ],
            ["Signal quality", assessment.signal_quality],
            ["Classification reliability", assessment.classification_reliability],
            ["Confidence state", decision.confidence_state],
            ["Simulation state", decision.simulation_state]
        ];

        for (const [label, value] of cards) {
            const card = createElement("div", "");
            card.appendChild(createElement("span", "", label));
            card.appendChild(createElement("strong", "", value ?? "Not reported"));
            summary.appendChild(card);
        }

        host.appendChild(summary);

        const actionPanel = createElement("div", "decision");
        const actionText = createElement(
            "p",
            "",
            decision.operator_action || "No operator recommendation is present."
        );
        actionPanel.appendChild(actionText);
        host.appendChild(actionPanel);

        const safetyPanel = createElement("div", "report-section");
        safetyPanel.appendChild(
            createElement("h3", "panel-title", "Safety boundaries")
        );
        appendObjectFields(safetyPanel, safety);
        host.appendChild(safetyPanel);

        const metadataPanel = createElement("div", "report-section");
        metadataPanel.appendChild(
            createElement("h3", "panel-title", "Report metadata")
        );

        const metadata = {
            project: report.project,
            decision_engine: report.decision_engine,
            generated_at: report.generated_at
        };

        appendObjectFields(metadataPanel, metadata);
        host.appendChild(metadataPanel);

        const rawDetails = document.getElementById("decisionRaw");
        if (rawDetails) {
            rawDetails.textContent = JSON.stringify(report, null, 2);
        }

        if (status) {
            setStatus(status, "Decision report loaded · Read-only", "success");
        }

    } catch (error) {
        setPanelMessage(
            host,
            `Could not load decision report: ${error.message}`,
            "error"
        );

        if (status) {
            setStatus(status, "Decision report unavailable", "error");
        }
    }
}


/* ============================================================
   GENERIC SAFE REPORT RENDERING
   ============================================================ */

function appendObjectFields(host, object, prefix = "") {
    if (!object || typeof object !== "object") {
        host.appendChild(
            createElement("p", "muted", "No additional fields reported.")
        );
        return;
    }

    const entries = Object.entries(object);

    if (!entries.length) {
        host.appendChild(
            createElement("p", "muted", "No additional fields reported.")
        );
        return;
    }

    const grid = createElement("div", "report-grid");

    for (const [key, value] of entries) {
        const label = prefix ? `${prefix} / ${prettyLabel(key)}` : prettyLabel(key);

        if (value && typeof value === "object" && !Array.isArray(value)) {
            const nested = createElement("div", "report-section");
            nested.appendChild(createElement("h4", "panel-title", label));
            appendObjectFields(nested, value, "");
            grid.appendChild(nested);
            continue;
        }

        let displayValue;

        if (Array.isArray(value)) {
            displayValue = JSON.stringify(value);
        } else if (typeof value === "number") {
            displayValue = formatNumber(value, 6);
        } else if (typeof value === "boolean") {
            displayValue = value ? "Yes" : "No";
        } else {
            displayValue = value === null || value === undefined
                ? "Not reported"
                : String(value);
        }

        const card = createElement("div", "report-field");
        card.appendChild(createElement("span", "report-label", label));
        card.appendChild(createElement("strong", "report-value", displayValue));
        grid.appendChild(card);
    }

    host.appendChild(grid);
}


/* ============================================================
   MANUAL REFRESH HOOKS
   ============================================================ */

async function refreshSignalAnalysis() {
    await Promise.all([
        predictSample(),
        loadSavedQualityReport()
    ]);
}

async function refreshDecisionReport() {
    await loadDecisionReport();
}
