from pathlib import Path
import json

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"

router = APIRouter(tags=["COG-EW Analysis Reports"])

REPORT_FILES = {
    "signal_quality": REPORTS_DIR / "rf_signal_quality_analysis.json",
    "decision_support": REPORTS_DIR / "cogew_decision.json",
}


def read_report(report_name: str) -> dict:
    """Read a known local JSON report without modifying it."""
    path = REPORT_FILES[report_name]

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Report not found: {path}. "
                "Run the corresponding project analysis script first."
            ),
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Report contains invalid JSON: {path.name}: {exc}",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {path.name}: {exc}",
        ) from exc

    if not isinstance(report, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Expected a JSON object in {path.name}.",
        )

    return report


@router.get("/api/signal-quality")
def signal_quality_report():
    """Return the saved RF signal-quality analysis."""
    return read_report("signal_quality")


@router.get("/api/decision-support")
def decision_support_report():
    """Return the saved COG-EW decision-engine report."""
    return read_report("decision_support")


@router.get("/api/analysis-status")
def analysis_status():
    """Report whether both analysis artifacts are present and valid."""
    result = {}

    for name, path in REPORT_FILES.items():
        entry = {
            "exists": path.is_file(),
            "file": path.name,
            "valid_json": False,
        }

        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as file:
                    data = json.load(file)
                entry["valid_json"] = isinstance(data, dict)
            except (OSError, json.JSONDecodeError):
                entry["valid_json"] = False

        result[name] = entry

    return {
        "status": (
            "ok"
            if all(item["valid_json"] for item in result.values())
            else "incomplete"
        ),
        "reports": result,
    }
