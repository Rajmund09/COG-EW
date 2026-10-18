"""
======================================================================
COG-EW DASHBOARD API ENTRYPOINT
======================================================================

Exposes the FastAPI application for the COG-EW RF monitoring dashboard.
Can be run via:
    uvicorn dashboard.api:app --host 127.0.0.1 --port 8000
    uvicorn dashboards.api:app --host 127.0.0.1 --port 8000
    python dashboard/api.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dashboard_api import *  # noqa: F401, F403

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.api:app", host="127.0.0.1", port=8000, reload=True)
