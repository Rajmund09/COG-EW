"""
======================================================================
COG-EW DASHBOARD API ENTRYPOINT
======================================================================

Exposes the FastAPI application for the COG-EW RF monitoring dashboard.
Can be run via:
    uvicorn dashboard.api:app --host 127.0.0.1 --port 8000
    uvicorn dashboards.api:app --host 127.0.0.1 --port 8000
"""

from scripts.dashboard_api import *  # noqa: F401, F403

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.api:app", host="127.0.0.1", port=8000, reload=True)
