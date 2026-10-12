"""Compatibility alias for dashboard.api."""
from dashboard.api import *  # noqa: F401, F403

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboards.api:app", host="127.0.0.1", port=8000, reload=True)
