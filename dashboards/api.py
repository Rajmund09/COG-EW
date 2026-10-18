"""Compatibility alias for dashboard.api."""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.api import *  # noqa: F401, F403

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboards.api:app", host="127.0.0.1", port=8000, reload=True)
