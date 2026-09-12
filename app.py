"""
PULMO·AI — Streamlit Community Cloud entrypoint.

Streamlit Cloud looks for app_file: app.py at the repo root.
This simply executes the full app from src/ui/streamlit_app.py.
"""
import sys
import runpy
from pathlib import Path

# Ensure project root is on sys.path so all src.* imports resolve correctly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

runpy.run_path(str(PROJECT_ROOT / "src" / "ui" / "streamlit_app.py"), run_name="__main__")
