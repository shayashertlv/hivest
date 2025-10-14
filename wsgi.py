# wsgi.py — Railway-friendly bootstrap that exposes a virtual "hivest" package
import sys, types
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# Create a virtual package so imports like `from hivest.x import ...` work
hivest_pkg = types.ModuleType("hivest")
hivest_pkg.__path__ = [str(APP_ROOT)]  # treat repo root as the hivest package path
sys.modules["hivest"] = hivest_pkg

# Gunicorn entrypoint
from api_openrouter import app as application
