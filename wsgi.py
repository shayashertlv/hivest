# wsgi.py — minimal, Railway-friendly
import sys, types
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# Create a virtual top-level package named "hivest" that points at the repo root
pkg = types.ModuleType("hivest")
pkg.__path__ = [str(APP_ROOT)]
sys.modules["hivest"] = pkg

from api_openrouter import app as application  # Gunicorn looks for `application`
