# Railway-friendly WSGI bootstrap with no path tricks
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# Import Flask app as `application` for Gunicorn
from api_openrouter import app as application
