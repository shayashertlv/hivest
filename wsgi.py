# wsgi.py
import os
import sys
from pathlib import Path

# Add the project's root directory to the Python path.
# This ensures that the 'hivest' package can be imported.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now, we can directly import the app from the hivest package
from hivest.api_openrouter import app as application

# The 'application' variable is what Gunicorn looks for by default.