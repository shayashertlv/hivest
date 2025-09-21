"""
WSGI entrypoint for running the hivest Flask app on Railway.

This file ensures that when the app root is the `hivest/` directory,
Python can still import the `hivest` package by adding the parent
directory to sys.path.

Gunicorn target: wsgi:application
"""
import os
import sys
from pathlib import Path

# Add the parent directory of this file (project root) to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# Import the Flask app robustly without assuming the top-level package name is known
try:
    # If this module is imported as part of the 'hivest' package
    from .api import app as application  # type: ignore
except Exception:
    # Fallback: import api.py directly by file path (works when running as top-level module)
    import importlib.util
    api_path = CURRENT_DIR / "api.py"
    spec = importlib.util.spec_from_file_location("api", api_path)
    if spec is None or spec.loader is None:
        raise
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    application = api.app  # gunicorn expects `application` by default
