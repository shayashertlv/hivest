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

# ... (other code)
try:
    # If this module is imported as part of the 'hivest' package
    # CHANGE THIS LINE: point to api_openrouter instead of api
    from .api_openrouter import app as application  # type: ignore
except Exception:
    # Fallback: load api_openrouter.py as 'hivest.api_openrouter'
    import importlib.util
    import types

    pkg_name = "hivest"

    # Ensure the directory containing api_openrouter.py is on sys.path
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))

    # Create a pseudo package 'hivest'
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(CURRENT_DIR)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg

    api_path = CURRENT_DIR / "api_openrouter.py" # CHANGE THIS LINE
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.api_openrouter", api_path) # CHANGE THIS LINE
    if spec is None or spec.loader is None:
        raise
    api = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = api
    spec.loader.exec_module(api)
    application = api.app