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

# Now import the Flask app from the package
from hivest.api import app as application  # gunicorn expects `application` by default
