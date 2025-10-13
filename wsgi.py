# wsgi.py
import os
import sys
import importlib.util
from pathlib import Path

# --- DYNAMIC PACKAGE LOADER ---
# This code solves the ModuleNotFoundError for 'hivest' on servers
# like Railway where the project directory may have a unique hash.

try:
    # The directory where this wsgi.py file lives (e.g., /app)
    APP_ROOT = Path(__file__).resolve().parent

    # Find the actual package directory, e.g., /app/hivest-f0704...
    package_dir = None
    for item in APP_ROOT.iterdir():
        # Find a directory that starts with 'hivest-' and contains an '__init__.py'
        if item.is_dir() and item.name.startswith('hivest-') and (item / '__init__.py').exists():
            package_dir = item
            break

    if not package_dir:
        raise FileNotFoundError("Could not find a directory starting with 'hivest-' in the app root.")

    # Trick Python's import system:
    # 1. Add the parent directory of the package to the system path.
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))

    # 2. Dynamically load the found directory ('hivest-f070...') as if it were 'hivest'.
    spec = importlib.util.spec_from_file_location('hivest', package_dir / '__init__.py')

    if not spec or not spec.loader:
        raise ImportError(f"Could not create a module spec for the package at {package_dir}")

    # Create the 'hivest' module in memory
    hivest_module = importlib.util.module_from_spec(spec)

    # Add it to the list of system modules BEFORE executing it.
    # This ensures that any import of 'hivest' inside the package itself will work correctly.
    sys.modules['hivest'] = hivest_module

    # Execute the module to make its contents available.
    spec.loader.exec_module(hivest_module)

    # Now that 'hivest' is a known module, this import will succeed.
    from hivest.api_openrouter import app as application

except (FileNotFoundError, ImportError) as e:
    # Provide a clear error message in the logs if something goes wrong.
    print(f"FATAL: Failed to initialize and load the application. Reason: {e}", file=sys.stderr)
    # Exit with a non-zero status code to make the server crash visibly.
    sys.exit(1)

# The 'application' variable is what Gunicorn looks for by default.