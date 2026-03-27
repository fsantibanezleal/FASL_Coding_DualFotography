"""
Standalone executable entry point for FASL_DualFotography.

When frozen by PyInstaller, this script:
1. Sets the working directory to the executable's location
2. Starts the embedded Dash application
3. Opens the default browser at http://127.0.0.1:8004

Usage (development):
    python run_app.py [--port 8004] [--no-browser]

Usage (frozen):
    ./FASL_DualFotography.exe [--port 8004] [--no-browser]
"""
import sys
import os
import argparse
import webbrowser
import threading
from pathlib import Path


def _exe_dir() -> Path:
    """Return the directory containing the executable (frozen) or script (dev)."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="FASL_DualFotography")
    parser.add_argument('--port', type=int, default=8004)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    # Set working directory for stable relative paths
    os.chdir(str(_exe_dir()))

    # Import after CWD is set
    from src.frontend.app import app

    url = f"http://{args.host}:{args.port}"
    print(f"Starting FASL_DualFotography at {url}")

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
