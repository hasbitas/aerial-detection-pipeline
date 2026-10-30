#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
trap 'echo "[Setup] Setup failed. Read the error above, fix it, and run ./setup.sh again." >&2' ERR

echo "[Setup] Checking Python 3.11 or newer..."
if [[ -x .venv/bin/python ]]; then
    python_cmd=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    python_cmd=python3
elif command -v python >/dev/null 2>&1; then
    python_cmd=python
else
    echo "[Setup] Install Python 3.11+ from https://www.python.org/downloads/ and try again." >&2
    exit 1
fi
if ! "$python_cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "[Setup] Python 3.11+ is required. Install it, then rerun setup." >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "[Setup] Creating .venv..."
    "$python_cmd" -m venv .venv
fi
echo "[Setup] Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip
echo "[Setup] Installing project and browser demo packages. Downloads may take a while..."
.venv/bin/python -m pip install -r requirements.txt -r demo/requirements.txt
echo "[Setup] Setup complete. Next, run ./start.sh."
