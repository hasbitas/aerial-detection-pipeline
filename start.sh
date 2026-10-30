#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    echo "[Start] Setup is missing. Run ./setup.sh first." >&2
    exit 1
fi
export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_SERVER_PORT=7860
echo "[Start] Starting the demo at http://127.0.0.1:7860"
echo "[Start] Keep this terminal open. Press Ctrl+C to stop."
echo "[Start] Model weights download on the first Analyze click if missing."
if .venv/bin/python -u demo/app.py --device cpu --open-browser; then
    exit 0
else
    result=$?
    echo "[Start] The demo stopped. Read any error above; check setup and port 7860." >&2
    exit "$result"
fi
