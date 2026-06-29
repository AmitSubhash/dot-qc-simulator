#!/usr/bin/env bash
# Launch the simulator as an interactive app (read-only, no token).
set -euo pipefail
PORT="${1:-8765}"
exec python3 -m marimo run dot_qc_simulator.py --headless --no-token --port "$PORT"
