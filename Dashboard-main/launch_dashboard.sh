#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN_DEFAULT="$PROJECT_ROOT/.fyp/bin/python"
PYTHON_BIN_FALLBACK="$PROJECT_ROOT/.venv/bin/python"
if [[ -x "$PYTHON_BIN_DEFAULT" ]]; then
	PYTHON_BIN="$PYTHON_BIN_DEFAULT"
elif [[ -x "$PYTHON_BIN_FALLBACK" ]]; then
	PYTHON_BIN="$PYTHON_BIN_FALLBACK"
else
	PYTHON_BIN="python3"
fi

DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"
UDP_HOST="${UDP_HOST:-127.0.0.1}"
UDP_PORT="${UDP_PORT:-9000}"
WS_HOST="${WS_HOST:-127.0.0.1}"
WS_PORT="${WS_PORT:-8765}"
WS_PATH="${WS_PATH:-/telemetry}"

cd "$SCRIPT_DIR"

"$PYTHON_BIN" -m http.server "$DASHBOARD_PORT" >/tmp/rf_dashboard_http.log 2>&1 &
HTTP_PID=$!

"$PYTHON_BIN" udp_ws_bridge.py \
	--udp-host "$UDP_HOST" \
	--udp-port "$UDP_PORT" \
	--ws-host "$WS_HOST" \
	--ws-port "$WS_PORT" \
	--ws-path "$WS_PATH" \
	>/tmp/rf_dashboard_bridge.log 2>&1 &
BRIDGE_PID=$!

"$PYTHON_BIN" udp_telemetry_sender.py \
	--host "$UDP_HOST" \
	--port "$UDP_PORT" \
	--hz 15 \
	--source-file "$PROJECT_ROOT/inference_results.txt" \
	>/tmp/rf_dashboard_sender.log 2>&1 &
SENDER_PID=$!

trap 'kill "$HTTP_PID" "$BRIDGE_PID" "$SENDER_PID" 2>/dev/null || true' INT TERM EXIT

echo "[RF Chain Dashboard] HTTP: http://127.0.0.1:${DASHBOARD_PORT}/index.html?ws=ws://${WS_HOST}:${WS_PORT}${WS_PATH}"
echo "[RF Chain Dashboard] Press Ctrl+C to stop"

sleep 1

if command -v xdg-open >/dev/null 2>&1; then
	xdg-open "http://127.0.0.1:${DASHBOARD_PORT}/index.html?ws=ws://${WS_HOST}:${WS_PORT}${WS_PATH}" >/dev/null 2>&1 || true
fi

wait
cd "$(dirname "$0")" && .venv/bin/python -m http.server "${DASHBOARD_PORT:-8080}" >/tmp/rf_dashboard_http.log 2>&1 & .venv/bin/python udp_ws_bridge.py --udp-host "${UDP_HOST:-127.0.0.1}" --udp-port "${UDP_PORT:-9000}" --ws-host "${WS_HOST:-127.0.0.1}" --ws-port "${WS_PORT:-8765}" --ws-path "${WS_PATH:-/telemetry}" >/tmp/rf_dashboard_bridge.log 2>&1 & sleep 1 && open "http://127.0.0.1:${DASHBOARD_PORT:-8080}/index.html?ws=ws://127.0.0.1:${WS_PORT:-8765}${WS_PATH:-/telemetry}" && wait
