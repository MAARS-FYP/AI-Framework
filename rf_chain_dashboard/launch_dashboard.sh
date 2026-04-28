#!/bin/bash
# RF Chain Dashboard Launcher Script

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
RFCHAIN_SOCKET="/tmp/maars_rfchain.sock"
INFERENCE_SOCKET="/tmp/maars_infer.sock"
WS_HOST="127.0.0.1"
WS_PORT=8877
BROWSER_CMD="xdg-open"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --rfchain-socket)
            RFCHAIN_SOCKET="$2"
            shift 2
            ;;
        --inference-socket)
            INFERENCE_SOCKET="$2"
            shift 2
            ;;
        --host)
            WS_HOST="$2"
            shift 2
            ;;
        --port)
            WS_PORT="$2"
            shift 2
            ;;
        --no-browser)
            BROWSER_CMD=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "[RF Chain Dashboard] Starting backend server..."
echo "  RF Chain Socket: $RFCHAIN_SOCKET"
echo "  Inference Socket: $INFERENCE_SOCKET"
echo "  WebSocket Server: ws://$WS_HOST:$WS_PORT"

# Change to project root to ensure imports work
cd "$PROJECT_ROOT"

# Start the backend server
python -m ai_framework.inference.rf_chain_worker \
    --socket-path "$RFCHAIN_SOCKET" &
WORKER_PID=$!

# Wait for worker to be ready
sleep 1

# Start the dashboard backend
python "$SCRIPT_DIR/app.py" \
    --rfchain-socket "$RFCHAIN_SOCKET" \
    --inference-socket "$INFERENCE_SOCKET" \
    --host "$WS_HOST" \
    --port "$WS_PORT" &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 2

# Open browser if command available
if [ -n "$BROWSER_CMD" ]; then
    echo "[RF Chain Dashboard] Opening dashboard in browser..."
    "$BROWSER_CMD" "file://$SCRIPT_DIR/index.html" 2>/dev/null || true
fi

echo "[RF Chain Dashboard] Dashboard available at file://$SCRIPT_DIR/index.html"
echo "[RF Chain Dashboard] Press Ctrl+C to stop"

# Wait for processes
trap "kill $WORKER_PID $BACKEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait
