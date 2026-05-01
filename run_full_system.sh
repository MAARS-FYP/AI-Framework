#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

MODE="hardware"
IPC_MODE="shm"
SOCKET_PATH="/tmp/maars_infer.sock"
RF_CHAIN_SOCKET_PATH="/tmp/maars_rfchain.sock"
SAMPLE_RATE_HZ="125000000"

SHM_NAME="maars_iq_ring"
SHM_SLOTS="8"
SHM_SLOT_CAPACITY="8192"
WORKER_SHM_CREATE="1"
WORKER_SHM_UNLINK_ON_EXIT="1"

CHECKPOINT_PATH="${ROOT_DIR}/checkpoints/best_model.pt"
SCALERS_PATH="${ROOT_DIR}/checkpoints/scalers.joblib"
WORKER_DEVICE="auto"

PYTHON_BIN_DEFAULT="${ROOT_DIR}/.fyp/bin/python"
PYTHON_BIN_FALLBACK="${ROOT_DIR}/.venv/bin/python"
if [[ -x "${PYTHON_BIN_DEFAULT}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
elif [[ -x "${PYTHON_BIN_FALLBACK}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK}"
else
  PYTHON_BIN="python3"
fi

UART_PORT="/dev/ttyUSB0"
UART_BAUD="115200"
ILA_CSV_PATH="${ROOT_DIR}/ila_probe0.csv"
ILA_REQUEST_FLAG_PATH="${ROOT_DIR}/ila_capture_request.txt"
ILA_POLL_INTERVAL_MS="20"
ILA_REQUEST_TIMEOUT_MS="5000"
ILA_BATCH_SAMPLES="2048"
PRINT_INFERENCE_RESULTS="1"
PLOT_CONSTELLATION="0"

SIMULATE_CYCLES="0"
SIMULATE_INTERVAL_MS="200"
SIMULATE_SAMPLES="4096"
SIMULATE_POWER_LNA="-35.2"
SIMULATE_POWER_PA="-22.8"
RUST_CLEANUP_SHM_ON_EXIT="0"

RF_CHAIN_CYCLES="0"
RF_CHAIN_INTERVAL_MS="100"
RF_CHAIN_ENABLE_INFERENCE="1"

SOCKET_WAIT_TIMEOUT_SEC="30"
WORKER_PID=""
VALON_PID=""
RFCHAIN_WORKER_PID=""
RFCHAIN_DASHBOARD_PID=""
TELEMETRY_DASHBOARD_PID=""
RF_DASHBOARD_HTTP_PID=""
TELEMETRY_SENDER_PID=""

VALON_SOCKET_PATH="/tmp/valon5019.sock"
VALON_PORT=""
VALON_TIMEOUT="1.0"
VALON_BAUD="115200"
VALON_LOG_LEVEL="INFO"
VALON_WAIT_TIMEOUT_SEC="5"
VALON_ENABLED_MODE="auto"
INFERENCE_TXT_PATH="${ROOT_DIR}/inference_results.txt"
DIGITAL_TWIN_PARAMS_PATH="/tmp/maars_digital_twin_params.txt"

usage() {
  cat <<'EOF'
Usage:
  ./run_full_system.sh [options] [-- <extra rust args>]

Options:
  --env-file <path>                  Load deployment variables from env file (default: ./.env if present)
  --mode <hardware|simulate|digital_twin>  Run full hardware loop, simulation mode, or digital twin
  --ipc-mode <direct|shm>            IPC mode for Rust + Python worker
  --socket-path <path>               Unix socket path for inference worker (default: /tmp/maars_infer.sock)
  --rf-chain-socket-path <path>      Unix socket path for RF chain worker (default: /tmp/maars_rfchain.sock)
  --sample-rate-hz <float>           Sample rate used by worker and Rust

  --shm-name <name>                  SHM segment name (default: maars_iq_ring)
  --shm-slots <int>                  SHM slot count (default: 8)
  --shm-slot-capacity <int>          SHM slot capacity (default: 8192)
  --no-worker-shm-create             Do not create SHM in Python worker
  --worker-no-unlink-on-exit         Do not pass --shm-unlink-on-exit to worker

  --checkpoint <path>                Python worker checkpoint path
  --scalers <path>                   Python worker scalers path
  --worker-device <auto|cpu|mps|cuda>

  --uart-port <path>                 Rust UART port (hardware mode)
  --uart-baud <int>                  Rust UART baud (hardware mode)
  --ila-csv-path <path>              Rust ILA probe0 CSV path (hardware mode)
  --ila-request-flag-path <path>     Rust ILA capture request flag file (hardware mode)
  --ila-poll-interval-ms <int>       Rust ILA poll interval in ms (hardware mode)
  --ila-request-timeout-ms <int>     Rust ILA request timeout in ms (hardware mode)
  --ila-batch-samples <int>          Rust probe0 rows consumed per inference (hardware mode)

  --simulate-cycles <int>            Simulation cycles (0 = continuous)
  --simulate-interval-ms <int>       Delay between simulation cycles
  --simulate-samples <int>           Simulation IQ samples per cycle
  --simulate-power-lna <float>       Simulation LNA power dBm
  --simulate-power-pa <float>        Simulation PA power dBm

  --rf-chain-cycles <int>            Digital twin cycles (0 = continuous, digital_twin mode)
  --rf-chain-interval-ms <int>       Delay between RF chain calls (digital_twin mode)
  --rf-chain-enable-inference        Enable inference worker chaining in digital_twin mode
  --rust-cleanup-shm-on-exit         Pass --cleanup-shm-on-exit to Rust
  --print-inference-results          Print inference summaries from Rust (default: on)
  --no-print-inference-results       Disable inference summaries from Rust
  --enable-constellation-plot        Launch the Rust-managed constellation plotter
  --disable-constellation-plot       Disable the Rust-managed constellation plotter
  --enable-valon                     Force-enable Valon worker and Rust Valon output
  --disable-valon                    Force-disable Valon worker and Rust Valon output
  --valon-socket-path <path>         Valon worker socket path (default: /tmp/valon5019.sock)
  --valon-port <path>                Explicit Valon serial port (optional)
  --valon-timeout <float>            Valon serial timeout seconds (default: 1.0)
  --valon-baud <int>                 Valon baud rate (default: 115200)
  --valon-log-level <level>          Valon worker log level (default: INFO)
  --valon-wait-timeout <int>         Valon socket wait timeout seconds (default: 5)
  --inference-txt-path <path>        Write latest inference snapshot to text file (default: ./inference_results.txt)
  --python-bin <path>                Override Python executable
  --help                             Show this help

Examples:
  ./run_full_system.sh --mode hardware --ipc-mode shm
  ./run_full_system.sh --mode simulate --ipc-mode shm --simulate-cycles 10
  ./run_full_system.sh --mode hardware --ipc-mode direct -- --sample-rate-hz 25000000
EOF
}

load_env_file() {
  local env_file="$1"
  if [[ -f "${env_file}" ]]; then
    echo "[launcher] loading env file: ${env_file}"
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
}

# First pass: allow --env-file to be set before loading variables.
ARGS=("$@")
idx=0
while [[ ${idx} -lt ${#ARGS[@]} ]]; do
  if [[ "${ARGS[$idx]}" == "--env-file" ]]; then
    if (( idx + 1 >= ${#ARGS[@]} )); then
      echo "Missing value for --env-file" >&2
      exit 2
    fi
    ENV_FILE="${ARGS[$((idx + 1))]}"
    break
  fi
  ((idx += 1))
done

load_env_file "${ENV_FILE}"

# --- PURGE ORPHANS ---
echo "[launcher] cleaning up old processes..."
pkill -9 -f software-framework || true
pkill -9 -f ai_framework.inference.worker || true
pkill -9 -f ai_framework.inference.rf_chain_worker || true
pkill -9 -f udp_ws_bridge.py || true
pkill -9 -f udp_telemetry_sender.py || true
pkill -9 -f "python3 -m http.server 8080" || true
rm -f /tmp/maars_*.sock /tmp/maars_digital_twin_params.txt || true
sleep 0.5

EXTRA_RUST_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --ipc-mode) IPC_MODE="$2"; shift 2 ;;
    --socket-path) SOCKET_PATH="$2"; shift 2 ;;
    --rf-chain-socket-path) RF_CHAIN_SOCKET_PATH="$2"; shift 2 ;;
    --sample-rate-hz) SAMPLE_RATE_HZ="$2"; shift 2 ;;
    --shm-name) SHM_NAME="$2"; shift 2 ;;
    --shm-slots) SHM_SLOTS="$2"; shift 2 ;;
    --shm-slot-capacity) SHM_SLOT_CAPACITY="$2"; shift 2 ;;
    --no-worker-shm-create) WORKER_SHM_CREATE="0"; shift ;;
    --worker-no-unlink-on-exit) WORKER_SHM_UNLINK_ON_EXIT="0"; shift ;;
    --checkpoint) CHECKPOINT_PATH="$2"; shift 2 ;;
    --scalers) SCALERS_PATH="$2"; shift 2 ;;
    --worker-device) WORKER_DEVICE="$2"; shift 2 ;;
    --uart-port) UART_PORT="$2"; shift 2 ;;
    --uart-baud) UART_BAUD="$2"; shift 2 ;;
    --ila-csv-path) ILA_CSV_PATH="$2"; shift 2 ;;
    --ila-request-flag-path) ILA_REQUEST_FLAG_PATH="$2"; shift 2 ;;
    --ila-poll-interval-ms) ILA_POLL_INTERVAL_MS="$2"; shift 2 ;;
    --ila-request-timeout-ms) ILA_REQUEST_TIMEOUT_MS="$2"; shift 2 ;;
    --ila-batch-samples) ILA_BATCH_SAMPLES="$2"; shift 2 ;;
    --print-inference-results) PRINT_INFERENCE_RESULTS="1"; shift ;;
    --no-print-inference-results) PRINT_INFERENCE_RESULTS="0"; shift ;;
    --enable-constellation-plot) PLOT_CONSTELLATION="1"; shift ;;
    --disable-constellation-plot) PLOT_CONSTELLATION="0"; shift ;;
    --simulate-cycles) SIMULATE_CYCLES="$2"; shift 2 ;;
    --simulate-interval-ms) SIMULATE_INTERVAL_MS="$2"; shift 2 ;;
    --simulate-samples) SIMULATE_SAMPLES="$2"; shift 2 ;;
    --simulate-power-lna) SIMULATE_POWER_LNA="$2"; shift 2 ;;
    --simulate-power-pa) SIMULATE_POWER_PA="$2"; shift 2 ;;
    --rf-chain-cycles) RF_CHAIN_CYCLES="$2"; shift 2 ;;
    --rf-chain-interval-ms) RF_CHAIN_INTERVAL_MS="$2"; shift 2 ;;
    --rf-chain-enable-inference) RF_CHAIN_ENABLE_INFERENCE="1"; shift ;;
    --rust-cleanup-shm-on-exit) RUST_CLEANUP_SHM_ON_EXIT="1"; shift ;;
    --enable-valon) VALON_ENABLED_MODE="1"; shift ;;
    --disable-valon) VALON_ENABLED_MODE="0"; shift ;;
    --valon-socket-path) VALON_SOCKET_PATH="$2"; shift 2 ;;
    --valon-port) VALON_PORT="$2"; shift 2 ;;
    --valon-timeout) VALON_TIMEOUT="$2"; shift 2 ;;
    --valon-baud) VALON_BAUD="$2"; shift 2 ;;
    --valon-log-level) VALON_LOG_LEVEL="$2"; shift 2 ;;
    --valon-wait-timeout) VALON_WAIT_TIMEOUT_SEC="$2"; shift 2 ;;
    --inference-txt-path) INFERENCE_TXT_PATH="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_RUST_ARGS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "${MODE}" != "hardware" && "${MODE}" != "simulate" && "${MODE}" != "digital_twin" ]]; then
  echo "Invalid --mode: ${MODE}" >&2
  exit 2
fi

if [[ "${IPC_MODE}" != "direct" && "${IPC_MODE}" != "shm" ]]; then
  echo "Invalid --ipc-mode: ${IPC_MODE}" >&2
  exit 2
fi

is_valon_enabled() {
  case "${VALON_ENABLED_MODE}" in
    1) return 0 ;;
    0) return 1 ;;
    *)
      if [[ "${MODE}" == "hardware" ]]; then
        return 0
      fi
      return 1
      ;;
  esac
}

cleanup() {
  set +e
  if [[ -n "${VALON_PID}" ]] && kill -0 "${VALON_PID}" 2>/dev/null; then
    kill "${VALON_PID}" >/dev/null 2>&1 || true
    sleep 0.2
    kill -9 "${VALON_PID}" >/dev/null 2>&1 || true
  fi
  rm -f "${VALON_SOCKET_PATH}" >/dev/null 2>&1 || true

  if [[ -n "${WORKER_PID}" ]] && kill -0 "${WORKER_PID}" 2>/dev/null; then
    "${PYTHON_BIN}" -m ai_framework.cli.inference_socket_client --socket-path "${SOCKET_PATH}" --shutdown >/dev/null 2>&1 || true
    sleep 0.3
    kill "${WORKER_PID}" >/dev/null 2>&1 || true
    sleep 0.2
    kill -9 "${WORKER_PID}" >/dev/null 2>&1 || true
  fi
  rm -f "${SOCKET_PATH}" >/dev/null 2>&1 || true

  if [[ -n "${RFCHAIN_DASHBOARD_PID}" ]] && kill -0 "${RFCHAIN_DASHBOARD_PID}" 2>/dev/null; then
    kill "${RFCHAIN_DASHBOARD_PID}" >/dev/null 2>&1 || true
    sleep 0.2
    kill -9 "${RFCHAIN_DASHBOARD_PID}" >/dev/null 2>&1 || true
  fi

  if [[ -n "${TELEMETRY_DASHBOARD_PID}" ]] && kill -0 "${TELEMETRY_DASHBOARD_PID}" 2>/dev/null; then
    kill "${TELEMETRY_DASHBOARD_PID}" >/dev/null 2>&1 || true
    sleep 0.2
    kill -9 "${TELEMETRY_DASHBOARD_PID}" >/dev/null 2>&1 || true
  fi

  if [[ -n "${TELEMETRY_SENDER_PID}" ]] && kill -0 "${TELEMETRY_SENDER_PID}" 2>/dev/null; then
    kill "${TELEMETRY_SENDER_PID}" >/dev/null 2>&1 || true
    sleep 0.2
    kill -9 "${TELEMETRY_SENDER_PID}" >/dev/null 2>&1 || true
  fi

  if [[ -n "${RF_DASHBOARD_HTTP_PID}" ]] && kill -0 "${RF_DASHBOARD_HTTP_PID}" 2>/dev/null; then
    kill "${RF_DASHBOARD_HTTP_PID}" >/dev/null 2>&1 || true
    sleep 0.2
    kill -9 "${RF_DASHBOARD_HTTP_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

start_valon_worker() {
  if ! is_valon_enabled; then
    echo "[launcher] valon worker disabled"
    return
  fi

  rm -f "${VALON_SOCKET_PATH}" >/dev/null 2>&1 || true

  local valon_args=(
    valon_worker.py
    --socket "${VALON_SOCKET_PATH}"
    --timeout "${VALON_TIMEOUT}"
    --baud "${VALON_BAUD}"
    --log-level "${VALON_LOG_LEVEL}"
  )

  if [[ -n "${VALON_PORT}" ]]; then
    valon_args+=(--port "${VALON_PORT}")
  fi

  echo "[launcher] starting Valon worker..."
  (
    cd "${ROOT_DIR}/valon_controller"
    "${PYTHON_BIN}" "${valon_args[@]}"
  ) &
  VALON_PID=$!
  echo "[launcher] valon pid=${VALON_PID}"
}

wait_for_valon_socket() {
  if ! is_valon_enabled; then
    return 0
  fi

  local waited=0
  while [[ ! -S "${VALON_SOCKET_PATH}" ]]; do
    sleep 0.1
    waited=$((waited + 1))
    if (( waited >= VALON_WAIT_TIMEOUT_SEC * 10 )); then
      echo "[launcher] valon socket not ready at ${VALON_SOCKET_PATH} after ${VALON_WAIT_TIMEOUT_SEC}s" >&2
      return 1
    fi
  done

  echo "[launcher] valon socket ready: ${VALON_SOCKET_PATH}"
}

start_rfchain_worker() {
  local rfchain_socket="${1:-/tmp/maars_rfchain.sock}"
  
  rm -f "${rfchain_socket}" >/dev/null 2>&1 || true
  
  echo "[launcher] starting Python RF chain worker..."
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -m ai_framework.inference.rf_chain_worker \
      --socket-path "${rfchain_socket}"
  ) &
  RFCHAIN_WORKER_PID=$!
  echo "[launcher] RF chain worker pid=${RFCHAIN_WORKER_PID}"
}

wait_for_rfchain_socket() {
  local socket="$1"
  local waited=0
  while [[ ! -S "${socket}" ]]; do
    sleep 0.2
    waited=$((waited + 1))
    if (( waited >= 30 * 5 )); then
      echo "[launcher] RF chain socket not ready at ${socket} after 30s" >&2
      return 1
    fi
  done
  echo "[launcher] RF chain socket ready: ${socket}"
}

start_rfchain_dashboard() {
  echo "[launcher] starting RF chain dashboard backend..."
  (
    cd "${ROOT_DIR}/rf_chain_dashboard"
    # Ensure Python deps for the dashboard are installed in the chosen Python
    if ! "${PYTHON_BIN}" -c "import websockets" >/dev/null 2>&1; then
      echo "[launcher] installing rf_chain_dashboard Python requirements..."
      "${PYTHON_BIN}" -m pip install --upgrade pip >/dev/null 2>&1 || true
      "${PYTHON_BIN}" -m pip install -r requirements.txt >/dev/null 2>&1 || true
    fi
    "${PYTHON_BIN}" app.py \
      --rfchain-socket "${RF_CHAIN_SOCKET_PATH}" \
      --inference-socket "${SOCKET_PATH}" \
      --params-path "${DIGITAL_TWIN_PARAMS_PATH}" \
      --host 127.0.0.1 \
      --port 8877
  ) >/tmp/rfchain_dashboard.log 2>&1 &
  RFCHAIN_DASHBOARD_PID=$!
  echo "[launcher] RF chain dashboard pid=${RFCHAIN_DASHBOARD_PID}"
}

wait_for_rfchain_dashboard() {
  local waited=0
  local port=8877
  echo "[launcher] waiting for RF chain dashboard to be ready on port ${port}..."
  # Portable port probe using Python (works on macOS and Linux)
  while ! "${PYTHON_BIN}" - <<PYTHON
import socket,sys
try:
    s=socket.create_connection(('127.0.0.1', ${port}), timeout=1)
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PYTHON
  do
    sleep 0.2
    waited=$((waited + 1))
    if (( waited >= 50 )); then
      echo "[launcher] RF chain dashboard not ready after 10s" >&2
      echo "[launcher] Check /tmp/rfchain_dashboard.log for errors" >&2
      return 1
    fi
  done
  echo "[launcher] RF chain dashboard ready on port ${port}"
}

start_telemetry_dashboard() {
  echo "[launcher] starting telemetry dashboard backend..."
  (
    cd "${ROOT_DIR}/Dashboard-main"
    "${PYTHON_BIN}" -m http.server 8080 >/tmp/telemetry_dashboard_http.log 2>&1 &
    HTTP_PID=$!
    RF_DASHBOARD_HTTP_PID=$HTTP_PID
    "${PYTHON_BIN}" udp_ws_bridge.py \
      --udp-host 127.0.0.1 \
      --udp-port 9000 \
      --ws-host 127.0.0.1 \
      --ws-port 8765 \
      --ws-path /telemetry \
      >/tmp/telemetry_dashboard_bridge.log 2>&1 &
    TELEMETRY_DASHBOARD_PID=$!

    "${PYTHON_BIN}" udp_telemetry_sender.py \
      --host 127.0.0.1 \
      --port 9000 \
      --hz 15 \
      --source-file "${ROOT_DIR}/inference_results.txt" \
      >/tmp/telemetry_dashboard_sender.log 2>&1 &
    TELEMETRY_SENDER_PID=$!
    wait
  ) &
  echo "[launcher] telemetry dashboard started"
  sleep 1
}

open_dashboard_browsers() {
  local open_cmd="xdg-open"
  if command -v open &>/dev/null; then
    open_cmd="open"
  fi
  
  sleep 2
  echo "[launcher] opening RF chain dashboard in browser..."
  ${open_cmd} "file://${ROOT_DIR}/rf_chain_dashboard/index.html" >/dev/null 2>&1 || true
  
  echo "[launcher] opening telemetry dashboard in browser..."
  ${open_cmd} "http://127.0.0.1:8080/index.html?ws=ws://127.0.0.1:8765/telemetry" >/dev/null 2>&1 || true
  
  echo "[launcher] dashboards should now be open in your browser"
}

start_worker() {
  local worker_args=(
    -m ai_framework.inference.worker
    --socket-path "${SOCKET_PATH}"
    --checkpoint "${CHECKPOINT_PATH}"
    --scalers "${SCALERS_PATH}"
    --device "${WORKER_DEVICE}"
    --sample-rate-hz "${SAMPLE_RATE_HZ}"
    --allow-center-shift
  )

  if [[ "${IPC_MODE}" == "shm" ]]; then
    worker_args+=(
      --shm-name "${SHM_NAME}"
      --shm-slots "${SHM_SLOTS}"
      --shm-slot-capacity "${SHM_SLOT_CAPACITY}"
    )
    if [[ "${WORKER_SHM_CREATE}" == "1" ]]; then
      worker_args+=(--shm-create)
    fi
    if [[ "${WORKER_SHM_UNLINK_ON_EXIT}" == "1" ]]; then
      worker_args+=(--shm-unlink-on-exit)
    fi
  fi

  rm -f "${SOCKET_PATH}" >/dev/null 2>&1 || true

  echo "[launcher] starting Python inference worker..."
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" "${worker_args[@]}"
  ) &
  WORKER_PID=$!
  echo "[launcher] inference worker pid=${WORKER_PID}"
}

wait_for_socket() {
  local waited=0
  while [[ ! -S "${SOCKET_PATH}" ]]; do
    sleep 0.2
    waited=$((waited + 1))
    if (( waited >= SOCKET_WAIT_TIMEOUT_SEC * 5 )); then
      echo "[launcher] socket not ready at ${SOCKET_PATH} after ${SOCKET_WAIT_TIMEOUT_SEC}s" >&2
      return 1
    fi
  done
  echo "[launcher] socket ready: ${SOCKET_PATH}"
}

run_rust() {
  local rust_args=(
    --mode "${MODE}"
    --ipc-mode "${IPC_MODE}"
    --socket-path "${SOCKET_PATH}"
    --rf-chain-socket-path "${RF_CHAIN_SOCKET_PATH}"
    --sample-rate-hz "${SAMPLE_RATE_HZ}"
  )

  if is_valon_enabled; then
    rust_args+=(
      --enable-valon
      --valon-socket-path "${VALON_SOCKET_PATH}"
    )
  else
    rust_args+=(--disable-valon)
  fi

  if [[ "${IPC_MODE}" == "shm" ]]; then
    rust_args+=(
      --shm-name "${SHM_NAME}"
      --shm-slots "${SHM_SLOTS}"
      --shm-slot-capacity "${SHM_SLOT_CAPACITY}"
    )
  fi

  if [[ "${MODE}" == "simulate" ]]; then
    rust_args+=(
      --simulate
      --simulate-cycles "${SIMULATE_CYCLES}"
      --simulate-interval-ms "${SIMULATE_INTERVAL_MS}"
      --dry-run-samples "${SIMULATE_SAMPLES}"
      --dry-run-power-lna "${SIMULATE_POWER_LNA}"
      --dry-run-power-pa "${SIMULATE_POWER_PA}"
    )
  elif [[ "${MODE}" == "digital_twin" ]]; then
    echo "[launcher] starting in digital_twin mode"
    rust_args+=(
      --rf-chain-cycles "${RF_CHAIN_CYCLES}"
      --rf-chain-interval-ms "${RF_CHAIN_INTERVAL_MS}"
      --digital-twin-params-path "${DIGITAL_TWIN_PARAMS_PATH}"
    )
    if [[ "${RF_CHAIN_ENABLE_INFERENCE}" == "1" ]]; then
      rust_args+=(--enable-inference)
    else
      rust_args+=(--disable-inference)
    fi
  else
    rust_args+=(
      --uart-port "${UART_PORT}"
      --uart-baud "${UART_BAUD}"
      --ila-csv-path "${ILA_CSV_PATH}"
      --ila-request-flag-path "${ILA_REQUEST_FLAG_PATH}"
      --ila-poll-interval-ms "${ILA_POLL_INTERVAL_MS}"
      --ila-request-timeout-ms "${ILA_REQUEST_TIMEOUT_MS}"
      --ila-batch-samples "${ILA_BATCH_SAMPLES}"
    )
  fi

  rust_args+=(--inference-txt-path "${INFERENCE_TXT_PATH}")

  if [[ "${PLOT_CONSTELLATION}" == "1" ]]; then
    rust_args+=(
      --enable-constellation-plot
      --plot-python-bin "${PYTHON_BIN}"
      --plot-script-path "${ROOT_DIR}/pluto_live_plot.py"
      --plot-slot-index-path "/tmp/maars_iq_ring_slot.txt"
    )
  else
    rust_args+=(--disable-constellation-plot)
  fi

  if [[ "${PRINT_INFERENCE_RESULTS}" == "1" ]]; then
    rust_args+=(--print-inference-results)
  else
    rust_args+=(--no-print-inference-results)
  fi

  if [[ "${RUST_CLEANUP_SHM_ON_EXIT}" == "1" ]]; then
    rust_args+=(--cleanup-shm-on-exit)
  fi

  if (( ${#EXTRA_RUST_ARGS[@]} > 0 )); then
    rust_args+=("${EXTRA_RUST_ARGS[@]}")
  fi

  echo "[launcher] running Rust (${MODE}, ${IPC_MODE})..."
  (
    cd "${ROOT_DIR}/software_framework"
    cargo run --release -- "${rust_args[@]}"
  )
}

start_worker
if ! wait_for_socket; then
  echo "[launcher] inference worker startup failed" >&2
  exit 1
fi
start_valon_worker
if ! wait_for_valon_socket; then
  echo "[launcher] valon startup failed" >&2
  exit 1
fi

# Start RF chain worker and dashboards if in digital_twin mode
if [[ "${MODE}" == "digital_twin" ]]; then
  start_rfchain_worker "${RF_CHAIN_SOCKET_PATH}"
  if ! wait_for_rfchain_socket "${RF_CHAIN_SOCKET_PATH}"; then
    echo "[launcher] RF chain worker startup failed" >&2
    exit 1
  fi
  
  # Start dashboards
  start_rfchain_dashboard
  if ! wait_for_rfchain_dashboard; then
    echo "[launcher] RF chain dashboard startup failed" >&2
    exit 1
  fi
  
  start_telemetry_dashboard
  open_dashboard_browsers
fi

run_rust
