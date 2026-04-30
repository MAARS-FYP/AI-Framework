#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

MODE="hardware"
REDUCED_SIMULATE="0"
IPC_MODE="shm"
SOCKET_PATH="/tmp/maars_infer.sock"
SAMPLE_RATE_HZ="25000000"
REDUCED_SOCKET_PATH="/tmp/maars_reduced_hw.sock"
REDUCED_CAPTURE_PATH="${ROOT_DIR}/ila_capture.csv"
REDUCED_CHECKPOINT_PATH="${ROOT_DIR}/checkpoints/reduced_hardware/reduced_hardware_fftnet.pt"

SHM_NAME="maars_iq_ring"
SHM_SLOTS="8"
SHM_SLOT_CAPACITY="8192"
WORKER_SHM_CREATE="1"
WORKER_SHM_UNLINK_ON_EXIT="1"

CHECKPOINT_PATH="${ROOT_DIR}/checkpoints/best_model.pt"
SCALERS_PATH="${ROOT_DIR}/checkpoints/scalers.joblib"
WORKER_DEVICE="auto"

PYTHON_BIN_DEFAULT="${ROOT_DIR}/.venv/bin/python"
if [[ -x "${PYTHON_BIN_DEFAULT}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
else
  PYTHON_BIN="python3"
fi

UART_PORT="/dev/ttyUSB0"
UART_BAUD="115200"
UDP_BIND="127.0.0.1:5000"

SIMULATE_CYCLES="0"
SIMULATE_INTERVAL_MS="200"
SIMULATE_SAMPLES="4096"
SIMULATE_POWER_LNA="-35.2"
SIMULATE_POWER_PA="-22.8"
RUST_CLEANUP_SHM_ON_EXIT="0"

SOCKET_WAIT_TIMEOUT_SEC="30"
WORKER_PID=""
VALON_PID=""

VALON_SOCKET_PATH="/tmp/valon5019.sock"
VALON_PORT=""
VALON_TIMEOUT="1.0"
VALON_BAUD="115200"
VALON_LOG_LEVEL="INFO"
VALON_WAIT_TIMEOUT_SEC="5"
VALON_ENABLED_MODE="auto"

usage() {
  cat <<'EOF'
Usage:
  ./run_full_system.sh [options] [-- <extra rust args>]

Options:
  --env-file <path>                  Load deployment variables from env file (default: ./.env if present)
  --mode <hardware|simulate|reduced-hardware>
                                     Run full hardware loop, simulation mode, or reduced-hardware mode
  --reduced-simulate                Run reduced-hardware mode with synthetic ADC samples only
  --ipc-mode <direct|shm>            IPC mode for Rust + Python worker
  --socket-path <path>               Unix socket path (default: /tmp/maars_infer.sock)
  --sample-rate-hz <float>           Sample rate used by worker and Rust
  --reduced-socket-path <path>       Reduced-hardware worker socket path (default: /tmp/maars_reduced_hw.sock)
  --reduced-capture-path <path>      ADC capture CSV path used by reduced-hardware mode (default: ./ila_capture.csv)

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
  --udp-bind <host:port>             Rust UDP bind (hardware mode)

  --simulate-cycles <int>            Simulation cycles (0 = continuous)
  --simulate-interval-ms <int>       Delay between simulation cycles
  --simulate-samples <int>           Simulation IQ samples per cycle
  --simulate-power-lna <float>       Simulation LNA power dBm
  --simulate-power-pa <float>        Simulation PA power dBm

  --rust-cleanup-shm-on-exit         Pass --cleanup-shm-on-exit to Rust
  --enable-valon                     Force-enable Valon worker and Rust Valon output
  --disable-valon                    Force-disable Valon worker and Rust Valon output
  --valon-socket-path <path>         Valon worker socket path (default: /tmp/valon5019.sock)
  --valon-port <path>                Explicit Valon serial port (optional)
  --valon-timeout <float>            Valon serial timeout seconds (default: 1.0)
  --valon-baud <int>                 Valon baud rate (default: 115200)
  --valon-log-level <level>          Valon worker log level (default: INFO)
  --valon-wait-timeout <int>         Valon socket wait timeout seconds (default: 5)
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

EXTRA_RUST_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --reduced-simulate) REDUCED_SIMULATE="1"; shift ;;
    --ipc-mode) IPC_MODE="$2"; shift 2 ;;
    --socket-path) SOCKET_PATH="$2"; shift 2 ;;
    --sample-rate-hz) SAMPLE_RATE_HZ="$2"; shift 2 ;;
    --reduced-socket-path) REDUCED_SOCKET_PATH="$2"; shift 2 ;;
    --reduced-capture-path) REDUCED_CAPTURE_PATH="$2"; shift 2 ;;
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
    --udp-bind) UDP_BIND="$2"; shift 2 ;;
    --simulate-cycles) SIMULATE_CYCLES="$2"; shift 2 ;;
    --simulate-interval-ms) SIMULATE_INTERVAL_MS="$2"; shift 2 ;;
    --simulate-samples) SIMULATE_SAMPLES="$2"; shift 2 ;;
    --simulate-power-lna) SIMULATE_POWER_LNA="$2"; shift 2 ;;
    --simulate-power-pa) SIMULATE_POWER_PA="$2"; shift 2 ;;
    --rust-cleanup-shm-on-exit) RUST_CLEANUP_SHM_ON_EXIT="1"; shift ;;
    --enable-valon) VALON_ENABLED_MODE="1"; shift ;;
    --disable-valon) VALON_ENABLED_MODE="0"; shift ;;
    --valon-socket-path) VALON_SOCKET_PATH="$2"; shift 2 ;;
    --valon-port) VALON_PORT="$2"; shift 2 ;;
    --valon-timeout) VALON_TIMEOUT="$2"; shift 2 ;;
    --valon-baud) VALON_BAUD="$2"; shift 2 ;;
    --valon-log-level) VALON_LOG_LEVEL="$2"; shift 2 ;;
    --valon-wait-timeout) VALON_WAIT_TIMEOUT_SEC="$2"; shift 2 ;;
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

if [[ "${MODE}" != "hardware" && "${MODE}" != "simulate" && "${MODE}" != "reduced-hardware" ]]; then
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
      if [[ "${MODE}" == "hardware" || "${MODE}" == "reduced-hardware" ]]; then
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

  local worker_socket_path="${SOCKET_PATH}"
  if [[ "${MODE}" == "reduced-hardware" ]]; then
    worker_socket_path="${REDUCED_SOCKET_PATH}"
  fi

  if [[ -n "${WORKER_PID}" ]] && kill -0 "${WORKER_PID}" 2>/dev/null; then
    if [[ "${MODE}" != "reduced-hardware" ]]; then
      "${PYTHON_BIN}" -m ai_framework.cli.inference_socket_client --socket-path "${worker_socket_path}" --shutdown >/dev/null 2>&1 || true
    fi
    sleep 0.3
    kill "${WORKER_PID}" >/dev/null 2>&1 || true
    sleep 0.2
    kill -9 "${WORKER_PID}" >/dev/null 2>&1 || true
  fi
  rm -f "${worker_socket_path}" >/dev/null 2>&1 || true
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

start_worker() {
  local worker_args=(
    -m ai_framework.inference.worker
    --socket-path "${SOCKET_PATH}"
    --checkpoint "${CHECKPOINT_PATH}"
    --scalers "${SCALERS_PATH}"
    --device "${WORKER_DEVICE}"
    --sample-rate-hz "${SAMPLE_RATE_HZ}"
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

  echo "[launcher] starting Python worker..."
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" "${worker_args[@]}"
  ) &
  WORKER_PID=$!
  echo "[launcher] worker pid=${WORKER_PID}"
}

start_reduced_worker() {
  local worker_args=(
    -m ai_framework.reduced_hardware.worker
    --socket-path "${REDUCED_SOCKET_PATH}"
    --checkpoint "${REDUCED_CHECKPOINT_PATH}"
    --device "${WORKER_DEVICE}"
    --sample-rate-hz "${SAMPLE_RATE_HZ}"
  )

  rm -f "${REDUCED_SOCKET_PATH}" >/dev/null 2>&1 || true

  echo "[launcher] starting reduced-hardware Python worker..."
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" "${worker_args[@]}"
  ) &
  WORKER_PID=$!
  echo "[launcher] reduced worker pid=${WORKER_PID}"
}

wait_for_socket() {
  local socket_path="$1"
  local waited=0
  while [[ ! -S "${socket_path}" ]]; do
    sleep 0.2
    waited=$((waited + 1))
    if (( waited >= SOCKET_WAIT_TIMEOUT_SEC * 5 )); then
      echo "[launcher] socket not ready at ${socket_path} after ${SOCKET_WAIT_TIMEOUT_SEC}s" >&2
      return 1
    fi
  done
  echo "[launcher] socket ready: ${socket_path}"
}

run_rust() {
  local rust_args=()

  if [[ "${MODE}" == "reduced-hardware" ]]; then
    rust_args+=(
      --reduced-hardware
      --reduced-socket-path "${REDUCED_SOCKET_PATH}"
      --reduced-capture-path "${REDUCED_CAPTURE_PATH}"
      --sample-rate-hz "${SAMPLE_RATE_HZ}"
      --uart-port "${UART_PORT}"
      --uart-baud "${UART_BAUD}"
    )
    if is_valon_enabled; then
      rust_args+=(
        --enable-valon
        --valon-socket-path "${VALON_SOCKET_PATH}"
      )
    else
      rust_args+=(--disable-valon)
    fi
    if [[ "${REDUCED_SIMULATE}" == "1" ]]; then
      rust_args+=(
        --simulate
        --simulate-cycles "${SIMULATE_CYCLES}"
        --simulate-interval-ms "${SIMULATE_INTERVAL_MS}"
        --dry-run-samples "${SIMULATE_SAMPLES}"
      )
    fi
  else
    rust_args+=(
      --ipc-mode "${IPC_MODE}"
      --socket-path "${SOCKET_PATH}"
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
    else
      rust_args+=(
        --uart-port "${UART_PORT}"
        --uart-baud "${UART_BAUD}"
        --udp-bind "${UDP_BIND}"
      )
    fi
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

if [[ "${MODE}" == "reduced-hardware" ]]; then
  start_reduced_worker
  wait_for_socket "${REDUCED_SOCKET_PATH}"
else
  start_worker
  wait_for_socket "${SOCKET_PATH}"
fi
start_valon_worker
if ! wait_for_valon_socket; then
  echo "[launcher] valon startup failed" >&2
  exit 1
fi
run_rust
