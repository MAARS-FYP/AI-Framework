#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="hardware"
IPC_MODE="shm"
SOCKET_PATH="/tmp/maars_infer.sock"
SAMPLE_RATE_HZ="25000000"

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

usage() {
  cat <<'EOF'
Usage:
  ./run_full_system.sh [options] [-- <extra rust args>]

Options:
  --mode <hardware|simulate>         Run full hardware loop or simulation mode
  --ipc-mode <direct|shm>            IPC mode for Rust + Python worker
  --socket-path <path>               Unix socket path (default: /tmp/maars_infer.sock)
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
  --udp-bind <host:port>             Rust UDP bind (hardware mode)

  --simulate-cycles <int>            Simulation cycles (0 = continuous)
  --simulate-interval-ms <int>       Delay between simulation cycles
  --simulate-samples <int>           Simulation IQ samples per cycle
  --simulate-power-lna <float>       Simulation LNA power dBm
  --simulate-power-pa <float>        Simulation PA power dBm

  --rust-cleanup-shm-on-exit         Pass --cleanup-shm-on-exit to Rust
  --python-bin <path>                Override Python executable
  --help                             Show this help

Examples:
  ./run_full_system.sh --mode hardware --ipc-mode shm
  ./run_full_system.sh --mode simulate --ipc-mode shm --simulate-cycles 10
  ./run_full_system.sh --mode hardware --ipc-mode direct -- --sample-rate-hz 25000000
EOF
}

EXTRA_RUST_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --ipc-mode) IPC_MODE="$2"; shift 2 ;;
    --socket-path) SOCKET_PATH="$2"; shift 2 ;;
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
    --udp-bind) UDP_BIND="$2"; shift 2 ;;
    --simulate-cycles) SIMULATE_CYCLES="$2"; shift 2 ;;
    --simulate-interval-ms) SIMULATE_INTERVAL_MS="$2"; shift 2 ;;
    --simulate-samples) SIMULATE_SAMPLES="$2"; shift 2 ;;
    --simulate-power-lna) SIMULATE_POWER_LNA="$2"; shift 2 ;;
    --simulate-power-pa) SIMULATE_POWER_PA="$2"; shift 2 ;;
    --rust-cleanup-shm-on-exit) RUST_CLEANUP_SHM_ON_EXIT="1"; shift ;;
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

if [[ "${MODE}" != "hardware" && "${MODE}" != "simulate" ]]; then
  echo "Invalid --mode: ${MODE}" >&2
  exit 2
fi

if [[ "${IPC_MODE}" != "direct" && "${IPC_MODE}" != "shm" ]]; then
  echo "Invalid --ipc-mode: ${IPC_MODE}" >&2
  exit 2
fi

cleanup() {
  set +e
  if [[ -n "${WORKER_PID}" ]] && kill -0 "${WORKER_PID}" 2>/dev/null; then
    "${PYTHON_BIN}" -m ai_framework.cli.inference_socket_client --socket-path "${SOCKET_PATH}" --shutdown >/dev/null 2>&1 || true
    sleep 0.3
    kill "${WORKER_PID}" >/dev/null 2>&1 || true
    sleep 0.2
    kill -9 "${WORKER_PID}" >/dev/null 2>&1 || true
  fi
  rm -f "${SOCKET_PATH}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

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
    --ipc-mode "${IPC_MODE}"
    --socket-path "${SOCKET_PATH}"
    --sample-rate-hz "${SAMPLE_RATE_HZ}"
  )

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
wait_for_socket
run_rust
