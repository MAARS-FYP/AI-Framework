#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOCKET_PATH="${SOCKET_PATH:-/tmp/valon5019.sock}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Could not find python3 or ${SCRIPT_DIR}/.venv/bin/python" >&2
    exit 1
  fi
fi

cleanup() {
  if [[ -n "${WORKER_PID:-}" ]] && kill -0 "${WORKER_PID}" >/dev/null 2>&1; then
    kill "${WORKER_PID}" >/dev/null 2>&1 || true
    wait "${WORKER_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

"${PYTHON_BIN}" "${SCRIPT_DIR}/valon_worker.py" --socket "${SOCKET_PATH}" "$@" &
WORKER_PID=$!

for _ in {1..50}; do
  if [[ -S "${SOCKET_PATH}" ]]; then
    break
  fi
  sleep 0.1
done

if [[ ! -S "${SOCKET_PATH}" ]]; then
  echo "Worker did not create socket: ${SOCKET_PATH}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/valon_cli_example.py" --socket "${SOCKET_PATH}"