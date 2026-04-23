# Command Line Flags Reference

This file is a compact reference for the command-line entry points in this repository.

The defaults below are taken from the current runtime code paths. For the Rust binary, the printed help text in `software_framework/src/main.rs` is close to the runtime behavior, but a few defaults are defined by `AppConfig` and are the authoritative values listed here.

## Python Inference Worker

Source: `ai_framework/inference/worker.py`

Usage:

```bash
python -m ai_framework.inference.worker [options]
```

Flags:

- `--socket-path <path>`: Unix socket path for the worker control socket. Default: `/tmp/maars_infer.sock`.
- `--checkpoint <path>`: Model checkpoint to load. Default: `checkpoints/best_model.pt`.
- `--scalers <path>`: Scaler bundle to load. Default: `checkpoints/scalers.joblib`.
- `--device <auto|cpu|mps|cuda>`: Torch device selection. Default: `auto`.
- `--sample-rate-hz <float>`: Sample rate supplied to inference preprocessing. Default: `25000000`.
- `--allow-center-shift`: Enable center-shift handling in inference preprocessing.
- `--shm-name <name>`: Shared-memory ring buffer name. Default: unset, which means SHM mode is off.
- `--shm-slots <int>`: Number of SHM slots to create or open. Default: `0`.
- `--shm-slot-capacity <int>`: Capacity of each SHM slot in IQ samples. Default: `0`.
- `--shm-create`: Create the SHM ring buffer if it does not already exist.
- `--shm-unlink-on-exit`: Remove the SHM segment when the worker exits.

Notes:

- This worker stays resident and serves repeated requests over a Unix socket.
- SHM mode is optional and is only used when the client sends a shared-memory inference request.

## Python Inference CLI

Source: `ai_framework/cli/inference_cli.py`

Usage:

```bash
python -m ai_framework.cli.inference_cli [options]
```

Flags:

- `--checkpoint <path>`: Model checkpoint to load. Default: `checkpoints/best_model.pt`.
- `--scalers <path>`: Scaler bundle to load. Default: `checkpoints/scalers.joblib`.
- `--device <auto|cpu|mps|cuda>`: Torch device selection. Default: `auto`.
- `--iq-npy <path>`: Path to a `.npy` file containing I/Q data. Accepts either a complex 1D array or an `Nx2` real-valued array.
- `--power-lna-dbm <float>`: LNA power measurement in dBm.
- `--power-pa-dbm <float>`: PA power measurement in dBm.
- `--sample-rate-hz <float>`: Optional sample rate override used when the JSON payload does not provide one. Default: unset.
- `--input-json <path>`: Read the inference payload from a JSON file.
- `--stdin-json`: Read the inference payload from standard input.
- `--output-json <path>`: Write the output JSON to a file instead of stdout.

Notes:

- Exactly one payload source should be used: `--iq-npy`, `--input-json`, or `--stdin-json`.
- When using `--iq-npy`, both `--power-lna-dbm` and `--power-pa-dbm` are required.
- When a JSON payload is used, the payload must provide `iq_real` and `iq_imag`, or `iq`.

## Python Socket Client

Source: `ai_framework/cli/inference_socket_client.py`

Usage:

```bash
python -m ai_framework.cli.inference_socket_client [options]
```

Flags:

- `--socket-path <path>`: Unix socket path used to connect to the inference worker. Default: `/tmp/maars_infer.sock`.
- `--ping`: Send a ping request and print a health-check response.
- `--shutdown`: Ask the worker to shut down gracefully.
- `--iq-npy <path>`: Path to a `.npy` IQ file for an inference request.
- `--power-lna-dbm <float>`: LNA power measurement in dBm.
- `--power-pa-dbm <float>`: PA power measurement in dBm.
- `--sample-rate-hz <float>`: Sample rate sent to the worker. Default: `0.0`.
- `--seq-id <int>`: Sequence identifier used in the socket protocol. Default: `1`.
- `--use-shm`: Send the IQ data through shared memory instead of in the socket payload.
- `--shm-name <name>`: Shared-memory ring buffer name. Required with `--use-shm`.
- `--shm-slots <int>`: Number of SHM slots. Default: `8`.
- `--shm-slot-capacity <int>`: Capacity of each SHM slot. Default: `8192`.
- `--slot-index <int>`: SHM slot index to write into. Default: `0`.
- `--shm-create`: Create the SHM ring buffer if it does not exist.
- `--shm-unlink`: Remove the SHM ring buffer after the request completes.

Notes:

- `--ping` and `--shutdown` are shortcut control operations; they do not send inference data.
- If `--use-shm` is set, `--shm-name` must also be set.
- For inference requests, `--iq-npy`, `--power-lna-dbm`, and `--power-pa-dbm` are required.

## Python Training CLI

Source: `ai_framework/train.py`

Usage:

```bash
python -m ai_framework.train [options]
```

Flags:

- `--csv <path>`: Training dataset CSV path. Default: `ai_framework/dataset/data/optimal_control_dataset.csv`.
- `--data-root <path>`: Dataset root directory. Default: `ai_framework/dataset/data`.
- `--epochs <int>`: Number of training epochs. Default: `100`.
- `--batch-size <int>`: Mini-batch size. Default: `8`.
- `--lr <float>`: Learning rate. Default: `1e-3`.
- `--latent-dim <int>`: Latent dimension for the model backbone. Default: `64`.
- `--save-dir <path>`: Directory for checkpoints and scaler artifacts. Default: `checkpoints`.

## Rust Runtime

Source: `software_framework/src/main.rs`

Usage:

```bash
cargo run --release -- [options]
```

Top-level flags:

- `--ipc-mode <direct|shm>`: IPC mode used between Rust and the Python worker. Default: `direct`.
- `--socket-path <path>`: Unix socket path for the Python inference worker. Default: `/tmp/maars_infer.sock`.
- `--sample-rate-hz <float>`: Sample rate forwarded to the worker and used by Rust. Default: `25000000.0`.
- `--shm-name <name>`: Shared-memory ring buffer name. Default: `maars_iq_ring`.
- `--shm-slots <int>`: Number of shared-memory slots. Default: `8`.
- `--shm-slot-capacity <int>`: Slot capacity in IQ samples. Default: `8192`.
- `--dry-run`: Run without UART or ILA hardware by generating synthetic IQ data.
- `--simulate`: Run a continuous hardware-free simulation loop.
- `--dry-run-cycles <int>`: Number of dry-run inference cycles. Default: `1`.
- `--simulate-cycles <int>`: Number of simulation cycles; `0` means run continuously. Default: `0`.
- `--simulate-interval-ms <int>`: Delay between simulation cycles. Default: `200`.
- `--dry-run-samples <int>`: Synthetic IQ sample count per dry-run cycle. Default: `4096`.
- `--dry-run-power-lna <float>`: Synthetic LNA power in dBm. Default: `-35`.
- `--dry-run-power-pa <float>`: Synthetic PA power in dBm. Default: `-22`.
- `--uart-port <path>`: UART device path. Default: `/dev/cu.usbmodem11203`.
- `--uart-baud <int>`: UART baud rate. Default: `115200`.
- `--ila-csv-path <path>`: ILA probe0 CSV path. Default: `./ila_probe0.csv`.
- `--ila-request-flag-path <path>`: ILA capture request flag path. Default: `./ila_capture_request.txt`.
- `--ila-poll-interval-ms <int>`: Poll interval for ILA handshake and CSV reads. Default: `20`.
- `--ila-request-timeout-ms <int>`: Timeout waiting for ILA request acknowledge. Default: `5000`.
- `--ila-batch-samples <int>`: Probe0 rows consumed per inference batch. Default: `256`.

Path and mode toggles:

- `--enable-uart-path`: Enable the UART input path. Default: on.
- `--disable-uart-path`: Disable the UART input path.
- `--uart-use-synthetic`: Use synthetic UART input instead of hardware input.
- `--uart-use-real`: Use real UART hardware input.
- `--enable-ila-path`: Enable the ILA CSV input path. Default: on.
- `--disable-ila-path`: Disable the ILA CSV input path.
- `--ila-use-synthetic`: Use synthetic IQ input instead of hardware input.
- `--ila-use-real`: Use real ILA CSV input.
- `--enable-inference`: Enable the Python inference path. Default: on.
- `--disable-inference`: Disable inference and run one displayed hardware path.

Print and logging toggles:

- `--print-inference-results`: Print inference summaries. Default: on.
- `--no-print-inference-results`: Disable inference summaries.
- `--print-uart-input`: Print UART input data.
- `--no-print-uart-input`: Disable UART input data printing.
- `--print-ila-input`: Print ILA CSV decode debug output.
- `--no-print-ila-input`: Disable ILA CSV decode debug output.

Valon output flags:

- `--enable-valon`: Enable Valon LO socket output. Default: on.
- `--disable-valon`: Disable Valon LO socket output.
- `--valon-socket-path <path>`: Unix socket path for the Valon worker. Default: `/tmp/valon5019.sock`.

Other runtime flags:

- `--cleanup-shm-on-exit`: Explicitly unlink the SHM segment when simulation or dry-run finishes.
- `--help` or `-h`: Print the help banner and exit.

Important runtime constraints:

- Inference mode requires both `--enable-uart-path` and `--enable-ila-path`.
- No-inference mode requires exactly one enabled path: either UART or ILA.
- No-inference UART mode must use real hardware, so `--uart-use-synthetic` is rejected there.
- No-inference ILA mode must use real hardware, so `--ila-use-synthetic` is rejected there.
- In no-inference mode, UART requires `--print-uart-input`, and ILA requires `--print-ila-input`.

## Valon Synthesizer Worker

Source: `valon_controller/valon_worker.py`

Usage:

```bash
python valon_controller/valon_worker.py [options]
```

Flags:

- `--socket <path>`: Unix socket path used by Valon clients. Default: `/tmp/valon5019.sock`.
- `--port <path>`: Optional explicit serial port.
- `--timeout <float>`: Serial timeout in seconds. Default: `1.0`.
- `--baud <int>`: Target baud rate. Default: `115200`.
- `--log-level <level>`: Python logging level. Default: `INFO`.

Notes:

- The worker listens on a Unix socket and shuts down cleanly on SIGINT or SIGTERM.
- The socket is removed on exit.

## Valon Example CLI

Source: `valon_controller/valon_cli_example.py`

Usage:

```bash
python valon_controller/valon_cli_example.py [options]
```

Flags:

- `--socket <path>`: Unix socket path to connect to. Default: `/tmp/valon5019.sock`.
- `--command <text>`: Optional one-shot command string.

Supported commands:

- `freq <mhz>`: Set frequency in MHz.
- `rflevel <dbm>`: Set RF level in dBm.
- `get`: Read the current frequency and RF level.
- `status`: Read worker or device status.
- `help`: Show the interactive help text.
- `quit` or `exit`: Leave the interactive shell.

Notes:

- If `--command` is not supplied, the script starts an interactive prompt.
- One-shot mode currently supports `freq`, `rflevel`, `get`, and `status`.

## Full System Launcher

Source: `run_full_system.sh`

Usage:

```bash
./run_full_system.sh [options] [-- <extra rust args>]
```

Launcher flags:

- `--env-file <path>`: Load deployment variables from an environment file before the rest of the script runs. Default: `./.env` if present.
- `--mode <hardware|simulate>`: Start the full hardware loop or the simulation loop.
- `--ipc-mode <direct|shm>`: IPC mode passed to both Rust and the Python worker.
- `--socket-path <path>`: Unix socket path for the inference worker. Default: `/tmp/maars_infer.sock`.
- `--sample-rate-hz <float>`: Sample rate used by both the worker and Rust.
- `--shm-name <name>`: Shared-memory ring buffer name. Default: `maars_iq_ring`.
- `--shm-slots <int>`: Shared-memory slot count. Default: `8`.
- `--shm-slot-capacity <int>`: Shared-memory slot capacity. Default: `8192`.
- `--no-worker-shm-create`: Do not pass `--shm-create` to the Python worker.
- `--worker-no-unlink-on-exit`: Do not pass `--shm-unlink-on-exit` to the Python worker.
- `--checkpoint <path>`: Python worker checkpoint path.
- `--scalers <path>`: Python worker scalers path.
- `--worker-device <auto|cpu|mps|cuda>`: Python worker device selection.
- `--uart-port <path>`: UART port passed to Rust in hardware mode.
- `--uart-baud <int>`: UART baud passed to Rust in hardware mode.
- `--ila-csv-path <path>`: ILA probe0 CSV path passed to Rust in hardware mode.
- `--ila-request-flag-path <path>`: ILA capture request flag passed to Rust in hardware mode.
- `--ila-poll-interval-ms <int>`: ILA CSV poll interval passed to Rust in hardware mode.
- `--ila-request-timeout-ms <int>`: ILA request timeout passed to Rust in hardware mode.
- `--ila-batch-samples <int>`: Probe0 rows consumed per inference in Rust hardware mode.
- `--simulate-cycles <int>`: Number of simulation cycles. `0` means continuous.
- `--simulate-interval-ms <int>`: Delay between simulation cycles.
- `--simulate-samples <int>`: Synthetic IQ samples per cycle in simulation mode.
- `--simulate-power-lna <float>`: Synthetic LNA power in dBm.
- `--simulate-power-pa <float>`: Synthetic PA power in dBm.
- `--rust-cleanup-shm-on-exit`: Forward `--cleanup-shm-on-exit` to Rust.
- `--enable-valon`: Force Valon worker and Rust Valon output on.
- `--disable-valon`: Force Valon worker and Rust Valon output off.
- `--valon-socket-path <path>`: Valon worker socket path. Default: `/tmp/valon5019.sock`.
- `--valon-port <path>`: Explicit Valon serial port.
- `--valon-timeout <float>`: Valon serial timeout in seconds. Default: `1.0`.
- `--valon-baud <int>`: Valon baud rate. Default: `115200`.
- `--valon-log-level <level>`: Valon worker log level. Default: `INFO`.
- `--valon-wait-timeout <int>`: Seconds to wait for the Valon socket. Default: `5`.
- `--python-bin <path>`: Override the Python executable used for the worker and Valon helper.
- `--help` or `-h`: Print launcher help and exit.

Forwarding behavior:

- Everything after `--` is forwarded to Rust unchanged as extra Rust arguments.
- In hardware mode, the launcher passes UART and ILA CSV handshake settings to Rust.
- In simulation mode, the launcher passes simulation-only values to Rust instead of hardware I/O settings.
- The launcher starts the Python inference worker first, then the Valon worker when enabled, and then the Rust runtime.

Examples:

```bash
./run_full_system.sh --mode hardware --ipc-mode shm
./run_full_system.sh --mode simulate --ipc-mode shm --simulate-cycles 10
./run_full_system.sh --mode hardware --ipc-mode direct -- --sample-rate-hz 25000000
```

## Bandwidth Extraction Test Utility

Source: `ai_framework/tests/test_bandwidth_extraction.py`

Usage:

```bash
python -m ai_framework.tests.test_bandwidth_extraction [options]
```

Flags:

- `--num-samples <int>` or `-n <int>`: Number of samples to test. Default: `30`.
- `--sample-rate <float>`: Sample rate in Hz. Default: `125000000.0`.
- `--n-fft <int>`: FFT size. Default: `2048`.
- `--threshold <float>`: Threshold in dB below the peak. Default: `3.0`.
- `--visualize` or `-v`: Generate visualization plots.
- `--sweep`: Run a parameter sweep to find an optimal configuration.
- `--calibrate-boundaries`: Auto-learn bandwidth boundaries for the chosen threshold.
- `--coupled`: Run the coupled symbolic filter and center-frequency evaluation.
- `--quiet` or `-q`: Suppress per-sample output.

## Practical Notes

- If you are starting the entire system, the launcher script is the safest entry point because it keeps the worker, Valon process, and Rust binary aligned on socket names, SHM names, and sample rate.
- If you only need inference, the persistent Python worker plus the socket client is the lowest-overhead path.
- If you are writing new automation, prefer documenting the exact flags you pass alongside the relevant source file path in this reference so the intent stays clear.
