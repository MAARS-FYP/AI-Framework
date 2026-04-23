# MAARS RF Control AI Framework

Neurosymbolic framework for adaptive RF receiver control using a multi-agent architecture that combines neural networks with symbolic (rule-based) signal processing.

## Features

- **Dual-Branch Backbone**: Fuses visual (spectrogram CNN) and parametric (sensor MLP) features into a shared latent vector
- **Neurosymbolic Multi-Agent System**: Neural agents (LNA, Mixer LO power, IF Amp) operate on the latent vector; symbolic agents (Filter, Mixer centre-freq) operate directly on raw STFT data
- **Automatic Normalization**: Per-sample Z-score for spectrograms, sklearn StandardScaler for metrics/targets (fit on train split only)
- **Symbolic DSP**: PSD-based bandwidth extraction and centre-frequency classification with no learnable parameters

## Architecture

```
Raw I/Q Signal → STFT + EVM → Two data paths:

  ┌─ Neural Path:  Normalized spectrogram + metrics → Backbone → Latent z
  │                 → LNA Agent (classification: 3V/5V)
  │                 → Mixer Agent (regression: LO power)
  │                 → IF Amp Agent (regression: IF gain)
  │
  └─ Symbolic Path: Raw complex STFT (unnormalized)
                    → Filter Agent (rule-based: bandwidth → 1/10/20 MHz)
                    → Mixer Agent (rule-based: centre-freq → 2405/2420/2435 MHz)
```

## Host-to-STM UART Contract

For deployment, the host software sends only high-level agent outputs over UART. The STM32 firmware owns low-level hardware actuation details (mux routing, IF mapping equation, and register-level programming).

Host command set:

- `lna 3` or `lna 5`
- `filter 1`, `filter 10`, or `filter 20`
- `ifamp x` where `x` is the raw IF-amp model output (`ifamp_db`)

Notes:

- `adc read` remains the telemetry path from STM32 to host for power monitoring.

## Host-to-Valon Socket Contract

LO control is sent to the Valon headless worker over Unix-socket JSON-line IPC.

Host command mapping:

- `mixer_dbm` -> `{"op":"set_rflevel","value_dbm":mixer_dbm}`
- `center_class` -> detected center frequency `{2405,2420,2435} MHz`
- LO frequency sent to Valon uses lower-side injection with 25 MHz IF offset:
  - class `0`: `2405 - 25 = 2380` MHz
  - class `1`: `2420 - 25 = 2395` MHz
  - class `2`: `2435 - 25 = 2410` MHz

Runtime behavior:

- Valon commands are change-driven (sent only when mapped value changes).
- Frequency updates are sent before RF-level updates when both change in one cycle.

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd AI-Framework

# Install dependencies
pip install torch numpy pandas scikit-learn
```

### Training

```bash
python -m ai_framework.train \
    --csv ai_framework/dataset/data/optimal_control_dataset.csv \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-3
```

Trained model and scalers are saved to `checkpoints/`.

### Inference (End-to-End)

Run end-to-end inference from fresh I/Q and two power measurements:

- Inputs: I/Q samples, `power_lna_dbm`, `power_pa_dbm`
- Internally computed: STFT and EVM
- Outputs: LNA class, Filter class, Mixer LO power, Mixer center frequency class, IF amp gain

```bash
python -m ai_framework.cli.inference_cli \
        --iq-npy input_iq.npy \
        --power-lna-dbm -35.2 \
        --power-pa-dbm -22.8 \
        --checkpoint checkpoints/best_model.pt \
        --scalers checkpoints/scalers.joblib
```

`input_iq.npy` can be:

- complex 1D array (`complex64`/`complex128`) of I/Q samples, or
- real `Nx2` array where columns are `[I, Q]`.

For integration with external programs (e.g., Rust), you can use JSON input/output:

```bash
python -m ai_framework.cli.inference_cli --stdin-json << 'JSON'
{
    "iq_real": [0.1, 0.2, 0.3],
    "iq_imag": [0.0, -0.1, 0.2],
    "power_lna_dbm": -35.2,
    "power_pa_dbm": -22.8,
    "sample_rate_hz": 25000000.0
}
JSON
```

The CLI prints a stable JSON output payload to stdout (or `--output-json <path>`).

### Continuous Low-Latency Inference (Recommended for Rust Loop)

For continuous RF control loops, do **not** spawn the JSON CLI every cycle.
Use the persistent socket worker so models are loaded once and reused.

Start worker (Python side):

```bash
python -m ai_framework.inference.worker \
    --socket-path /tmp/maars_infer.sock \
    --checkpoint checkpoints/best_model.pt \
    --scalers checkpoints/scalers.joblib \
    --sample-rate-hz 25000000
```

Then your Rust process can keep one Unix domain socket connection open and send binary inference requests repeatedly.

Debug/test client examples:

```bash
# health check
python -m ai_framework.cli.inference_socket_client --socket-path /tmp/maars_infer.sock --ping

# one inference request
python -m ai_framework.cli.inference_socket_client \
    --socket-path /tmp/maars_infer.sock \
    --iq-npy input_iq.npy \
    --power-lna-dbm -35.2 \
    --power-pa-dbm -22.8 \
    --sample-rate-hz 25000000

# graceful shutdown
python -m ai_framework.cli.inference_socket_client --socket-path /tmp/maars_infer.sock --shutdown
```

This path removes major overheads:

- No model/scaler reload per request
- No per-cycle process startup
- Compact binary IPC (no long JSON payloads in hot path)

### Shared-Memory Ring Buffer Mode (Lower Copy Overhead)

To reduce socket payload copies further, use shared memory for IQ samples and send only descriptors over the socket.

Worker with SHM enabled:

```bash
python -m ai_framework.inference.worker \
    --socket-path /tmp/maars_infer.sock \
    --checkpoint checkpoints/best_model.pt \
    --scalers checkpoints/scalers.joblib \
    --sample-rate-hz 25000000 \
    --shm-name maars_iq_ring \
    --shm-slots 8 \
    --shm-slot-capacity 8192 \
    --shm-create
```

Optional cleanup behavior:

- Add `--shm-unlink-on-exit` if you want the worker to remove the SHM segment when it shuts down.

Test client using SHM descriptor requests:

```bash
python -m ai_framework.cli.inference_socket_client \
    --socket-path /tmp/maars_infer.sock \
    --iq-npy input_iq.npy \
    --power-lna-dbm -35.2 \
    --power-pa-dbm -22.8 \
    --sample-rate-hz 25000000 \
    --use-shm \
    --shm-name maars_iq_ring \
    --shm-slots 8 \
    --shm-slot-capacity 8192 \
    --slot-index 0 \
    --shm-create
```

For Rust integration, keep the ring buffer alive and reuse slots in a producer/consumer loop.

### One-Command Deployment Launcher

Use the root launcher script to start the full system in correct order (Python worker first, then Rust app) with matching IPC/SHM settings:

```bash
./run_full_system.sh --mode hardware --ipc-mode shm
```

Hardware mode now expects I/Q ingress from Vivado ILA CSV capture:

1. Run `vivado_ila_capture_to_csv.tcl` in Vivado (`maars_ila_capture_loop`).
2. Rust creates `ila_capture_request.txt` to request a capture.
3. TCL writes `ila_probe0.csv` (probe0, depth 16384), then deletes the request file.
4. Rust consumes the first 256 samples, runs inference, then truncates the CSV.

Simulation (no real UART/ILA hardware required):

```bash
./run_full_system.sh --mode simulate --ipc-mode shm --simulate-cycles 10
```

You can keep deployment configuration in one file:

```bash
cp .env.example .env
```

The launcher auto-loads `.env` if present. You can also pass a specific env file:

```bash
./run_full_system.sh --env-file ./deploy.env --mode hardware
```

CLI arguments always override values from `.env`.

What the launcher does:

- Starts Python worker with configured socket/checkpoint/scalers and SHM parameters.
- Waits for Unix socket readiness.
- Starts Rust with matching `--ipc-mode`, socket path, sample-rate, and SHM args.
- Handles graceful shutdown and socket cleanup on exit/signals.

Useful options:

- `--mode hardware|simulate`
- `--env-file ./deploy.env`
- `--ipc-mode direct|shm`
- `--socket-path /tmp/maars_infer.sock`
- `--sample-rate-hz 25000000`
- `--shm-name maars_iq_ring --shm-slots 8 --shm-slot-capacity 8192`
- `--worker-no-unlink-on-exit` (if you do not want worker SHM cleanup)
- `--rust-cleanup-shm-on-exit` (Rust-side explicit SHM cleanup)

Tip: Use only one SHM cleanup owner to avoid duplicate cleanup warnings. Default launcher behavior uses Python worker cleanup (`--shm-unlink-on-exit`).

## Data Normalization

**All data is automatically normalized** for optimal training:

- Spectrograms: Per-sample Z-score normalization (real/imag channels independently)
- Sensor metrics: Dataset-wide StandardScaler (fit on training split only)
- Regression targets: StandardScaler normalized during training, inverse-transformed for inference
- Scalers saved alongside model checkpoint for deployment

## Project Structure

```
ai_framework/
├── __init__.py            # Package version
├── config.py              # DSPConfig, logger utility
├── train.py               # Training script (CLI entry point)
├── inference/
│   ├── config.py          # InferenceConfig (STFT/symbolic/IO parameters)
│   ├── engine.py          # RFInferenceEngine (end-to-end inference runtime)
│   ├── output.py          # Structured inference outputs
│   ├── protocol.py        # Binary IPC protocol for persistent worker
│   ├── shm_ring.py        # Shared-memory ring buffer helper
│   └── worker.py          # Persistent Unix-socket inference worker
├── cli/
│   ├── inference_cli.py   # JSON-capable inference CLI for external programs
│   └── inference_socket_client.py # Socket client (debug/integration testing)
├── core/
│   └── dsp.py             # DSP utilities: STFT, EVM, PSD, symbolic classifiers
├── dataset/
│   ├── dataset.py         # RFDataset, DataLoader, collate, scalers
│   ├── data_analysis.ipynb # Dataset exploration notebook
│   └── data/              # STFT .npy files + CSV labels
├── models/
│   ├── backbone.py        # Dual-branch CNN+MLP backbone → latent z
│   └── agents.py          # LNA, Filter, Mixer, IF Amp agent heads
└── tests/
    └── test_bandwidth_extraction.py  # Symbolic bandwidth extraction tests
```

## Training Output

After training, you'll have:

```
checkpoints/
├── best_model.pt           # State dicts: backbone, lna, mixer, if_amp
└── scalers.joblib          # StandardScalers for metrics, if_gain, mixer_power
```

## Documentation

- [architecture_diagram.txt](architecture_diagram.txt) - Full system architecture with data flow

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- scikit-learn

## License

[Your License Here]
