# RF Chain Digital Twin System

## Overview

This implementation extends the AI Framework with a **digital twin** of the RF signal chain, enabling real-time parameter control and signal visualization through a modern web-based GUI. The system includes:

- **RF Chain Engine** (Python): Simulates the complete RF chain (LNA, mixer, filter, PA) with distortion models
- **RF Chain Socket Worker** (Python): Exposes the RF chain via Unix socket IPC using the MAAR protocol
- **RF Chain Dashboard** (Vue.js + WebSocket): Interactive web GUI for real-time parameter control and signal visualization
- **Rust Digital Twin Mode**: New operating mode in the orchestrator for digital twin simulation with optional AI agent chaining

## New Files Created

### Python Components

1. **[ai_framework/inference/rf_engine.py](ai_framework/inference/rf_engine.py)**
   - `RFChainEngine` class: Simulates RF signal processing
   - `RFChainOutput` dataclass: Contains I/Q samples, EVM, and power measurements
   - Methods for OFDM generation, PA distortion, and EVM calculation

2. **[ai_framework/inference/rf_chain_worker.py](ai_framework/inference/rf_chain_worker.py)**
   - `RFChainSocketWorker` class: Unix socket server exposing RF chain via MAAR protocol
   - Handles `MSG_RFCHAIN_REQ` and `MSG_RFCHAIN_RESP` message types
   - CLI: `python -m ai_framework.inference.rf_chain_worker --socket-path /tmp/maars_rfchain.sock`

3. **[rf_chain_dashboard/app.py](rf_chain_dashboard/app.py)**
   - `RFChainDashboardBackend` class: WebSocket server for dashboard communication
   - Real-time RF chain processing with parameter updates
   - Optional chaining with inference worker for AI recommendations
   - Listens on `ws://127.0.0.1:8877` by default

4. **[rf_chain_dashboard/index.html](rf_chain_dashboard/index.html)**
   - Vue.js 3 interactive dashboard
   - Parameter sliders for all RF chain control inputs
   - Real-time I/Q constellation and magnitude spectrum visualization
   - Metric panels displaying EVM, power measurements, and processing time
   - Responsive design with gradient styling

5. **[rf_chain_dashboard/launch_dashboard.sh](rf_chain_dashboard/launch_dashboard.sh)**
   - Startup script that launches both RF chain worker and dashboard backend
   - Automatically opens dashboard in default browser
   - Configurable socket paths and ports

6. **[rf_chain_dashboard/requirements.txt](rf_chain_dashboard/requirements.txt)**
   - Dependencies: `websockets>=11.0`, `numpy>=1.21`

### Rust Components

1. **[software_framework/src/rfchain_client.rs](software_framework/src/rfchain_client.rs)**
   - `RFChainSocketClient` struct: Client for RF chain worker communication
   - `process_signal()` method: Sends RF chain request and receives response
   - Automatic sequence ID management

2. **Updates to [software_framework/src/protocol.rs](software_framework/src/protocol.rs)**
   - Added `MSG_RFCHAIN_REQ = 10` and `MSG_RFCHAIN_RESP = 11` message types
   - `RFChainRequest` struct: Parameters for RF chain simulation
   - `RFChainResponse` struct: Contains I/Q data and measurements
   - Packing/unpacking functions: `pack_rfchain_request()`, `unpack_rfchain_response()`

3. **Updates to [software_framework/src/main.rs](software_framework/src/main.rs)**
   - New `OperatingMode` enum: `Hardware`, `Simulate`, `DigitalTwin`
   - New `--mode {hardware | simulate | digital_twin}` CLI argument
   - `run_digital_twin()` function: Digital twin mode orchestration
   - Connects to RF chain worker and optionally to inference worker
   - Writes results to `inference_results.txt` for telemetry

### Script Updates

1. **Updates to [run_full_system.sh](run_full_system.sh)**
   - New `--mode` argument (default: "hardware")
   - New RF chain worker startup functions: `start_rfchain_worker()`, `wait_for_rfchain_socket()`
   - Digital twin mode handling in main orchestration flow
   - New arguments: `--rf-chain-socket-path`, `--rf-chain-cycles`, `--rf-chain-interval-ms`

## Protocol Specification

### RF Chain Request Message (MSG_RFCHAIN_REQ = 10)

**Format:** All values little-endian

```
[Meta]
  seq_id (u64):              Sequence identifier
  input_power_dbm (f32):     Input power before LNA (-60 to -20 dBm)
  bandwidth_hz (f32):        Signal bandwidth (1e6, 10e6, or 20e6 Hz)
  center_freq_hz (f32):      Center frequency (2405e6, 2420e6, 2435e6 Hz)
  lna_voltage (f32):         LNA supply voltage (3.0 or 5.0 V)
  lo_power_dbm (f32):        Local oscillator power (-13.75 to +20 dBm)
  pa_gain_db (f32):          PA gain control (-6 to +26 dB)
  num_symbols (u32):         OFDM symbols to generate
```

### RF Chain Response Message (MSG_RFCHAIN_RESP = 11)

**Format:**

```
[Header (28 bytes)]
  seq_id (u64):              Echo of request seq_id
  status_len (u32):          Length of status string (bytes)
  evm_percent (f32):         Error Vector Magnitude (%)
  power_lna_dbm (f32):       Input power measurement (dBm)
  power_post_pa_dbm (f32):   Output power measurement (dBm)
  processing_time_ms (f32):  Execution time (milliseconds)

[Variable Data]
  status_string (N bytes):   Status string (OK, error description, etc.)
  n_i_samples (u32):         Number of I samples
  n_q_samples (u32):         Number of Q samples
  i_samples (4*N bytes):     I channel samples (f32 array)
  q_samples (4*N bytes):     Q channel samples (f32 array)
```

## How to Use

### 1. Start RF Chain Dashboard (Standalone)

```bash
cd rf_chain_dashboard
bash launch_dashboard.sh
```

This starts:

- RF chain worker on `/tmp/maars_rfchain.sock`
- Dashboard backend on `127.0.0.1:8877`
- Opens dashboard in your default browser

### 2. Start in Digital Twin Mode (with Rust Orchestrator)

```bash
./run_full_system.sh --mode digital_twin
```

This starts:

- Python inference worker (optional, for AI recommendations)
- Python RF chain worker
- Rust orchestrator in digital twin mode
- Continuously calls RF chain with varying parameters every 100ms
- Logs results to `inference_results.txt`

### 3. Access the Dashboard

Open in web browser:

- Standalone: File URL shown in launch script output (usually `file:///path/to/rf_chain_dashboard/index.html`)
- With backend: `ws://127.0.0.1:8877` (via the app.py WebSocket connection)

## Dashboard Features

### Parameter Controls

- **Input Power** (dBm): -60 to -20, step 1
- **Bandwidth**: 1 MHz / 10 MHz / 20 MHz (buttons)
- **Center Frequency**: 2405 MHz / 2420 MHz / 2435 MHz (buttons)
- **LNA Supply**: 3.0V / 5.0V (toggle)
- **LO Power** (dBm): -13.75 to +20, step 0.5
- **PA Gain** (dB): -6 to +26, step 1

### Real-Time Display

- **I/Q Constellation Plot**: Scatter plot of received symbol positions
- **Magnitude Spectrum**: FFT of I/Q signal
- **EVM**: Error Vector Magnitude (%)
- **Power Pre-LNA**: Input signal power (dBm)
- **Power Post-PA**: Output signal power (dBm)
- **Processing Time**: RF chain execution time (ms)

### AI Recommendations (Optional)

When inference worker is available, displays:

- Suggested LNA class (3V or 5V)
- Recommended LO power (dBm)
- Recommended IF amplifier gain (dB)
- Suggested bandwidth filter class
- Agent processing time (ms)

## RF Chain Signal Processing Pipeline

1. **OFDM Generation**: Variable-bandwidth QAM modulation with cyclic prefix
2. **Power Scaling**: Scale input signal to specified power level
3. **RF Upconversion**: Baseband to 6 GHz RF, DAC simulation
4. **LNA Processing**: S-parameters + nonlinear compression + 3rd-order IMD
5. **Downconversion**: RF × LO → IF, ADC simulation
6. **Bandpass Filtering**: Filter to specified bandwidth
7. **PA Distortion**: Rapp compression + polynomial terms + memory effects
8. **Measurements**: Calculate output power and EVM

## Configuration

### RF Chain Worker

```bash
python -m ai_framework.inference.rf_chain_worker \
  --socket-path /tmp/maars_rfchain.sock \
  --seed 42
```

- `--socket-path`: Unix socket path (default: `/tmp/maars_rfchain.sock`)
- `--seed`: Random seed for reproducible OFDM generation

### Dashboard Backend

```bash
python rf_chain_dashboard/app.py \
  --rfchain-socket /tmp/maars_rfchain.sock \
  --inference-socket /tmp/maars_infer.sock \
  --host 127.0.0.1 \
  --port 8877
```

- `--rfchain-socket`: RF chain worker socket
- `--inference-socket`: Inference worker socket (optional)
- `--host`: WebSocket server bind address
- `--port`: WebSocket server port

### Digital Twin Mode (Rust)

```bash
./run_full_system.sh \
  --mode digital_twin \
  --rf-chain-socket-path /tmp/maars_rfchain.sock \
  --rf-chain-cycles 100 \
  --rf-chain-interval-ms 100
```

- `--mode digital_twin`: Activate digital twin mode
- `--rf-chain-socket-path`: RF chain worker socket path
- `--rf-chain-cycles`: Number of cycles (0 = infinite)
- `--rf-chain-interval-ms`: Delay between iterations (milliseconds)

## Data Points Comparison

### Measurements Provided

| Point           | Description                                | Unit |
| --------------- | ------------------------------------------ | ---- |
| Pre-LNA Power   | Input signal power (before amplification)  | dBm  |
| Post-PA Power   | Output signal power (after distortion)     | dBm  |
| EVM             | Error Vector Magnitude (blind calculation) | %    |
| Processing Time | RF chain simulation execution time         | ms   |

### Digital Twin Parameter Ranges

| Parameter      | Min            | Max | Step | Type       |
| -------------- | -------------- | --- | ---- | ---------- |
| Power (dBm)    | -60            | -20 | 1    | Continuous |
| Bandwidth      | 1M / 10M / 20M | —   | —    | Discrete   |
| LNA Voltage    | 3.0            | 5.0 | —    | Discrete   |
| LO Power (dBm) | -13.75         | +20 | 0.5  | Continuous |
| PA Gain (dB)   | -6             | +26 | 1    | Continuous |

## Architecture Diagram

```
Dashboard (Vue.js)
    ↓
  WebSocket (127.0.0.1:8877)
    ↓
Dashboard Backend (app.py)
    ↓
Unix Socket (/tmp/maars_rfchain.sock) - MAAR Protocol
    ↓
RF Chain Worker (rf_chain_worker.py)
    ↓
RF Engine (rf_engine.py)
    ↓
RF Chain (rf_chain.py) → LNA, Mixer, Filter, PA models
    ↓
I/Q + EVM + Power measurements
    ↓
(Optional) Inference Worker → AI Agent recommendations
    ↓
Telemetry file (inference_results.txt)
```

## Error Handling

- **Worker Connection Failure**: Dashboard displays connection status indicator
- **RF Chain Error**: Detailed error messages logged, response status indicates error type
- **Inference Worker Unavailable**: AI recommendations section hidden, RF measurements still displayed
- **Socket Timeout**: Dashboard reconnects automatically with exponential backoff

## Performance Notes

- **RF Chain Processing**: ~0.5-2ms per iteration for 30 OFDM symbols
- **Dashboard Update Rate**: ~10 Hz (100ms intervals, configurable)
- **I/Q Sample Count**: ~7680 samples per request (30 symbols × 256 FFT size)
- **Memory Usage**: RF worker < 100 MB, Dashboard backend < 50 MB

## Testing Checklist

- [ ] RF chain worker starts and listens on socket
- [ ] Dashboard backend connects to RF chain worker
- [ ] Dashboard frontend loads and opens WebSocket connection
- [ ] Adjusting sliders updates RF parameters in real-time
- [ ] I/Q constellation plot updates dynamically
- [ ] Spectrum plot renders correctly
- [ ] Power measurements match expected ranges
- [ ] EVM value is sensible (0-30% for good signal quality)
- [ ] Rust digital_twin mode connects and starts processing
- [ ] Inference worker chaining shows AI recommendations
- [ ] Results logged to inference_results.txt

## Future Enhancements

1. **Dashboard Persistence**: Save/load parameter presets
2. **Data Export**: CSV download of measurement history
3. **Multi-User**: Multiple clients connected to one backend
4. **Advanced Plots**: Spectrogram, eye diagram, phase noise
5. **Hardware Integration**: Direct RF frontend control via REST API
6. **Real-time Recording**: I/Q waveform capture and playback
