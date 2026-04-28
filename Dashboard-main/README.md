# Multi-Agent Neurosymbolic AI for Self-Adapting RF Transceivers

Single-page Vanilla HTML dashboard for an AI-controlled RF transceiver chain.

It renders a full-screen spatial hardware diagram:

Antenna -> LNA -> Mixer & LO -> Filter -> IF Amp -> ADC -> FPGA

Measurement and control panels float around the chain and are connected with SVG signal lines.

## Files

- `index.html`: full implementation (HTML, CSS, JavaScript)
- `udp_telemetry_sender.py`: sends JSON telemetry packets over UDP for upstream testing
- `udp_ws_bridge.py`: receives UDP JSON and forwards to WebSocket clients
- `launch_dashboard.sh`: one-line launcher for static server + UDP->WebSocket bridge

## One-Line Launcher

Run bridge and dashboard server together (no UDP sender):

```bash
./launch_dashboard.sh
```

Notes:

- This starts only the dashboard static server and the UDP -> WebSocket bridge.
- UDP sender remains optional and separate simulation input.
- Logs are written to `/tmp/rf_dashboard_http.log` and `/tmp/rf_dashboard_bridge.log`.

## Run

1. Start the UDP -> WebSocket bridge:

```bash
python udp_ws_bridge.py --udp-host 127.0.0.1 --udp-port 9000 --ws-host 127.0.0.1 --ws-port 8765 --ws-path /telemetry
```

2. Start the UDP sender:

```bash
python udp_telemetry_sender.py --host 127.0.0.1 --port 9000 --hz 15
```

3. Open dashboard with the bridge WebSocket endpoint:

```text
index.html?ws=ws://127.0.0.1:8765/telemetry
```

4. Dashboard values update only when new WebSocket messages arrive.

Without incoming data:

- Source mode stays `IDLE` or reconnect states.
- Health shows `STALE`.
- Rate decays to `0 Hz`.

## Connect to Live WebSocket

Pass the endpoint as a query parameter:

- `index.html?ws=ws://127.0.0.1:8765/telemetry`

Or set this global before the main script executes:

- `window.RF_DASHBOARD_WS_URL = "ws://127.0.0.1:8765/telemetry"`

Behavior:

- If WebSocket is connected: source mode shows `LIVE`.
- If WebSocket is unavailable: source mode shows `RETRY` or `ERROR` and retries with backoff.
- There is no built-in simulator mode.

## Panel Drag Persistence

- Floating panel locations are persisted in browser local storage.
- Drag any panel and refresh: positions are restored automatically.
- To reset positions, clear local storage key `rf-dashboard-panel-positions-v1`.

## UDP Test Packet Generator

Use the included script to generate telemetry-shaped UDP packets:

```bash
python udp_telemetry_sender.py --host 127.0.0.1 --port 9000 --hz 15
```

Optional flags:

- `--count 200` send a fixed number of packets
- `--verbose` print each packet
- `--start-seq 1000` set starting sequence ID

Important:

- Browsers cannot consume UDP directly.
- The dashboard expects WebSocket JSON, so run `udp_ws_bridge.py` between this sender and the dashboard.

## Expected Payload

```json
{
  "seq_id": 12345,
  "status_code": 0,
  "lna_class": 1,
  "filter_class": 2,
  "center_class": 1,
  "mixer_dbm": -15.4,
  "ifamp_db": 18.2,
  "power_lna_raw": -35.2,
  "power_pa_raw": -22.8,
  "evm_value": 4.5,
  "processing_time_ms": 0.85
}
```

Class mappings used by UI:

- `lna_class`: `0 -> 3.0V`, `1 -> 5.0V`
- `filter_class`: `0 -> 1 MHz`, `1 -> 10 MHz`, `2 -> 20 MHz`
- `center_class`: `0 -> 2405 MHz`, `1 -> 2420 MHz`, `2 -> 2435 MHz`

Range mappings used by UI bars:

- `mixer_dbm`: `-30` to `+15`
- `ifamp_db`: `-10` to `+30`

## Notes

- Numeric values use a monospace font with tabular numbers to avoid visual jitter.
- EVM panel includes a compact live sparkline.
- Connector lines are pure SVG for zero external dependency.
