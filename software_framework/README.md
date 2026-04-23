# Software Framework for MAARS

## Overview

This will be the final wrapper of the software framework running in the computer to control the physical hardware. It will use the trained neurosymbolic models and wrap them with other necessary software components as described below, to create the full system.

## Plan

- `AI-Framework` will create the neurosymbolic models.
- `Software-Framework` (this) will wrap them with other software components.
- Data receiving layer.
  - Receive UDP packets containng I/Q samples, through Ethernet.
  - Receive power measurements through UART.
  - May run in a separate thread/ core.
- Buffering and storing.
  - Data is buffered in memory, using a circular buffer.
- Pass to AI models.
  - May need to convert them to .onnx and load.
- Send configuration commands back, through UART.

For the implementation, it is planned to use Rust programming language. The reasons are:

1. Good opportunity to learn Rust.
2. Provide high speed systems programming capability similar to C/C++.
3. Memory safety and advanced features.

## UART Output Contract (Host -> STM32)

The software framework emits only high-level agent outputs to STM32. Firmware is responsible for low-level routing and actuator math.

Commands sent by host runtime:

- `lna 3` or `lna 5`
- `filter 1`, `filter 10`, or `filter 20`
- `ifamp x` where `x` is the raw model output for IF amp (`ifamp_db`)

Runtime behavior:

- Commands are sent only when values change (change-driven transmission).
- Command order is deterministic when multiple values change in one cycle: `lna`, then `filter`, then `ifamp`.
- Existing `adc read` power polling remains active for telemetry.

## Valon LO Output Contract (Host -> Valon Worker)

The software framework also sends mixer-agent outputs to the Valon headless worker through Unix-socket IPC (`valon_controller/valon_worker.py`).

Command mapping:

- `center_class` -> `set_freq` with lower-side IF offset: `LO = detected_center_mhz - 25`
  - class 0 -> 2380 MHz
  - class 1 -> 2395 MHz
  - class 2 -> 2410 MHz
- `mixer_dbm` -> `set_rflevel` with the same dBm value

Runtime behavior:

- Valon updates are change-driven.
- When both values change in one cycle, `set_freq` is sent before `set_rflevel`.
- Main loop uses fire-and-forget dispatch (does not block per command waiting for response).

Launcher behavior (`run_full_system.sh`):

- Hardware mode: starts AI worker, waits for socket, starts Valon worker, waits for Valon socket, then starts Rust.
- Hardware mode fails fast if Valon socket is not ready.
- Simulate mode disables Valon by default (override with `--enable-valon`).

Out of current scope:

- Additional LO control channels beyond `set_freq` and `set_rflevel`
