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
