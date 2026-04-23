# Valon headless worker (Unix socket IPC)

This adds a GUI-independent control path for:

- Frequency in MHz
- RF level in dBm

The worker owns serial access (port scan/open/baud/reconnect), and clients send
high-level commands through a Unix domain socket.

## Files

- valon_worker.py: long-running worker process
- valon_cli_example.py: interactive example external client
- valon_core.py: device control logic
- valon_serial_py3.py: serial discovery/open/read/write helpers
- valon_protocol.py: range validation + response helpers

## Safety limits enforced

From Defaults.py:

- Frequency: 10.0 to 19000.0 MHz
- RF level: -50.0 to 20.0 dBm

Out-of-range requests are rejected before sending device commands.

## Dependency

Python 3 + pyserial

Install:

```
python3 -m pip install pyserial
```

## Run worker

```
python3 valon_worker.py --socket /tmp/valon5019.sock
```

Optional explicit serial port:

```
python3 valon_worker.py --socket /tmp/valon5019.sock --port /dev/ttyUSB0
```

## Run example external client

```
python3 valon_cli_example.py --socket /tmp/valon5019.sock
```

Then type commands:

- `freq 2420`
- `rflevel 2`
- `get`
- `status`
- `quit`

## JSON protocol (line-delimited)

Request examples:

```
{"id":"1","op":"set_freq","value_mhz":2420}
{"id":"2","op":"set_rflevel","value_dbm":2}
{"id":"3","op":"get"}
{"id":"4","op":"status"}
```

Success response shape:

```
{"id":"1","ok":true,"result":{...}}
```

Error response shape:

```
{"id":"1","ok":false,"error":{"code":"OUT_OF_RANGE","message":"...","retryable":false}}
```

## Linux permissions

If device access fails, ensure your user has serial permissions (commonly `dialout`).
