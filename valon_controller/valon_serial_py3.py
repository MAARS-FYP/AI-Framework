"""Python 3 serial adapter for Valon synthesizer devices."""

from __future__ import annotations

import threading
import time
import os
from dataclasses import dataclass
from typing import List, Optional

import serial
import serial.tools.list_ports


@dataclass
class PortInfo:
    device: str
    description: str
    hwid: str
    vid: Optional[int] = None
    pid: Optional[int] = None


class ValonSerial:
    PROMPT = "-->"

    def __init__(
        self,
        port: Optional[str] = None,
        timeout: float = 1.0,
        preferred_baud: int = 115200,
        probe_bauds: tuple[int, ...] = (9600, 115200),
    ) -> None:
        self._explicit_port = port
        self._timeout = timeout
        self._preferred_baud = preferred_baud
        self._probe_bauds = probe_bauds
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.RLock()
        self.last_error: Optional[str] = None

    @property
    def port(self) -> Optional[str]:
        return None if self._serial is None else self._serial.port

    @property
    def baudrate(self) -> Optional[int]:
        return None if self._serial is None else int(self._serial.baudrate)

    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def close(self) -> None:
        with self._lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                finally:
                    self._serial = None

    def list_ports(self) -> List[PortInfo]:
        ports: List[PortInfo] = []
        for p in serial.tools.list_ports.comports():
            ports.append(
                PortInfo(
                    device=p.device,
                    description=p.description or "",
                    hwid=p.hwid or "",
                    vid=getattr(p, "vid", None),
                    pid=getattr(p, "pid", None),
                )
            )
        return ports

    def _score_port(self, p: PortInfo) -> int:
        score = 0
        desc_u = p.description.upper()
        hw_u = p.hwid.upper()
        if "USB" in desc_u:
            score += 2
        if "FTDI" in desc_u or "FTDI" in hw_u:
            score += 4
        if (p.vid, p.pid) == (0x0403, 0x6001):
            score += 8
        if "VALON" in desc_u:
            score += 8
        return score

    def candidate_ports(self) -> List[str]:
        if self._explicit_port:
            return [self._explicit_port]
        ports = self.list_ports()
        if not ports:
            return []
        ranked = sorted(ports, key=self._score_port, reverse=True)
        strong = [p.device for p in ranked if self._score_port(p) > 0]
        if strong:
            return strong
        return [p.device for p in ranked]

    def connect(self) -> None:
        with self._lock:
            if self.is_open():
                return

            if self._explicit_port:
                available = [p.device for p in self.list_ports()]
                if self._explicit_port not in available and not os.path.exists(self._explicit_port):
                    self.last_error = (
                        f"Configured serial port '{self._explicit_port}' was not found"
                    )
                    raise RuntimeError(self.last_error)

            candidates = self.candidate_ports()
            if not candidates:
                self.last_error = "No serial ports available"
                raise RuntimeError(self.last_error)

            errors: List[str] = []
            for dev in candidates:
                try:
                    ser = serial.Serial(port=dev, timeout=self._timeout)
                    ok = self._probe_device(ser)
                    if not ok:
                        ser.close()
                        errors.append(f"{dev}: no valid prompt")
                        continue
                    self._serial = ser
                    self.last_error = None
                    return
                except Exception as exc:
                    errors.append(f"{dev}: {exc}")

            if self._explicit_port:
                detail = "; ".join(errors) if errors else "no response"
                self.last_error = (
                    f"Device was not detected on configured port '{self._explicit_port}': {detail}"
                )
                raise RuntimeError(self.last_error)

            self.last_error = "; ".join(errors) if errors else "Unable to open serial port"
            raise RuntimeError(self.last_error)

    def _probe_device(self, ser: serial.Serial) -> bool:
        for baud in self._probe_bauds:
            ser.baudrate = baud
            self._write_raw(ser, "\r")
            lines = self._read_all(ser)
            if lines:
                if self._preferred_baud != baud:
                    if not self._change_baud(ser, self._preferred_baud):
                        return False
                return True
        return False

    def _change_baud(self, ser: serial.Serial, new_baud: int) -> bool:
        old = int(ser.baudrate)
        self._write_line(ser, f"Baud {new_baud}")
        self._read_all(ser)
        ser.baudrate = new_baud
        self._write_raw(ser, "\r")
        lines = self._read_all(ser)
        for _ in range(3):
            if lines:
                return True
            time.sleep(0.2)
            lines = self._read_all(ser)
        ser.baudrate = old
        self._write_raw(ser, "\r")
        return bool(self._read_all(ser))

    def _write_raw(self, ser: serial.Serial, text: str) -> None:
        ser.write(text.encode("ascii", errors="ignore"))

    def _write_line(self, ser: serial.Serial, text: str) -> None:
        self._write_raw(ser, f"{text}\r")

    def _decode_line(self, data: bytes) -> str:
        return data.decode("ascii", errors="ignore").replace("\x00", "")

    def _read_all(self, ser: serial.Serial) -> List[str]:
        lines: List[str] = []
        while True:
            raw = ser.readline()
            if not raw:
                break
            line = self._decode_line(raw).rstrip("\n")
            lines.append(line)
            if line.strip() == self.PROMPT:
                break
        return lines

    def command(self, text: str) -> List[str]:
        with self._lock:
            if not self.is_open():
                self.connect()
            assert self._serial is not None
            try:
                self._write_line(self._serial, text)
                return self._read_all(self._serial)
            except Exception as exc:
                self.last_error = str(exc)
                self.close()
                raise
