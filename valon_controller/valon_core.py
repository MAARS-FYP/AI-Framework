"""Headless Valon controller used by IPC worker."""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional

from valon_protocol import (
    build_set_freq_command,
    build_set_rflevel_command,
    validate_set_freq_mhz,
    validate_set_rflevel_dbm,
)
from valon_serial_py3 import ValonSerial


_RX_FREQ = re.compile(r"^F\s+([-+]?\d+(?:\.\d+)?)\s+([A-Za-z]+)")
_RX_PWR = re.compile(r"^PWR\s+([-+]?\d+(?:\.\d+)?)")


class ValonController:
    def __init__(self, port: Optional[str] = None, timeout: float = 1.0, baud: int = 115200):
        self._serial = ValonSerial(port=port, timeout=timeout, preferred_baud=baud)
        self._lock = threading.RLock()

    def close(self) -> None:
        self._serial.close()

    def _filtered(self, cmd: str, lines: List[str]) -> List[str]:
        out: List[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s == "-->":
                continue
            # skip command echo line
            if s.upper().startswith(cmd.upper()):
                continue
            out.append(s)
        return out

    def _send(self, cmd: str) -> List[str]:
        try:
            lines = self._serial.command(cmd)
        except Exception:
            # one retry after reconnect
            self._serial.close()
            lines = self._serial.command(cmd)
        return self._filtered(cmd, lines)

    def set_frequency_mhz(self, mhz: Any) -> Dict[str, Any]:
        value = validate_set_freq_mhz(mhz)
        cmd = build_set_freq_command(value)
        with self._lock:
            self._send(cmd)
            state = self.get_current_state()
            state["requested_mhz"] = value
            state["applied_command"] = cmd
            return state

    def set_rf_level_dbm(self, dbm: Any) -> Dict[str, Any]:
        value = validate_set_rflevel_dbm(dbm)
        cmd = build_set_rflevel_command(value)
        with self._lock:
            self._send(cmd)
            state = self.get_current_state()
            state["requested_dbm"] = value
            state["applied_command"] = cmd
            return state

    def get_current_state(self) -> Dict[str, Any]:
        with self._lock:
            lines = self._send("DALL")
            freq: Optional[float] = None
            rf: Optional[float] = None
            for line in lines:
                clean = line.replace(";", "")
                m_f = _RX_FREQ.match(clean)
                if m_f:
                    value = float(m_f.group(1))
                    units = m_f.group(2).upper()
                    if units == "MHZ":
                        freq = value
                    elif units == "GHZ":
                        freq = value * 1000.0
                    elif units == "KHZ":
                        freq = value / 1000.0
                    continue
                m_p = _RX_PWR.match(clean)
                if m_p:
                    rf = float(m_p.group(1))
            return {
                "frequency_mhz": freq,
                "rf_level_dbm": rf,
                "port": self._serial.port,
                "baudrate": self._serial.baudrate,
            }

    def get_lock(self) -> Dict[str, Any]:
        with self._lock:
            lines = self._send("lock")
            sub1 = sub2 = main = None
            for line in lines:
                u = line.upper()
                if u.startswith("SUB1"):
                    sub1 = "LOCKED" in u
                elif u.startswith("SUB2"):
                    sub2 = "LOCKED" in u
                elif u.startswith("MAIN"):
                    main = "LOCKED" in u
            locked = bool(sub1 and sub2 and main) if None not in (sub1, sub2, main) else None
            return {
                "sub1_locked": sub1,
                "sub2_locked": sub2,
                "main_locked": main,
                "locked": locked,
            }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            try:
                lines = self._send("status")
                status_line = lines[0] if lines else None
                connected = True
            except Exception as exc:
                status_line = None
                connected = False
                return {
                    "connected": connected,
                    "port": self._serial.port,
                    "baudrate": self._serial.baudrate,
                    "last_error": str(exc),
                    "status_line": status_line,
                    "available_ports": [p.device for p in self._serial.list_ports()],
                }

            return {
                "connected": connected,
                "port": self._serial.port,
                "baudrate": self._serial.baudrate,
                "last_error": self._serial.last_error,
                "status_line": status_line,
                "available_ports": [p.device for p in self._serial.list_ports()],
            }
