from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np


@dataclass
class SharedMemoryRingSpec:
    name: str
    num_slots: int
    slot_capacity: int

    @property
    def slot_bytes(self) -> int:
        return self.slot_capacity * 2 * np.dtype(np.float32).itemsize

    @property
    def total_bytes(self) -> int:
        return self.num_slots * self.slot_bytes


class SharedMemoryRingBuffer:
    def __init__(self, spec: SharedMemoryRingSpec, create: bool = False):
        self.spec = spec
        self._owner = create
        self.shm = SharedMemory(name=spec.name, create=create, size=spec.total_bytes if create else 0)

    def close(self):
        self.shm.close()

    def unlink(self):
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass

    def _slot_view(self, slot_index: int) -> np.ndarray:
        if slot_index < 0 or slot_index >= self.spec.num_slots:
            raise ValueError(f"slot_index out of range: {slot_index}")
        offset = slot_index * self.spec.slot_bytes
        return np.ndarray(
            shape=(self.spec.slot_capacity, 2),
            dtype=np.float32,
            buffer=self.shm.buf,
            offset=offset,
        )

    def write_slot(self, slot_index: int, iq_complex: np.ndarray) -> int:
        iq = np.asarray(iq_complex, dtype=np.complex64).reshape(-1)
        if iq.size > self.spec.slot_capacity:
            raise ValueError(
                f"IQ sample count {iq.size} exceeds slot capacity {self.spec.slot_capacity}"
            )
        slot = self._slot_view(slot_index)
        slot[: iq.size, 0] = iq.real
        slot[: iq.size, 1] = iq.imag
        if iq.size < self.spec.slot_capacity:
            slot[iq.size :, :] = 0.0
        return int(iq.size)

    def read_slot(self, slot_index: int, n_samples: int) -> np.ndarray:
        if n_samples < 0 or n_samples > self.spec.slot_capacity:
            raise ValueError(f"n_samples out of range: {n_samples}")
        slot = self._slot_view(slot_index)
        data = slot[:n_samples]
        iq = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64, copy=False)
        return iq.copy()
