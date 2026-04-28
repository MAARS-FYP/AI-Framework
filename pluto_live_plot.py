#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from multiprocessing import shared_memory

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live ILA IQ plot: time domain / FFT / constellation")
    parser.add_argument("--shm-name", default="maars_iq_ring", help="Shared-memory ring name")
    parser.add_argument("--shm-slots", type=int, default=8, help="Number of shared-memory slots")
    parser.add_argument(
        "--shm-slot-capacity",
        type=int,
        default=8192,
        help="IQ pairs per shared-memory slot",
    )
    parser.add_argument(
        "--slot-index-path",
        default="/tmp/maars_iq_ring_slot.txt",
        help="Sidecar file written by Rust with the latest slot index",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=25e6,
        help="Sample rate used for time axis and FFT frequency axis",
    )
    parser.add_argument("--time-samples", type=int, default=2048, help="Number of IQ samples in time-domain panel")
    parser.add_argument("--fft-size", type=int, default=4096, help="FFT size")
    parser.add_argument("--refresh-hz", type=float, default=8.0, help="Plot refresh rate")
    parser.add_argument("--const-points", type=int, default=3000, help="Maximum constellation points shown")
    return parser.parse_args()


def read_slot_state(path: str) -> tuple[int, int, int]:
    text = Path(path).read_text(encoding="utf-8").strip()
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    if "slot_index" in values:
        slot_index = int(values.get("slot_index", "0"))
        n_samples = int(values.get("n_samples", "0"))
        seq_id = int(values.get("seq_id", "0"))
        return slot_index, n_samples, seq_id

    slot_index = int(text.splitlines()[0]) if text else 0
    return slot_index, 0, 0


def read_latest_iq(
    shm_buf: memoryview,
    shm_slots: int,
    slot_capacity: int,
    slot_index: int,
    n_samples: int,
) -> np.ndarray:
    slot_floats = slot_capacity * 2
    total_floats = shm_slots * slot_floats
    floats = np.ndarray((total_floats,), dtype=np.float32, buffer=shm_buf)

    slot_index = slot_index % max(shm_slots, 1)
    start = slot_index * slot_floats
    stop = start + slot_floats
    slot = floats[start:stop].reshape(-1, 2)

    if n_samples <= 0 or n_samples > slot_capacity:
        n_samples = slot_capacity

    return slot[:n_samples].copy()


def compute_fft_db(iq_complex: np.ndarray, nfft: int) -> np.ndarray:
    if iq_complex.size >= nfft:
        frame = iq_complex[:nfft]
    else:
        frame = np.zeros(nfft, dtype=np.complex64)
        frame[: iq_complex.size] = iq_complex

    window = np.hanning(nfft).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft(frame * window, n=nfft))
    return 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-12))


def prepare_plot(args: argparse.Namespace):
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])

    ax_time = fig.add_subplot(gs[0, 0])
    ax_fft = fig.add_subplot(gs[1, 0])
    ax_const = fig.add_subplot(gs[:, 1])

    time_axis_us = (np.arange(args.time_samples) / args.sample_rate_hz) * 1e6
    line_i, = ax_time.plot(time_axis_us, np.zeros(args.time_samples), label="I", lw=1.0)
    line_q, = ax_time.plot(time_axis_us, np.zeros(args.time_samples), label="Q", lw=1.0)
    ax_time.set_title("Signal in Time Domain")
    ax_time.set_xlabel("Time (µs)")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="upper right")

    fft_axis_mhz = np.fft.fftshift(np.fft.fftfreq(args.fft_size, d=1.0 / args.sample_rate_hz)) / 1e6
    line_fft, = ax_fft.plot(fft_axis_mhz, np.full(args.fft_size, -140.0), lw=1.0)
    ax_fft.set_title("FFT of Signal")
    ax_fft.set_xlabel("Baseband Frequency (MHz)")
    ax_fft.set_ylabel("Magnitude (dB)")
    ax_fft.grid(True, alpha=0.3)

    const_line = ax_const.scatter([], [], s=3.0, alpha=0.65)
    ax_const.set_title("Constellation")
    ax_const.set_xlabel("I")
    ax_const.set_ylabel("Q")
    ax_const.grid(True, alpha=0.3)
    ax_const.set_aspect("equal", adjustable="box")
    ax_const.set_xlim(-2.0, 2.0)
    ax_const.set_ylim(-2.0, 2.0)

    fig.tight_layout()
    return fig, line_i, line_q, line_fft, const_line


def main() -> int:
    args = parse_args()
    stop = {"value": False}

    def _handle_signal(_sig, _frame):
        stop["value"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    refresh_dt = 1.0 / max(args.refresh_hz, 0.1)
    fig, line_i, line_q, line_fft, const_line = prepare_plot(args)

    shm = shared_memory.SharedMemory(name=args.shm_name, create=False)
    try:
        last_state: tuple[int, int, int] | None = None
        while not stop["value"] and plt.fignum_exists(fig.number):
            try:
                slot_index, n_samples, seq_id = read_slot_state(args.slot_index_path)
                last_state = (slot_index, n_samples, seq_id)
                iq = read_latest_iq(shm.buf, args.shm_slots, args.shm_slot_capacity, slot_index, n_samples)
            except FileNotFoundError:
                if last_state is None:
                    line_i.axes.set_title("Signal in Time Domain - waiting for slot index")
                    line_fft.axes.set_title("FFT of Signal - waiting for slot index")
                    const_line.axes.set_title("Constellation - waiting for slot index")
                plt.pause(refresh_dt)
                continue
            except (ValueError, OSError) as exc:
                line_i.axes.set_title(f"Signal in Time Domain - waiting for data ({exc})")
                line_fft.axes.set_title(f"FFT of Signal - waiting for data ({exc})")
                const_line.axes.set_title(f"Constellation - waiting for data ({exc})")
                plt.pause(refresh_dt)
                continue

            if iq.size == 0:
                line_i.axes.set_title("Signal in Time Domain - no samples yet")
                line_fft.axes.set_title("FFT of Signal - no samples yet")
                const_line.axes.set_title("Constellation - no samples yet")
                plt.pause(refresh_dt)
                continue

            i_all = iq[:, 0]
            q_all = iq[:, 1]
            iq_complex = (i_all + 1j * q_all).astype(np.complex64)

            n_time = min(args.time_samples, iq.shape[0])
            if n_time > 0:
                i_data = i_all[:n_time]
                q_data = q_all[:n_time]
                line_i.set_data(line_i.get_xdata()[:n_time], i_data)
                line_q.set_data(line_q.get_xdata()[:n_time], q_data)
                max_abs_time = max(1.0, float(np.max(np.abs(np.concatenate([i_data, q_data])))))
                line_i.axes.set_ylim(-1.1 * max_abs_time, 1.1 * max_abs_time)
                line_i.axes.set_xlim(line_i.get_xdata()[0], line_i.get_xdata()[n_time - 1])

            fft_db = compute_fft_db(iq_complex, args.fft_size)
            line_fft.set_ydata(fft_db)
            line_fft.axes.set_ylim(float(np.min(fft_db) - 5.0), float(np.max(fft_db) + 5.0))

            const = iq[: min(args.const_points, iq.shape[0])]
            const_line.set_offsets(np.column_stack((const[:, 0], const[:, 1])))
            max_abs_const = max(1.0, float(np.max(np.abs(const))))
            const_line.axes.set_xlim(-1.1 * max_abs_const, 1.1 * max_abs_const)
            const_line.axes.set_ylim(-1.1 * max_abs_const, 1.1 * max_abs_const)

            line_i.axes.set_title(f"Signal in Time Domain | seq={seq_id} slot={slot_index}")
            line_fft.axes.set_title("FFT of Signal")
            const_line.axes.set_title(f"Constellation | samples={const.shape[0]}")
            fig.canvas.draw_idle()
            plt.pause(refresh_dt)
    finally:
        shm.close()
        plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
