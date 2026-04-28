import numpy as np
import scipy.signal as signal
import math
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --- Constants from combined_dataset_gen.py ---
FS_BB = 125e6       # Baseband/IF Sampling Rate
FS_RF = 6e9         # RF Sampling Rate
FC_RF = 2.4e9       # RF Carrier Frequency
FC_IF = 25e6        # Intermediate Frequency

LNA_DATA_PATH = "predictions_at_30.0C.csv"

@dataclass
class OperatingPoint:
    power_pre_lna_dbm: float
    bandwidth: float 

@dataclass
class Setting:
    lna_voltage: float # Discrete Control
    pa_drive_db: float # Continuous Control
    lo_power_dbm: float = 0.0 # Synthesizer Output Power

class RFChain:
    def __init__(self, lna_data_path=LNA_DATA_PATH):
        self.lna_data_map = {}
        self.available_voltages = []
        self.load_lna_data(lna_data_path)

    def load_lna_data(self, path):
        """Loads LNA parameters from the CSV file."""
        # Try finding the file in current dir or parent dir
        search_paths = [path, os.path.join("..", path), os.path.join("data", path)]
        found_path = None
        for p in search_paths:
            if os.path.exists(p):
                found_path = p
                break
        
        if not found_path:
            print(f"Warning: {path} not found in search paths. Using mock data.")
            self._load_mock_data()
            return

        print(f"Loading LNA data from {found_path}...")
        try:
            with open(found_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        v = float(row["Voltage (V)"])
                        params = {
                            "S11_dB": float(row["S11_dB"]),
                            "S11_deg": float(row["S11_deg"]),
                            "S21_dB": float(row["S21_dB"]),
                            "S21_deg": float(row["S21_deg"]),
                            "S12_dB": float(row["S12_dB"]),
                            "S12_deg": float(row["S12_deg"]),
                            "S22_dB": float(row["S22_dB"]),
                            "S22_deg": float(row["S22_deg"]),
                            "IP3_dBm": float(row["IP3_dBm"]),
                            "P1dB_dBm": float(row["P1dB_dBm"]),
                            "Noise_Figure_dB": float(row["Noise_Figure_dB"])
                        }
                        self.lna_data_map[v] = params
                    except ValueError:
                        continue
            self.available_voltages = sorted(list(self.lna_data_map.keys()))
        except Exception as e:
            print(f"Error reading LNA data: {e}. Using mock data.")
            self._load_mock_data()

    def _load_mock_data(self):
        # Provide some default voltages
        self.available_voltages = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for v in self.available_voltages:
            self.lna_data_map[v] = {
                "S11_dB": -10.0, "S11_deg": 0.0,
                "S21_dB": 15.0 + v, "S21_deg": 10.0,
                "S12_dB": -30.0, "S12_deg": 0.0,
                "S22_dB": -10.0, "S22_deg": 0.0,
                "IP3_dBm": 10.0,
                "P1dB_dBm": 0.0,
                "Noise_Figure_dB": 3.0
            }

    def get_lna_params(self, voltage: float) -> Dict:
        if not self.available_voltages:
            return {}
        if voltage in self.lna_data_map:
            return self.lna_data_map[voltage]
        closest = min(self.available_voltages, key=lambda x: abs(x - voltage))
        return self.lna_data_map[closest]

    def lna_model(self, signal_in, lna_params, bw_hz, fs):
        s11_c = self._sparam_to_complex(lna_params["S11_dB"], lna_params["S11_deg"])
        s21_c = self._sparam_to_complex(lna_params["S21_dB"], lna_params["S21_deg"])
        s12_c = self._sparam_to_complex(lna_params["S12_dB"], lna_params["S12_deg"])
        s22_c = self._sparam_to_complex(lna_params["S22_dB"], lna_params["S22_deg"])

        refl_mag = min(0.999, np.abs(s11_c))
        mismatch_factor = np.sqrt(max(0.0, 1.0 - refl_mag**2))

        analytic_in = signal.hilbert(signal_in)
        forward = analytic_in * mismatch_factor
        linear_out = forward * s21_c + forward * s12_c * s22_c

        p1_lin = self._dbm_to_linear(lna_params["P1dB_dBm"])
        v_sat = math.sqrt(p1_lin + 1e-12)
        amp = np.abs(linear_out)
        phase = np.angle(linear_out)
        
        compressed_amp = amp / (1.0 + (amp / (v_sat + 1e-12))**2)
        nonlinear_out = compressed_amp * np.exp(1j * phase)

        ip3_lin = self._dbm_to_linear(lna_params["IP3_dBm"])
        alpha3 = 1.0 / ((math.sqrt(ip3_lin) + 1e-12) ** 2) * 0.01 
        third_order = alpha3 * (np.abs(forward) ** 2) * forward
        nonlinear_out += third_order

        out_real = np.real(nonlinear_out)

        kTB = -174 
        noise_power_dbm = kTB + 10*np.log10(bw_hz) + lna_params["Noise_Figure_dB"]
        p_in_ref_dbm = -60.0
        snr_db = p_in_ref_dbm - noise_power_dbm
        snr_lin = 10**(snr_db / 10.0)
        noise_std = np.sqrt(1.0 / snr_lin)
        
        noise = noise_std * np.random.randn(len(out_real))
        return out_real + noise

    def bandpass_filter(self, signal_in, fs, center_freq, bw):
        if abs(bw - 1e6) < 1e-3:
            wp = [24.2e6, 25.7e6]
            sos = signal.cheby1(3, 0.1, wp, btype='bandpass', fs=fs, output='sos')
        elif abs(bw - 10e6) < 1e-3:
            wp = [19e6, 30.5e6]
            sos = signal.ellip(4, 0.1, 40, wp, btype='bandpass', fs=fs, output='sos')
        elif abs(bw - 20e6) < 1e-3:
            wp = [14e6, 35.5e6]
            sos = signal.ellip(4, 0.1, 40, wp, btype='bandpass', fs=fs, output='sos')
        else:
            nyquist = 0.5 * fs
            low = max(0.001, (center_freq - bw/2) / nyquist)
            high = min(0.999, (center_freq + bw/2) / nyquist)
            sos = signal.butter(4, [low, high], btype='band', output='sos')

        return signal.sosfiltfilt(sos, signal_in)
    
    def generate_variable_bw_ofdm(self, bw, fs, num_symbols=20):
        fft_size = 256
        cp_len = fft_size // 4
        num_active = int(fft_size * (bw / fs))
        if num_active % 2 != 0: num_active += 1
        num_active = max(2, num_active)
        
        num_data_points = num_active * num_symbols
        map_sym = np.array([-3, -1, 1, 3])
        idx_i = np.random.randint(0, 4, num_data_points)
        idx_q = np.random.randint(0, 4, num_data_points)
        qam_symbols = (map_sym[idx_i] + 1j * map_sym[idx_q]) / np.sqrt(10)
        qam_symbols = qam_symbols.reshape(num_symbols, num_active)
        
        ifft_input = np.zeros((num_symbols, fft_size), dtype=np.complex128)
        half_active = num_active // 2
        ifft_input[:, 1:half_active+1] = qam_symbols[:, :half_active]
        ifft_input[:, -half_active:] = qam_symbols[:, half_active:]
        
        time_signal = np.fft.ifft(ifft_input, axis=1)
        signal_with_cp = np.concatenate([time_signal[:, -cp_len:], time_signal], axis=1)
        tx_signal = signal_with_cp.flatten()
        
        tx_signal = tx_signal / np.std(tx_signal)
        return tx_signal, qam_symbols, num_active, cp_len

    def process_chain_pre_pa(self, base_signal, op_point: OperatingPoint, setting: Setting):
        """
        Runs the signal through the RF chain up to the PA input.
        Returns the signal that should enter the PA.
        """
        # Power Scaling
        target_pow_lin = self._dbm_to_linear(op_point.power_pre_lna_dbm)
        current_pow = np.mean(np.abs(base_signal)**2)
        scale_factor = np.sqrt(target_pow_lin / (current_pow + 1e-12))
        tx_bb = base_signal * scale_factor
        
        # Upconversion
        num_samples_rf = int(len(tx_bb) * (FS_RF / FS_BB))
        tx_bb_upsampled = signal.resample(tx_bb, num_samples_rf)
        t_rf = np.arange(num_samples_rf) / FS_RF
        
        carrier_rf = np.exp(1j * 2 * np.pi * FC_RF * t_rf)
        tx_rf = np.real(tx_bb_upsampled * carrier_rf)
        
        # LNA
        lna_params = self.get_lna_params(setting.lna_voltage)
        rx_rf_amp = self.lna_model(tx_rf, lna_params, op_point.bandwidth, FS_RF)
        
        # Downconversion
        FC_LO = FC_RF - FC_IF
        jitter = np.random.normal(0, 0.005, len(t_rf))
        
        # Synthesizer Amplitude Calculation (50 Ohm System)
        # P(W) = 10^(P_dBm/10) / 1000
        # V_rms = sqrt(P * 50)
        # V_peak = V_rms * sqrt(2)
        p_lo_watts = 10**(setting.lo_power_dbm / 10.0) / 1000.0
        v_lo_peak = math.sqrt(p_lo_watts * 50.0) * math.sqrt(2)
        
        # LO Signal with modelled amplitude
        lo_signal = v_lo_peak * np.cos(2 * np.pi * FC_LO * t_rf + jitter)
        
        # Mixer: RF * LO
        mixer_out = rx_rf_amp * lo_signal * 2.0
        
        sos = signal.butter(4, 40e6, 'low', fs=FS_RF, output='sos')
        mixer_out_filtered = signal.sosfilt(sos, mixer_out)
        
        decim_analog = int(FS_RF / FS_BB)
        rx_if_analog = mixer_out_filtered[::decim_analog]
        
        rx_if_filtered = self.bandpass_filter(rx_if_analog, FS_BB, FC_IF, op_point.bandwidth)
        
        rx_if_analytic = signal.hilbert(rx_if_filtered)
        
        # PA Drive
        gain_linear = 10**(setting.pa_drive_db / 20.0)
        pa_in = rx_if_analytic  * gain_linear
        
        return pa_in

    @staticmethod
    def _dbm_to_linear(power_dbm: float) -> float:
        return 10 ** (power_dbm / 10.0)

    @staticmethod
    def _sparam_to_complex(db: float, deg: float) -> complex:
        mag = 10 ** (db / 20.0)
        phase = math.radians(deg)
        return mag * np.exp(1j * phase)
