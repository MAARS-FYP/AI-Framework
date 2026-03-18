from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import torch

from ai_framework.config import DSPConfig
from ai_framework.core.dsp import calculate_evm, compute_spectrogram, symbolic_coupled_filter_center_select
from ai_framework.inference.config import InferenceConfig
from ai_framework.inference.output import AgentOutput, InferenceOutput
from ai_framework.models.agents import IFAmpAgent, LNAAgent, MixerAgent
from ai_framework.models.backbone import Backbone


LNA_LABELS = {0: "3V", 1: "5V"}
FILTER_LABELS = {0: "1MHz", 1: "10MHz", 2: "20MHz"}


class RFInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        scalers_path: str = "checkpoints/scalers.joblib",
        device: str = "auto",
        config: Optional[InferenceConfig] = None,
        latent_dim: int = 64,
    ):
        self.config = config or InferenceConfig()
        self.device = self._resolve_device(device)

        self.backbone = Backbone(latent_dim=latent_dim).to(self.device)
        self.lna = LNAAgent(latent_dim).to(self.device)
        self.mixer = MixerAgent(latent_dim).to(self.device)
        self.if_amp = IFAmpAgent(latent_dim).to(self.device)

        ckpt = torch.load(Path(checkpoint_path), map_location=self.device)
        self.backbone.load_state_dict(ckpt["backbone"])
        self.lna.load_state_dict(ckpt["lna"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.if_amp.load_state_dict(ckpt["if_amp"])

        self.backbone.eval()
        self.lna.eval()
        self.mixer.eval()
        self.if_amp.eval()

        self.scalers = joblib.load(Path(scalers_path))
        metrics_scaler = self.scalers["metrics"]
        self.metrics_mean = np.asarray(metrics_scaler.mean_, dtype=np.float32)
        self.metrics_scale = np.asarray(metrics_scaler.scale_, dtype=np.float32)
        self.metrics_scale[self.metrics_scale == 0.0] = 1.0

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _to_complex_iq(iq_samples: Any) -> np.ndarray:
        iq = np.asarray(iq_samples)
        if np.iscomplexobj(iq):
            if iq.ndim != 1:
                raise ValueError("Complex IQ input must be 1D.")
            return iq.astype(np.complex64)

        if iq.ndim == 2 and iq.shape[1] == 2:
            return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)

        raise ValueError("IQ input must be complex 1D or real Nx2 [I, Q].")

    @staticmethod
    def _zscore_spectrogram(spec: torch.Tensor) -> torch.Tensor:
        out = spec.clone()
        for ch in range(2):
            channel = out[:, ch, :, :]
            mean = channel.mean(dim=(1, 2), keepdim=True)
            std = channel.std(dim=(1, 2), keepdim=True)
            out[:, ch, :, :] = (channel - mean) / (std + 1e-8)
        return out

    def _compute_stft_and_metrics(
        self,
        iq_complex: np.ndarray,
        power_lna_dbm: float,
        power_pa_dbm: float,
        sample_rate_hz: Optional[float] = None,
    ):
        iq_tensor = torch.from_numpy(iq_complex)
        iq_batch = iq_tensor.unsqueeze(0)

        sr = self.config.sample_rate_hz if sample_rate_hz is None else float(sample_rate_hz)
        dsp_cfg = DSPConfig(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            center=self.config.center,
            sample_rate_hz=sr,
        )

        spec = compute_spectrogram(iq_batch, config=dsp_cfg)
        spec = spec.to(torch.float32)

        stft_complex = torch.complex(spec[:, 0, :, :], spec[:, 1, :, :]).to(torch.complex64)
        stft_raw = torch.view_as_real(stft_complex)

        if self.config.evm_mode != "blind":
            raise ValueError("Only blind EVM mode is currently implemented.")
        evm_value = float(calculate_evm(iq_batch, reference_data=None, normalize=True).item())

        metrics = np.array([[evm_value, float(power_lna_dbm), float(power_pa_dbm)]], dtype=np.float32)
        metrics_norm = ((metrics - self.metrics_mean[None, :]) / self.metrics_scale[None, :]).astype(np.float32)

        spec_norm = self._zscore_spectrogram(spec)
        metrics_tensor = torch.from_numpy(metrics_norm)

        return spec_norm, metrics_tensor, stft_raw, evm_value, sr

    def infer_from_iq_and_power(
        self,
        iq_samples: Any,
        power_lna_dbm: float,
        power_pa_dbm: float,
        sample_rate_hz: Optional[float] = None,
    ) -> InferenceOutput:
        raw = self.infer_compact(
            iq_samples=iq_samples,
            power_lna_dbm=power_lna_dbm,
            power_pa_dbm=power_pa_dbm,
            sample_rate_hz=sample_rate_hz,
        )

        center_label = f"{self.config.center_freqs_mhz[int(raw['center_class'])]} MHz"

        return InferenceOutput(
            lna=AgentOutput(value=raw["lna_class"], unit="class", label=LNA_LABELS[raw["lna_class"]]),
            filter=AgentOutput(
                value=raw["filter_class"],
                unit="class",
                label=FILTER_LABELS[raw["filter_class"]],
                status=raw["status"],
            ),
            mixer_power=AgentOutput(value=raw["mixer_dbm"], unit="dBm", label="Optimal_LO_Power_dBm"),
            mixer_center_freq=AgentOutput(
                value=raw["center_class"],
                unit="class",
                label=center_label,
                status=raw["status"],
            ),
            if_amp=AgentOutput(value=raw["ifamp_db"], unit="dB", label="Optimal_IF_Gain_dB"),
            metadata={
                "evm": {
                    "value": raw["evm_value"],
                    "unit": "percent",
                    "mode": self.config.evm_mode,
                    "modulation": self.config.modulation,
                },
                "sample_rate_hz": raw["sample_rate_hz"],
                "processing_time_ms": raw["processing_time_ms"],
                "engine": "RFInferenceEngine",
                "version": "1.0",
                "device": str(self.device),
                "config": asdict(self.config),
            },
        )

    def infer_compact(
        self,
        iq_samples: Any,
        power_lna_dbm: float,
        power_pa_dbm: float,
        sample_rate_hz: Optional[float] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        iq_complex = self._to_complex_iq(iq_samples)

        spec, metrics, stft_raw, evm_value, sr = self._compute_stft_and_metrics(
            iq_complex=iq_complex,
            power_lna_dbm=power_lna_dbm,
            power_pa_dbm=power_pa_dbm,
            sample_rate_hz=sample_rate_hz,
        )

        with torch.no_grad():
            z = self.backbone(spec.to(self.device), metrics.to(self.device))

            lna_logits = self.lna(z)
            lna_class = int(lna_logits.argmax(dim=1).item())

            mixer_norm = self.mixer(z).detach().cpu().view(-1, 1).numpy()
            ifamp_norm = self.if_amp(z).detach().cpu().view(-1, 1).numpy()
            mixer_dbm = float(self.scalers["mixer_power"].inverse_transform(mixer_norm)[0, 0])
            ifamp_db = float(self.scalers["if_gain"].inverse_transform(ifamp_norm)[0, 0])

            stft_np = torch.view_as_complex(stft_raw).detach().cpu().numpy()[0]
            filter_class, center_class, sym_status = symbolic_coupled_filter_center_select(
                stft_np,
                sample_rate_hz=sr,
                n_fft=2 * stft_np.shape[0],
                threshold_db=self.config.threshold_db,
                boundary_low_mhz=self.config.boundary_low_mhz,
                boundary_high_mhz=self.config.boundary_high_mhz,
                center_freqs_mhz=self.config.center_freqs_mhz,
                edge_margin_bins=self.config.edge_margin_bins,
                center_tolerance_bins=self.config.center_tolerance_bins,
                energy_floor_db=self.config.energy_floor_db,
                min_span_bins=self.config.min_span_bins,
                allow_center_shift=self.config.allow_center_shift,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "lna_class": int(lna_class),
            "filter_class": int(filter_class),
            "center_class": int(center_class),
            "mixer_dbm": float(mixer_dbm),
            "ifamp_db": float(ifamp_db),
            "evm_value": float(evm_value),
            "sample_rate_hz": float(sr),
            "processing_time_ms": round(float(elapsed_ms), 4),
            "status": str(sym_status),
        }

    def batch_infer_from_iq_and_power(
        self,
        batch_iq_samples: Iterable[Any],
        batch_power_lna_dbm: Iterable[float],
        batch_power_pa_dbm: Iterable[float],
        sample_rate_hz: Optional[float] = None,
    ) -> List[InferenceOutput]:
        outputs = []
        for iq, p_lna, p_pa in zip(batch_iq_samples, batch_power_lna_dbm, batch_power_pa_dbm):
            outputs.append(
                self.infer_from_iq_and_power(
                    iq_samples=iq,
                    power_lna_dbm=float(p_lna),
                    power_pa_dbm=float(p_pa),
                    sample_rate_hz=sample_rate_hz,
                )
            )
        return outputs

    def infer_to_dict(self, *args, **kwargs) -> Dict[str, Any]:
        return self.infer_from_iq_and_power(*args, **kwargs).to_dict()
