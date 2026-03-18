# MAARS RF Control AI Framework

Neurosymbolic framework for adaptive RF receiver control using a multi-agent architecture that combines neural networks with symbolic (rule-based) signal processing.

## Features

- **Dual-Branch Backbone**: Fuses visual (spectrogram CNN) and parametric (sensor MLP) features into a shared latent vector
- **Neurosymbolic Multi-Agent System**: Neural agents (LNA, Mixer LO power, IF Amp) operate on the latent vector; symbolic agents (Filter, Mixer centre-freq) operate directly on raw STFT data
- **Automatic Normalization**: Per-sample Z-score for spectrograms, sklearn StandardScaler for metrics/targets (fit on train split only)
- **Symbolic DSP**: PSD-based bandwidth extraction and centre-frequency classification with no learnable parameters

## Architecture

```
Raw I/Q Signal → STFT + EVM → Two data paths:

  ┌─ Neural Path:  Normalized spectrogram + metrics → Backbone → Latent z
  │                 → LNA Agent (classification: 3V/5V)
  │                 → Mixer Agent (regression: LO power)
  │                 → IF Amp Agent (regression: IF gain)
  │
  └─ Symbolic Path: Raw complex STFT (unnormalized)
                    → Filter Agent (rule-based: bandwidth → 1/10/20 MHz)
                    → Mixer Agent (rule-based: centre-freq → 2405/2420/2435 MHz)
```

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd AI-Framework

# Install dependencies
pip install torch numpy pandas scikit-learn
```

### Training

```bash
python -m ai_framework.train \
    --csv ai_framework/dataset/data/optimal_control_dataset.csv \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-3
```

Trained model and scalers are saved to `checkpoints/`.

### Inference (End-to-End)

Run end-to-end inference from fresh I/Q and two power measurements:

- Inputs: I/Q samples, `power_lna_dbm`, `power_pa_dbm`
- Internally computed: STFT and EVM
- Outputs: LNA class, Filter class, Mixer LO power, Mixer center frequency class, IF amp gain

```bash
python -m ai_framework.cli.inference_cli \
        --iq-npy input_iq.npy \
        --power-lna-dbm -35.2 \
        --power-pa-dbm -22.8 \
        --checkpoint checkpoints/best_model.pt \
        --scalers checkpoints/scalers.joblib
```

`input_iq.npy` can be:

- complex 1D array (`complex64`/`complex128`) of I/Q samples, or
- real `Nx2` array where columns are `[I, Q]`.

For integration with external programs (e.g., Rust), you can use JSON input/output:

```bash
python -m ai_framework.cli.inference_cli --stdin-json << 'JSON'
{
    "iq_real": [0.1, 0.2, 0.3],
    "iq_imag": [0.0, -0.1, 0.2],
    "power_lna_dbm": -35.2,
    "power_pa_dbm": -22.8,
    "sample_rate_hz": 25000000.0
}
JSON
```

The CLI prints a stable JSON output payload to stdout (or `--output-json <path>`).

## Data Normalization

**All data is automatically normalized** for optimal training:

- Spectrograms: Per-sample Z-score normalization (real/imag channels independently)
- Sensor metrics: Dataset-wide StandardScaler (fit on training split only)
- Regression targets: StandardScaler normalized during training, inverse-transformed for inference
- Scalers saved alongside model checkpoint for deployment

## Project Structure

```
ai_framework/
├── __init__.py            # Package version
├── config.py              # DSPConfig, logger utility
├── train.py               # Training script (CLI entry point)
├── inference/
│   ├── config.py          # InferenceConfig (STFT/symbolic/IO parameters)
│   ├── engine.py          # RFInferenceEngine (end-to-end inference runtime)
│   └── output.py          # Structured inference outputs
├── cli/
│   └── inference_cli.py   # JSON-capable inference CLI for external programs
├── core/
│   └── dsp.py             # DSP utilities: STFT, EVM, PSD, symbolic classifiers
├── dataset/
│   ├── dataset.py         # RFDataset, DataLoader, collate, scalers
│   ├── data_analysis.ipynb # Dataset exploration notebook
│   └── data/              # STFT .npy files + CSV labels
├── models/
│   ├── backbone.py        # Dual-branch CNN+MLP backbone → latent z
│   └── agents.py          # LNA, Filter, Mixer, IF Amp agent heads
└── tests/
    └── test_bandwidth_extraction.py  # Symbolic bandwidth extraction tests
```

## Training Output

After training, you'll have:

```
checkpoints/
├── best_model.pt           # State dicts: backbone, lna, mixer, if_amp
└── scalers.joblib          # StandardScalers for metrics, if_gain, mixer_power
```

## Documentation

- [architecture_diagram.txt](architecture_diagram.txt) - Full system architecture with data flow

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- scikit-learn

## License

[Your License Here]
