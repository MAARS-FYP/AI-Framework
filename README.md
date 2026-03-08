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
