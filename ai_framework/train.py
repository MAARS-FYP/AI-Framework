"""
Training script for the MAARS RF Control System.

Usage: python -m ai_framework.train [--epochs N] [--batch-size N] [--lr F]
"""

import argparse
import joblib
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ai_framework.dataset.dataset import create_dataloaders
from ai_framework.models.backbone import Backbone
from ai_framework.models.agents import LNAAgent, FilterAgent, MixerAgent, IFAmpAgent, SymbolicFilterAgent


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    csv_path="ai_framework/dataset/data/optimal_control_dataset.csv",
    data_root="ai_framework/dataset/data",
    epochs=100,
    batch_size=8,
    lr=1e-3,
    latent_dim=64,
    val_split=0.2,
    save_dir="checkpoints",
):
    device = get_device()
    print(f"Device: {device}")

    # --- Data ---
    train_loader, val_loader, scalers = create_dataloaders(
        csv_path, data_root, batch_size=batch_size, val_split=val_split,
    )
    print(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    # --- Models ---
    backbone = Backbone(latent_dim=latent_dim).to(device)
    lna = LNAAgent(latent_dim).to(device)
    mixer = MixerAgent(latent_dim).to(device)
    filt_neural = FilterAgent(latent_dim).to(device)  # kept for comparison
    filt_symbolic = SymbolicFilterAgent()               # symbolic replacement
    if_amp = IFAmpAgent(latent_dim).to(device)

    # Only neural model parameters go into the optimizer
    # (symbolic filter has no learnable parameters)
    all_params = (
        list(backbone.parameters()) + list(lna.parameters())
        + list(mixer.parameters()) + list(filt_neural.parameters())
        + list(if_amp.parameters())
    )
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # --- Training Loop ---
    for epoch in range(1, epochs + 1):
        # Train
        backbone.train(); lna.train(); mixer.train(); filt_neural.train(); if_amp.train()
        train_loss = 0.0

        for (specs, mets, stft_raw), tgt in train_loader:
            specs, mets = specs.to(device), mets.to(device)
            tgt = {k: v.to(device) for k, v in tgt.items()}

            z = backbone(specs, mets)
            loss = (
                ce(lna(z), tgt["lna"])
                + ce(filt_neural(z), tgt["filter"])
                + mse(mixer(z), tgt["mixer_power"])
                + mse(if_amp(z), tgt["if_amp"])
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        backbone.eval(); lna.eval(); mixer.eval(); filt_neural.eval(); if_amp.eval()
        val_loss = 0.0
        lna_correct = filt_neural_correct = filt_symbolic_correct = total = 0
        mixer_ae_sum = ifamp_ae_sum = 0.0  # absolute error sums for regression
        mixer_se_sum = ifamp_se_sum = 0.0  # squared error sums
        mixer_tgt_sq_sum = ifamp_tgt_sq_sum = 0.0  # for R² calculation
        mixer_tgt_sum = ifamp_tgt_sum = 0.0

        with torch.no_grad():
            for (specs, mets, stft_raw), tgt in val_loader:
                specs, mets = specs.to(device), mets.to(device)
                tgt = {k: v.to(device) for k, v in tgt.items()}

                z = backbone(specs, mets)
                loss = (
                    ce(lna(z), tgt["lna"])
                    + ce(filt_neural(z), tgt["filter"])
                    + mse(mixer(z), tgt["mixer_power"])
                    + mse(if_amp(z), tgt["if_amp"])
                )
                val_loss += loss.item()

                lna_correct += (lna(z).argmax(1) == tgt["lna"]).sum().item()
                filt_neural_correct += (filt_neural(z).argmax(1) == tgt["filter"]).sum().item()

                # Symbolic filter prediction (runs on CPU, no grad needed)
                sym_preds = filt_symbolic(stft_raw)
                filt_symbolic_correct += (sym_preds == tgt["filter"].cpu()).sum().item()

                # Regression metrics (on normalized scale)
                mixer_pred = mixer(z)
                ifamp_pred = if_amp(z)
                mixer_ae_sum += torch.abs(mixer_pred - tgt["mixer_power"]).sum().item()
                ifamp_ae_sum += torch.abs(ifamp_pred - tgt["if_amp"]).sum().item()
                mixer_se_sum += ((mixer_pred - tgt["mixer_power"]) ** 2).sum().item()
                ifamp_se_sum += ((ifamp_pred - tgt["if_amp"]) ** 2).sum().item()
                mixer_tgt_sum += tgt["mixer_power"].sum().item()
                ifamp_tgt_sum += tgt["if_amp"].sum().item()
                mixer_tgt_sq_sum += (tgt["mixer_power"] ** 2).sum().item()
                ifamp_tgt_sq_sum += (tgt["if_amp"] ** 2).sum().item()

                total += len(tgt["lna"])

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        lna_acc = lna_correct / total * 100
        filt_neural_acc = filt_neural_correct / total * 100
        filt_symbolic_acc = filt_symbolic_correct / total * 100

        # R² score for regression agents (clamp to 0-100%)
        mixer_tgt_var = mixer_tgt_sq_sum - (mixer_tgt_sum ** 2) / total
        ifamp_tgt_var = ifamp_tgt_sq_sum - (ifamp_tgt_sum ** 2) / total
        mixer_r2 = max(0.0, (1 - mixer_se_sum / (mixer_tgt_var + 1e-10))) * 100
        ifamp_r2 = max(0.0, (1 - ifamp_se_sum / (ifamp_tgt_var + 1e-10))) * 100

        mixer_mae = mixer_ae_sum / total
        ifamp_mae = ifamp_ae_sum / total

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LNA: {lna_acc:.1f}% | "
            f"Filter(NN): {filt_neural_acc:.1f}% | Filter(Sym): {filt_symbolic_acc:.1f}% | "
            f"Mixer R²: {mixer_r2:.1f}% | IFAmp R²: {ifamp_r2:.1f}%"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "backbone": backbone.state_dict(),
                "lna": lna.state_dict(),
                "mixer": mixer.state_dict(),
                "filter_neural": filt_neural.state_dict(),
                "if_amp": if_amp.state_dict(),
                "filter_method": "symbolic",  # flag indicating symbolic filter is primary
            }, save_path / "best_model.pt")
            joblib.dump(scalers, save_path / "scalers.joblib")
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model: {save_path / 'best_model.pt'}")
    print(f"Scalers: {save_path / 'scalers.joblib'}")
    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION METRICS (last epoch)")
    print(f"{'='*60}")
    print(f"  LNA Agent        (neural, classification):  {lna_acc:.1f}% accuracy")
    print(f"  Filter Agent NN  (neural, classification):  {filt_neural_acc:.1f}% accuracy")
    print(f"  Filter Agent SYM (symbolic, rule-based):    {filt_symbolic_acc:.1f}% accuracy")
    print(f"  Mixer Agent      (neural, regression):      R²={mixer_r2:.1f}%  MAE={mixer_mae:.3f}")
    print(f"  IF Amp Agent     (neural, regression):      R²={ifamp_r2:.1f}%  MAE={ifamp_mae:.3f}")
    print(f"{'='*60}")
    print(f"  Note: Symbolic filter uses PSD histogram with -3dB tail")
    print(f"  removal. 8/300 dataset samples (2.7%) have identical")
    print(f"  spectra but different filter labels — theoretical max")
    print(f"  from spectral analysis alone is 97.3%.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAARS RF Control AI")
    parser.add_argument("--csv", default="ai_framework/dataset/data/optimal_control_dataset.csv")
    parser.add_argument("--data-root", default="ai_framework/dataset/data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--save-dir", default="checkpoints")
    args = parser.parse_args()

    train(
        csv_path=args.csv, data_root=args.data_root,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, latent_dim=args.latent_dim, save_dir=args.save_dir,
    )
