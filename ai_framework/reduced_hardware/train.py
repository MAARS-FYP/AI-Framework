from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ai_framework.reduced_hardware.config import ReducedHardwareConfig
from ai_framework.reduced_hardware.dataset import create_dataloaders
from ai_framework.reduced_hardware.model import ReducedHardwareFFTNet
from ai_framework.reduced_hardware.synthetic import generate_synthetic_dataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model: ReducedHardwareFFTNet, loader, device: torch.device):
    model.eval()
    center_correct = 0
    bandwidth_correct = 0
    total = 0
    center_confusion = torch.zeros(3, 3, dtype=torch.int64)
    bandwidth_confusion = torch.zeros(3, 3, dtype=torch.int64)

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            center_targets = targets["center_class"].to(device)
            bandwidth_targets = targets["bandwidth_class"].to(device)

            outputs = model(inputs)
            center_preds = outputs["center_logits"].argmax(dim=1)
            bandwidth_preds = outputs["bandwidth_logits"].argmax(dim=1)

            center_correct += (center_preds == center_targets).sum().item()
            bandwidth_correct += (bandwidth_preds == bandwidth_targets).sum().item()
            total += inputs.size(0)

            for true_class, pred_class in zip(center_targets.tolist(), center_preds.tolist()):
                center_confusion[true_class, pred_class] += 1
            for true_class, pred_class in zip(bandwidth_targets.tolist(), bandwidth_preds.tolist()):
                bandwidth_confusion[true_class, pred_class] += 1

    return {
        "center_accuracy": center_correct / max(1, total),
        "bandwidth_accuracy": bandwidth_correct / max(1, total),
        "center_confusion": center_confusion,
        "bandwidth_confusion": bandwidth_confusion,
    }


def train(args: argparse.Namespace) -> Path:
    device = get_device() if args.device == "auto" else torch.device(args.device)
    config = ReducedHardwareConfig(sample_rate_hz=args.sample_rate_hz, n_fft=args.n_fft)

    manifest_path = Path(args.manifest)
    data_root = Path(args.data_root) if args.data_root else manifest_path.parent
    if args.generate_synthetic:
        synthetic_root = Path(args.synthetic_output_dir)
        manifest_path = generate_synthetic_dataset(
            output_dir=synthetic_root,
            samples_per_class=args.synthetic_samples_per_class,
            sample_rate_hz=args.sample_rate_hz,
            num_samples=args.n_fft,
        )
        data_root = synthetic_root

    train_loader, val_loader, dataset = create_dataloaders(
        manifest_csv_path=manifest_path,
        data_root=data_root,
        config=config,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        sample_column=args.sample_column,
    )

    model = ReducedHardwareFFTNet(input_length=config.n_fft, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            center_targets = targets["center_class"].to(device)
            bandwidth_targets = targets["bandwidth_class"].to(device)

            outputs = model(inputs)
            center_loss = criterion(outputs["center_logits"], center_targets)
            bandwidth_loss = criterion(outputs["bandwidth_logits"], bandwidth_targets)
            loss = center_loss + bandwidth_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item())

        val_metrics = evaluate(model, val_loader, device)
        val_loss = float(1.0 - 0.5 * (val_metrics["center_accuracy"] + val_metrics["bandwidth_accuracy"]))

        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={running_loss / max(1, len(train_loader)):.4f} | "
            f"center_acc={val_metrics['center_accuracy'] * 100:.1f}% | bandwidth_acc={val_metrics['bandwidth_accuracy'] * 100:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "reduced_hardware_fftnet.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "sample_rate_hz": config.sample_rate_hz,
                        "n_fft": config.n_fft,
                        "use_log_magnitude": config.use_log_magnitude,
                        "normalize_feature": config.normalize_feature,
                    },
                    "manifest": str(manifest_path),
                    "data_root": str(data_root),
                    "center_classes": config.center_frequency_class_to_mhz,
                    "bandwidth_classes": config.bandwidth_class_to_mhz,
                },
                checkpoint_path,
            )
            metadata_path = output_dir / "reduced_hardware_fftnet.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "manifest": str(manifest_path),
                        "data_root": str(data_root),
                        "center_accuracy": val_metrics["center_accuracy"],
                        "bandwidth_accuracy": val_metrics["bandwidth_accuracy"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )

    print("Center confusion matrix:\n", val_metrics["center_confusion"].cpu().numpy())
    print("Bandwidth confusion matrix:\n", val_metrics["bandwidth_confusion"].cpu().numpy())
    return output_dir / "reduced_hardware_fftnet.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the reduced-hardware FFT classifier")
    parser.add_argument("--manifest", default="ai_framework/reduced_hardware/data/synthetic/manifest.csv")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="checkpoints/reduced_hardware")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--sample-rate-hz", type=float, default=125e6)
    parser.add_argument("--n-fft", type=int, default=16384)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample-column", default=None)
    parser.add_argument("--generate-synthetic", action="store_true")
    parser.add_argument("--synthetic-output-dir", default="ai_framework/reduced_hardware/data/synthetic")
    parser.add_argument("--synthetic-samples-per-class", type=int, default=6)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
