"""
Training script for the MAARS RF Control System.

Usage: python -m ai_framework.train [--epochs N] [--batch-size N] [--lr F]
"""

import argparse
import datetime
import json
import joblib
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ai_framework.dataset.dataset import create_dataloaders
from ai_framework.models.backbone import Backbone
from ai_framework.models.agents import LNAAgent, FilterAgent, MixerAgent, IFAmpAgent


class UncertaintyLossBalancer(nn.Module):
    """Learn task weights from homoscedastic uncertainty parameters."""

    def __init__(self, loss_names, min_log_var: float = -4.0, max_log_var: float = 4.0):
        super().__init__()
        self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in loss_names})
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)

    def combine(self, losses):
        first_loss = next(iter(losses.values()))
        total = torch.zeros((), device=first_loss.device, dtype=first_loss.dtype)
        weights = {}
        for name, loss in losses.items():
            log_var = torch.clamp(self.log_vars[name], self.min_log_var, self.max_log_var)
            precision = torch.exp(-log_var)
            total = total + 0.5 * (precision * loss + log_var)
            weights[name] = float(precision.detach().cpu().item())
        return total, weights


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
    report_symbolic_baseline=True,
    tensorboard=False,
    tb_logdir="runs/maars",
    adapter_dim=16,
):
    device = get_device()
    print(f"Device: {device}")

    # --- Data ---
    train_loader, val_loader, scalers = create_dataloaders(
        csv_path, data_root, batch_size=batch_size, val_split=val_split,
    )
    print(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    # --- Models ---
    metric_dim = int(len(scalers["metrics"].mean_))
    backbone = Backbone(latent_dim=latent_dim, metric_dim=metric_dim).to(device)
    lna = LNAAgent(latent_dim, adapter_dim=adapter_dim).to(device)
    filt = FilterAgent()  # symbolic — no learnable parameters
    mixer = MixerAgent(latent_dim, adapter_dim=adapter_dim).to(device)
    if_amp = IFAmpAgent(latent_dim, adapter_dim=adapter_dim).to(device)
    loss_balancer = UncertaintyLossBalancer(["lna", "mixer", "if_amp"]).to(device)

    all_params = (
        list(backbone.parameters()) + list(lna.parameters())
        + list(mixer.parameters()) + list(if_amp.parameters())
        + list(loss_balancer.parameters())
    )
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.9)

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    writer = None
    if tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_run_dir = Path(tb_logdir) / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tb_run_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_run_dir))
            writer.add_text(
                "hparams/config",
                json.dumps(
                    {
                        "csv_path": csv_path,
                        "data_root": data_root,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "latent_dim": latent_dim,
                        "val_split": val_split,
                        "adapter_dim": adapter_dim,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                0,
            )
        except ImportError:
            print("TensorBoard requested, but torch.utils.tensorboard is unavailable. Continuing without logging.")
            writer = None

    filt_baseline_acc = None
    if report_symbolic_baseline:
        print("\nComputing FilterAgent baseline accuracy...")
        filt_correct_total = 0
        filt_total_samples = 0
        with torch.no_grad():
            for (_, _, stft_raw), tgt in val_loader:
                filt_preds = filt(stft_raw)
                filt_correct_total += (filt_preds == tgt["filter"].cpu()).sum().item()
                filt_total_samples += len(tgt["filter"])
        filt_baseline_acc = (filt_correct_total / filt_total_samples * 100) if filt_total_samples > 0 else 0.0
        print(f"FilterAgent (symbolic, 0 params): {filt_baseline_acc:.1f}%\n")
        if writer is not None:
            writer.add_scalar("symbolic/filter_baseline_acc_pct", filt_baseline_acc, 0)

    try:
        # --- Training Loop ---
        for epoch in range(1, epochs + 1):
            # Train
            backbone.train(); lna.train(); mixer.train(); if_amp.train(); loss_balancer.train()
            train_loss = 0.0
            train_loss_components = {"lna": 0.0, "mixer": 0.0, "if_amp": 0.0}

            for batch_idx, ((specs, mets, stft_raw), tgt) in enumerate(train_loader):
                specs, mets = specs.to(device), mets.to(device)
                tgt = {k: v.to(device) for k, v in tgt.items()}

                if writer is not None and epoch == 1 and batch_idx == 0:
                    try:
                        writer.add_graph(backbone, (specs, mets))
                    except Exception as exc:
                        print(f"TensorBoard graph logging skipped: {exc}")

                z = backbone(specs, mets)
                loss_terms = {
                    "lna": ce(lna(z), tgt["lna"]),
                    "mixer": mse(mixer(z), tgt["mixer_power"]),
                    "if_amp": mse(if_amp(z), tgt["if_amp"]),
                }
                loss, train_weights = loss_balancer.combine(loss_terms)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()
                for name, component in loss_terms.items():
                    train_loss_components[name] += component.item()

            train_loss /= len(train_loader)
            for name in train_loss_components:
                train_loss_components[name] /= len(train_loader)

            # Validate
            backbone.eval(); lna.eval(); mixer.eval(); if_amp.eval(); loss_balancer.eval()
            val_loss = 0.0
            val_component_loss = {"lna": 0.0, "mixer": 0.0, "if_amp": 0.0}
            lna_correct = total = 0
            mixer_ae_sum = ifamp_ae_sum = 0.0  # absolute error sums for regression
            mixer_se_sum = ifamp_se_sum = 0.0  # squared error sums
            mixer_tgt_sq_sum = ifamp_tgt_sq_sum = 0.0  # for R² calculation
            mixer_tgt_sum = ifamp_tgt_sum = 0.0
            center_freq_counts = [0, 0, 0]  # distribution of predicted centre freqs

            with torch.no_grad():
                for (specs, mets, stft_raw), tgt in val_loader:
                    specs, mets = specs.to(device), mets.to(device)
                    tgt = {k: v.to(device) for k, v in tgt.items()}

                    z = backbone(specs, mets)
                    lna_logits = lna(z)
                    mixer_pred = mixer(z)
                    ifamp_pred = if_amp(z)
                    loss_terms = {
                        "lna": ce(lna_logits, tgt["lna"]),
                        "mixer": mse(mixer_pred, tgt["mixer_power"]),
                        "if_amp": mse(ifamp_pred, tgt["if_amp"]),
                    }
                    loss, _ = loss_balancer.combine(loss_terms)
                    val_loss += loss.item()
                    for name, component in loss_terms.items():
                        val_component_loss[name] += component.item()

                    lna_correct += (lna_logits.argmax(1) == tgt["lna"]).sum().item()

                    # Symbolic centre-frequency classification (coupled with filter decision)
                    # Note: FilterAgent predictions are computed but not stored (symbolic, fixed)
                    filt_preds = filt(stft_raw)
                    cf_preds = filt.last_center_freq_preds()
                    if cf_preds.numel() == 0:
                        cf_preds = mixer.classify_center_freq(stft_raw, filt_preds)
                    for c in cf_preds.tolist():
                        if 0 <= c < len(center_freq_counts):
                            center_freq_counts[c] += 1

                    # Regression metrics (on normalized scale)
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
            for name in val_component_loss:
                val_component_loss[name] /= len(val_loader)

            lna_acc = lna_correct / total * 100

            # R² score for regression agents (clamp to 0-100%)
            mixer_tgt_var = mixer_tgt_sq_sum - (mixer_tgt_sum ** 2) / total
            ifamp_tgt_var = ifamp_tgt_sq_sum - (ifamp_tgt_sum ** 2) / total
            mixer_r2 = max(0.0, (1 - mixer_se_sum / (mixer_tgt_var + 1e-10))) * 100
            ifamp_r2 = max(0.0, (1 - ifamp_se_sum / (ifamp_tgt_var + 1e-10))) * 100

            mixer_mae = mixer_ae_sum / total
            ifamp_mae = ifamp_ae_sum / total
            learned_weights = {
                name: float(torch.exp(-torch.clamp(param, loss_balancer.min_log_var, loss_balancer.max_log_var)).detach().cpu().item())
                for name, param in loss_balancer.log_vars.items()
            }

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"LNA: {lna_acc:.1f}% | "
                f"Mixer R²: {mixer_r2:.1f}% | IFAmp R²: {ifamp_r2:.1f}% | "
                f"CtrFreq: {center_freq_counts} | "
                f"W: lna={learned_weights['lna']:.2f}, mixer={learned_weights['mixer']:.2f}, if_amp={learned_weights['if_amp']:.2f}"
            )

            if writer is not None:
                learning_rate = optimizer.param_groups[0]["lr"]
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.add_scalar("loss/train_lna", train_loss_components["lna"], epoch)
                writer.add_scalar("loss/train_mixer", train_loss_components["mixer"], epoch)
                writer.add_scalar("loss/train_if_amp", train_loss_components["if_amp"], epoch)
                writer.add_scalar("loss/val_lna", val_component_loss["lna"], epoch)
                writer.add_scalar("loss/val_mixer", val_component_loss["mixer"], epoch)
                writer.add_scalar("loss/val_if_amp", val_component_loss["if_amp"], epoch)
                writer.add_scalar("metrics/lna_acc_pct", lna_acc, epoch)
                writer.add_scalar("metrics/mixer_r2_pct", mixer_r2, epoch)
                writer.add_scalar("metrics/ifamp_r2_pct", ifamp_r2, epoch)
                writer.add_scalar("metrics/mixer_mae", mixer_mae, epoch)
                writer.add_scalar("metrics/ifamp_mae", ifamp_mae, epoch)
                writer.add_scalar("optim/lr", learning_rate, epoch)
                for name, value in learned_weights.items():
                    writer.add_scalar(f"loss_weights/{name}", value, epoch)
                writer.add_scalar("symbolic/center_freq_count_2405", center_freq_counts[0], epoch)
                writer.add_scalar("symbolic/center_freq_count_2420", center_freq_counts[1], epoch)
                writer.add_scalar("symbolic/center_freq_count_2435", center_freq_counts[2], epoch)

                for module_name, module in (
                    ("backbone", backbone),
                    ("lna", lna),
                    ("mixer", mixer),
                    ("if_amp", if_amp),
                    ("loss_balancer", loss_balancer),
                ):
                    for param_name, param in module.named_parameters():
                        writer.add_histogram(f"params/{module_name}.{param_name}", param, epoch)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "backbone": backbone.state_dict(),
                    "lna": lna.state_dict(),
                    "mixer": mixer.state_dict(),
                    "if_amp": if_amp.state_dict(),
                    "loss_balancer": loss_balancer.state_dict(),
                }, save_path / "best_model.pt")
                joblib.dump(scalers, save_path / "scalers.joblib")
                print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
        print(f"Model: {save_path / 'best_model.pt'}")
        print(f"Scalers: {save_path / 'scalers.joblib'}")
        print(f"\n{'='*60}")
        print(f"FINAL VALIDATION METRICS (last epoch)")
        print(f"{'='*60}")
        print(f"  LNA Agent     (neural, classification):  {lna_acc:.1f}% accuracy")
        if filt_baseline_acc is None:
            print("  Filter Agent  (symbolic):               baseline not evaluated")
        else:
            print(f"  Filter Agent  (symbolic vs Bandwidth_Hz):      {filt_baseline_acc:.1f}% accuracy")
        print(f"  Mixer Agent   (neural, LO power):        R²={mixer_r2:.1f}%  MAE={mixer_mae:.3f}")
        print(f"  Mixer Agent   (symbolic, centre freq):   "
              f"2405={center_freq_counts[0]}, "
              f"2420={center_freq_counts[1]}, "
              f"2435={center_freq_counts[2]} MHz")
        print(f"  IF Amp Agent  (neural, regression):      R²={ifamp_r2:.1f}%  MAE={ifamp_mae:.3f}")
        print(f"{'='*60}")
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAARS RF Control AI")
    parser.add_argument("--csv", default="ai_framework/dataset/data/optimal_control_dataset.csv")
    parser.add_argument("--data-root", default="ai_framework/dataset/data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--adapter-dim", type=int, default=16)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write TensorBoard event files for scalars, graph, and histograms.",
    )
    parser.add_argument(
        "--tb-logdir",
        default="runs/maars",
        help="Root directory for TensorBoard event files.",
    )
    parser.add_argument(
        "--report-symbolic-baseline",
        action="store_true",
        help="Compute and print one-time symbolic FilterAgent accuracy on validation data.",
    )
    args = parser.parse_args()

    train(
        csv_path=args.csv, data_root=args.data_root,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, latent_dim=args.latent_dim, save_dir=args.save_dir,
        report_symbolic_baseline=args.report_symbolic_baseline,
        tensorboard=args.tensorboard,
        tb_logdir=args.tb_logdir,
        adapter_dim=args.adapter_dim,
    )
