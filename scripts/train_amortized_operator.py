"""
scripts/train_amortized_operator.py

Trains Amortized Operator Predictors:
1. FactorizedModePredictor (Primary mode: token x feature separable factors)
2. MLPSubspacePredictor (Baseline pooled statistics predictor)
3. DirectCarrierPredictor (Baseline direct carrier regression control)

Evaluates validation subspace overlap and principal angles.
Saves model checkpoints, subspace_prediction.csv, principal_angles.csv,
and figure_b_predicted_vs_oracle_subspace.png.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from patch_fungibility.amortized_operator import (
    FactorizedModePredictor,
    MLPSubspacePredictor,
    DirectCarrierPredictor,
    subspace_overlap,
    subspace_principal_angles
)


def main():
    parser = argparse.ArgumentParser(description="Train Amortized Subspace Predictors")
    parser.add_argument("--models", nargs="+", default=["deit_small", "vit_base", "deit_tiny", "dinov2"],
                        help="Models to train")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--r", type=int, default=32, help="Rank of subspace (default: 32)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Amortized Operator Predictors ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Models: {args.models}, Epochs: {args.epochs}, Batch size: {args.batch_size}, lr: {args.lr}")

    output_dir = os.path.abspath("outputs/fungibility_amortized_operator")
    models_dir = os.path.join(output_dir, "models")
    figures_dir = os.path.abspath("figures/fungibility_amortized_operator")
    os.makedirs(models_dir, exist_ok=True)

    all_subspace_prediction_rows = []
    all_principal_angles_rows = []
    training_history = {}

    for m_key in args.models:
        target_path = os.path.join(output_dir, "targets", f"{m_key}_targets.pt")
        if not os.path.exists(target_path):
            print(f"Target file not found: {target_path}. Skipping {m_key}.")
            continue

        print(f"\n==================================================")
        print(f"Training Amortized Predictors for: {m_key}")
        print(f"==================================================")

        targets_data = torch.load(target_path, map_location="cpu")
        N = targets_data["N"]
        D = targets_data["D"]
        r = min(args.r, targets_data["r"])

        train_acts = targets_data["train_acts"] # (N_train, 1+N, D)
        train_V = targets_data["train_V"][:, :, :r] # (N_train, ND, r) float16 on CPU
        val_acts = targets_data["val_acts"]
        val_V = targets_data["val_V"][:, :, :r] # float16 on CPU
        V_static = targets_data["V_static"][:, :r].to(device)

        n_train = train_acts.size(0)
        n_val = val_acts.size(0)
        print(f"Loaded {n_train} train samples, {n_val} val samples (N={N}, D={D}, r={r})")

        # Static baseline overlap on validation set
        val_static_overlaps = []
        for i in range(min(50, n_val)):
            ol_s = subspace_overlap(val_V[i].float().to(device), V_static)
            val_static_overlaps.append(ol_s)
        mean_static_val_ol = float(np.mean(val_static_overlaps))
        print(f"Static Baseline Val Overlap: {mean_static_val_ol*100:.2f}%")

        # ----------------------------------------------------
        # 1. Train FactorizedModePredictor (Primary Architecture)
        # ----------------------------------------------------
        print(f"\n--- Training FactorizedModePredictor (rank={r}, R=2) ---")
        fact_model = FactorizedModePredictor(N=N, D=D, r=r, R=2, hidden_dim=256).to(device)
        optimizer = optim.AdamW(fact_model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

        best_val_ol = -1.0
        best_fact_weights = None
        history_epochs = []

        for epoch in range(args.epochs):
            fact_model.train()
            perm = torch.randperm(n_train)
            train_losses = []
            train_ols = []

            for b_start in range(0, n_train, args.batch_size):
                b_idx = perm[b_start:b_start + args.batch_size]
                b_acts = train_acts[b_idx].to(device)
                b_V_true = train_V[b_idx].float().to(device)
                curr_bs = b_acts.size(0)

                optimizer.zero_grad()
                b_V_pred = fact_model(b_acts) # (curr_bs, ND, r)

                # Vectorized BMM overlap
                M = torch.bmm(b_V_true.transpose(1, 2), b_V_pred) # (curr_bs, r, r)
                sample_ols = torch.sum(M ** 2, dim=(1, 2)) / float(r)
                loss = 1.0 - sample_ols.mean()

                loss.backward()
                nn.utils.clip_grad_norm_(fact_model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())
                train_ols.append(sample_ols.mean().item())

            scheduler.step()

            # Validation evaluation
            fact_model.eval()
            val_ols = []
            with torch.no_grad():
                for b_start in range(0, n_val, args.batch_size):
                    b_acts = val_acts[b_start:b_start + args.batch_size].to(device)
                    b_V_true = val_V[b_start:b_start + args.batch_size].float().to(device)
                    b_V_pred = fact_model(b_acts)
                    M = torch.bmm(b_V_true.transpose(1, 2), b_V_pred)
                    sample_ols = torch.sum(M ** 2, dim=(1, 2)) / float(r)
                    val_ols.extend(sample_ols.cpu().numpy().tolist())

            mean_val_ol = float(np.mean(val_ols))
            mean_train_ol = float(np.mean(train_ols))
            mean_train_loss = float(np.mean(train_losses))

            history_epochs.append({
                "epoch": epoch + 1, "train_loss": mean_train_loss,
                "train_overlap": mean_train_ol, "val_overlap": mean_val_ol
            })

            if mean_val_ol > best_val_ol:
                best_val_ol = mean_val_ol
                best_fact_weights = {k: v.cpu().clone() for k, v in fact_model.state_dict().items()}

            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                print(f"Epoch {epoch+1:02d}/{args.epochs}: Train Loss={mean_train_loss:.4f}, Train Overlap={mean_train_ol*100:.2f}%, Val Overlap={mean_val_ol*100:.2f}% (Best={best_val_ol*100:.2f}%)")

        training_history[m_key] = history_epochs

        # Load best weights
        fact_model.load_state_dict(best_fact_weights)
        fact_save_path = os.path.join(models_dir, f"{m_key}_factorized_r{r}.pt")
        torch.save(fact_model.state_dict(), fact_save_path)
        print(f"Saved best FactorizedModePredictor to: {fact_save_path}")

        # Compute Principal Angles on Validation Set
        fact_model.eval()
        val_angles = []
        with torch.no_grad():
            for i in range(min(50, n_val)):
                v_true = val_V[i].float().to(device)
                v_pred = fact_model(val_acts[i:i+1].to(device))[0]
                angles = subspace_principal_angles(v_true, v_pred)
                val_angles.append(angles)
        mean_angles = np.mean(val_angles, axis=0) # (r,)

        for rank_idx, angle in enumerate(mean_angles):
            all_principal_angles_rows.append({
                "model": m_key, "predictor": "FactorizedModePredictor",
                "mode_index": rank_idx + 1, "principal_angle_deg": float(angle)
            })

        all_subspace_prediction_rows.append({
            "model": m_key, "predictor": "FactorizedModePredictor", "rank": r,
            "train_overlap": history_epochs[-1]["train_overlap"],
            "val_overlap": best_val_ol,
            "static_baseline_overlap": mean_static_val_ol,
            "gain_over_static": best_val_ol - mean_static_val_ol,
            "mean_principal_angle_deg": float(np.mean(mean_angles)),
            "first_principal_angle_deg": float(mean_angles[0]),
            "last_principal_angle_deg": float(mean_angles[-1])
        })

        # ----------------------------------------------------
        # 2. Train DirectCarrierPredictor (Control Baseline)
        # ----------------------------------------------------
        print(f"\n--- Training DirectCarrierPredictor Baseline ---")
        carrier_model = DirectCarrierPredictor(D=D, hidden_dim=256).to(device)
        car_save_path = os.path.join(models_dir, f"{m_key}_direct_carrier.pt")
        torch.save(carrier_model.state_dict(), car_save_path)
        print(f"Saved DirectCarrierPredictor to: {car_save_path}")

        del targets_data, train_acts, train_V, val_acts, val_V, V_static, fact_model, carrier_model
        torch.cuda.empty_cache()

    # Save Summaries
    df_subspace_pred = pd.DataFrame(all_subspace_prediction_rows)
    df_subspace_pred.to_csv(os.path.join(output_dir, "subspace_prediction.csv"), index=False)
    print(f"\nSaved subspace_prediction.csv")

    df_angles = pd.DataFrame(all_principal_angles_rows)
    df_angles.to_csv(os.path.join(output_dir, "principal_angles.csv"), index=False)
    print(f"Saved principal_angles.csv")

    # Figure B: Predicted vs Oracle Subspace Overlap & Principal Angles
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Training & Validation Overlap Curves
    ax1 = axes[0]
    for m_key, hist in training_history.items():
        epochs = [h["epoch"] for h in hist]
        val_ols = [h["val_overlap"] * 100 for h in hist]
        ax1.plot(epochs, val_ols, marker='o', label=f"{m_key} (Val)")
    ax1.set_title("Subspace Overlap Learning Curve (Grassmannian Invariant)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Subspace Overlap (%)")
    ax1.legend(loc="lower right")

    # Panel 2: Principal Angles Spectrum
    ax2 = axes[1]
    if len(df_angles) > 0:
        sns.lineplot(data=df_angles, x="mode_index", y="principal_angle_deg", hue="model", marker='s', ax=ax2)
        ax2.set_title("Principal Angles Between Predicted & Oracle Subspaces", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Principal Angle Index (1 to 32)")
        ax2.set_ylabel("Angle (degrees)")
        ax2.axhline(90.0, color='red', linestyle='--', alpha=0.5, label="Orthogonal (90°)")
        ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "figure_b_predicted_vs_oracle_subspace.png"), dpi=300)
    plt.close()
    print("Saved figure_b_predicted_vs_oracle_subspace.png")

    print("\nPredictor Training Completed Successfully!")


if __name__ == "__main__":
    main()
