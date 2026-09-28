import os
import time
import json
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from patch_fungibility.models import load_deit_model, get_model_parameter_hash
from patch_fungibility.pipeline import compute_logits_and_margins
from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits
from patch_fungibility.v0_7_masks import get_nested_patch_masks_v0_7
from patch_fungibility.v0_8_pca import compute_calibration_pca, get_random_orthonormal_basis
from patch_fungibility.v0_8_diagnostics import compute_representation_ranks, forward_block_with_diagnostics
from patch_fungibility.v0_8_interventions import (
    apply_zero_to_mask,
    apply_centroid_to_mask,
    apply_perturbation_to_mask,
    generate_diagonal_gaussian_eps,
    generate_shared_noise_eps,
    generate_isotropic_full_eps,
    generate_pca_rank_eps,
    generate_random_rank_eps,
    generate_grouped_diversity_eps,
)


class PatchFungibilityV08Pipeline:
    def __init__(
        self,
        model_name: str = "deit_tiny_patch16_224",
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "outputs/fungibility_v0_8",
        mask_seed: int = 9601,
        pca_ranks: List[int] = [1, 2, 4, 8, 16, 32, 64],
        grouped_k: List[int] = [1, 2, 4, 8, 16, 32, 64, 196],
        diag_gaussian_seeds: List[int] = [11001, 11002, 11003, 11004, 11005],
        pca_rank_seeds: List[int] = [11001, 11002, 11003],
        random_rank_seeds: List[int] = [12001, 12002, 12003],
        isotropic_seeds: List[int] = [13001, 13002, 13003],
        shared_vs_indep_seeds: List[int] = [14001, 14002, 14003, 14004, 14005],
        grouped_seeds: List[int] = [15001, 15002, 15003],
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.output_dir = output_dir
        self.target_depth = 8
        self.mask_seed = mask_seed
        self.pca_ranks = pca_ranks
        self.grouped_k = grouped_k
        self.diag_gaussian_seeds = diag_gaussian_seeds
        self.pca_rank_seeds = pca_rank_seeds
        self.random_rank_seeds = random_rank_seeds
        self.isotropic_seeds = isotropic_seeds
        self.shared_vs_indep_seeds = shared_vs_indep_seeds
        self.grouped_seeds = grouped_seeds

        os.makedirs(self.output_dir, exist_ok=True)
        self.nested_masks = get_nested_patch_masks_v0_7(seed=self.mask_seed)

    def run(
        self,
        calib_loader: DataLoader,
        eval_loader: DataLoader,
        n_samples: int = 1000
    ) -> Dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.time()
        print(f"\n{'='*70}\n[V0.8 Pipeline] Initializing {self.model_name} on {self.device}\n{'='*70}")
        model, initial_hash = load_deit_model(self.model_name, self.device)
        embed_dim = model.embed_dim

        # -----------------------------------------------------------------
        # STEP 1: Compute PCA on Calibration Data (Strictly N=1,000 images)
        # -----------------------------------------------------------------
        print(f"[{self.model_name}] Step 1: Computing PCA basis on Block-8 calibration patch representations...")
        pca_result = compute_calibration_pca(
            model=model,
            model_name=self.model_name,
            calib_loader=calib_loader,
            target_depth=self.target_depth,
            device=self.device
        )
        mu_8 = torch.from_numpy(pca_result["mu_8"]).to(self.device)
        sigma_8 = torch.from_numpy(pca_result["sigma_8"]).to(self.device)
        E_full = pca_result["E_full"]
        eigenvalues = torch.from_numpy(pca_result["eigenvalues"])  # (D,)
        eigenvectors = torch.from_numpy(pca_result["eigenvectors"])  # (D, D)

        print(f"[{self.model_name}] Calibration E_full = {E_full:.4f}")
        print(f"[{self.model_name}] Top 5 PCA eigenvalues: {eigenvalues[:5].numpy().tolist()}")

        # -----------------------------------------------------------------
        # STEP 2: Cache Evaluation Block-8 Activations
        # -----------------------------------------------------------------
        print(f"[{self.model_name}] Step 2: Caching Block-8 evaluation features and clean baseline...")
        cached_h8 = []
        all_targets: List[int] = []
        clean_preds: List[int] = []
        clean_corrects: List[int] = []
        clean_margins: List[float] = []

        with torch.no_grad():
            for images, targets in eval_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                feat = model.patch_embed(images)
                feat = model._pos_embed(feat)
                feat = model.patch_drop(feat)
                feat = model.norm_pre(feat)

                for b in range(self.target_depth + 1):
                    feat = model.blocks[b](feat)

                cached_h8.append(feat.detach().cpu())

                # Continue through blocks 9..11 for clean baseline
                for b in range(self.target_depth + 1, len(model.blocks)):
                    feat = model.blocks[b](feat)

                logits = model.forward_head(model.norm(feat))
                preds, corrects, _, _, margins = compute_logits_and_margins(logits, targets)

                all_targets.extend(targets.cpu().numpy().tolist())
                clean_preds.extend(preds.cpu().numpy().tolist())
                clean_corrects.extend(corrects.cpu().numpy().tolist())
                clean_margins.extend(margins.cpu().numpy().tolist())

        n_eval = len(all_targets)
        eval_h8 = torch.cat(cached_h8, dim=0)  # Shape: (N, 197, D)
        targets_tensor = torch.tensor(all_targets, dtype=torch.long)
        clean_acc = float(np.mean(clean_corrects))
        clean_margin_mean = float(np.mean(clean_margins))
        print(f"[{self.model_name}] Clean Baseline (N={n_eval}): Acc = {clean_acc*100:.2f}%, Mean Margin = {clean_margin_mean:.4f}")

        # -----------------------------------------------------------------
        # STEP 3: Setup Condition Definitions
        # -----------------------------------------------------------------
        # Primary 100% replacement mask: all 196 spatial patch tokens (indices 1..196)
        mask_100 = list(range(1, 197))
        mask_75 = self.nested_masks["75%"]["seq_mask"]

        # Precompute Random Orthonormal Bases
        random_bases: Dict[Tuple[int, int], torch.Tensor] = {}
        max_rand_ortho_err = 0.0
        for r in self.pca_ranks:
            for s in self.random_rank_seeds:
                basis_np = get_random_orthonormal_basis(embed_dim, r, seed=s)
                g_err = float(np.max(np.abs(basis_np.T @ basis_np - np.eye(r))))
                max_rand_ortho_err = max(max_rand_ortho_err, g_err)
                random_bases[(r, s)] = torch.from_numpy(basis_np)

        # -----------------------------------------------------------------
        # Helper: Evaluate Single Condition
        # -----------------------------------------------------------------
        def evaluate_condition_stream(
            cond_name: str,
            seq_mask: List[int],
            gen_type: str,
            seed: Optional[int] = None,
            rank: Optional[int] = None,
            k_val: Optional[int] = None,
            basis_type: Optional[str] = None
        ) -> Dict[str, Any]:
            num_batches = (n_eval + self.batch_size - 1) // self.batch_size
            cond_preds: List[int] = []
            cond_corrects: List[int] = []
            cond_margins: List[float] = []

            # Representation diagnostics per block (depth 8 mod, 9, 10, 11)
            rep_diags = {
                d: {"numerical_rank": [], "stable_rank": [], "effective_rank": [], "mean_pairwise_cosine": []}
                for d in [8, 9, 10, 11]
            }
            # Attention diagnostics per block (9, 10, 11)
            attn_diags = {
                d: {"key_variance": [], "value_variance": [], "cls_attention_entropy": [], "attention_stable_rank": []}
                for d in [9, 10, 11]
            }

            total_pert_energy = 0.0
            total_tokens_perturbed = 0

            # Deterministic generator per condition
            rng = torch.Generator(device="cpu").manual_seed(seed if seed is not None else 42)
            M = len(seq_mask)

            with torch.no_grad():
                for b_idx in range(num_batches):
                    b_start = b_idx * self.batch_size
                    b_end = min(b_start + self.batch_size, n_eval)
                    B_curr = b_end - b_start

                    h_batch = eval_h8[b_start:b_end].to(self.device)
                    b_targets = targets_tensor[b_start:b_end].to(self.device)

                    if gen_type == "clean":
                        h_mod = h_batch
                    elif gen_type == "zero":
                        h_mod = apply_zero_to_mask(h_batch, seq_mask)
                    elif gen_type == "centroid":
                        h_mod = apply_centroid_to_mask(h_batch, seq_mask, mu_8)
                    elif gen_type == "diagonal_gaussian":
                        eps = generate_diagonal_gaussian_eps(B_curr, M, sigma_8, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, seq_mask, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                    elif gen_type == "shared_noise":
                        eps = generate_shared_noise_eps(B_curr, M, sigma_8, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, seq_mask, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                    elif gen_type == "independent_noise":
                        eps = generate_diagonal_gaussian_eps(B_curr, M, sigma_8, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, seq_mask, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                    elif gen_type == "isotropic_full":
                        eps = generate_isotropic_full_eps(B_curr, M, embed_dim, E_full, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, seq_mask, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                    elif gen_type == "pca_rank":
                        assert rank is not None
                        V_R = eigenvectors[:, :rank]
                        e_R = eigenvalues[:rank]
                        eps = generate_pca_rank_eps(B_curr, M, V_R, e_R, E_full, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, seq_mask, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                    elif gen_type == "random_rank":
                        assert rank is not None and seed is not None
                        U_R = random_bases[(rank, seed)]
                        eps = generate_random_rank_eps(B_curr, M, U_R, E_full, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, seq_mask, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                    elif gen_type == "grouped":
                        assert k_val is not None
                        eps = generate_grouped_diversity_eps(B_curr, M, sigma_8, k_val, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, seq_mask, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                    else:
                        raise ValueError(f"Unknown gen_type: {gen_type}")

                    # Representation rank at injection (depth 8 mod)
                    r_diag_8 = compute_representation_ranks(h_mod[:, 1:, :])
                    for k_diag, v_val in r_diag_8.items():
                        rep_diags[8][k_diag].append(v_val * B_curr)

                    # Pass through Blocks 9, 10, 11
                    h_curr = h_mod
                    for b in range(self.target_depth + 1, len(model.blocks)):
                        h_curr, a_diag = forward_block_with_diagnostics(model.blocks[b], h_curr)
                        for k_diag, v_val in a_diag.items():
                            attn_diags[b][k_diag].append(v_val * B_curr)
                        r_diag = compute_representation_ranks(h_curr[:, 1:, :])
                        for k_diag, v_val in r_diag.items():
                            rep_diags[b][k_diag].append(v_val * B_curr)

                    logits = model.forward_head(model.norm(h_curr))
                    preds, corrects, _, _, margins = compute_logits_and_margins(logits, b_targets)

                    cond_preds.extend(preds.cpu().numpy().tolist())
                    cond_corrects.extend(corrects.cpu().numpy().tolist())
                    cond_margins.extend(margins.cpu().numpy().tolist())

            # Summarize diagnostics
            avg_rep_diags = {
                d: {k_diag: float(np.sum(vals) / n_eval) for k_diag, vals in rep_diags[d].items()}
                for d in [8, 9, 10, 11]
            }
            avg_attn_diags = {
                d: {k_diag: float(np.sum(vals) / n_eval) for k_diag, vals in attn_diags[d].items()}
                for d in [9, 10, 11]
            }

            mean_realized_energy = float(total_pert_energy / max(total_tokens_perturbed, 1))
            energy_rel_err = float(abs(mean_realized_energy - E_full) / E_full) if total_tokens_perturbed > 0 else 0.0

            return {
                "cond_name": cond_name,
                "gen_type": gen_type,
                "rank": rank,
                "k_val": k_val,
                "seed": seed,
                "acc": float(np.mean(cond_corrects)),
                "mean_margin": float(np.mean(cond_margins)),
                "preds": cond_preds,
                "corrects": cond_corrects,
                "margins": cond_margins,
                "mean_realized_energy": mean_realized_energy,
                "energy_rel_error": energy_rel_err,
                "representation_diagnostics": avg_rep_diags,
                "attention_diagnostics": avg_attn_diags
            }

        # -----------------------------------------------------------------
        # STEP 4: Execute Primary 100% Replacement Experiment
        # -----------------------------------------------------------------
        print(f"\n[{self.model_name}] Step 4: Executing Primary 100% Replacement Conditions...")
        all_results: Dict[str, Dict[str, Any]] = {}

        # 4A: Clean
        print("  Evaluating clean baseline...")
        all_results["clean"] = evaluate_condition_stream("clean", mask_100, "clean")

        # 4B: Zero
        print("  Evaluating zero condition...")
        all_results["zero"] = evaluate_condition_stream("zero", mask_100, "zero")

        # 4C: Static Centroid
        print("  Evaluating static centroid condition...")
        all_results["static_centroid"] = evaluate_condition_stream("static_centroid", mask_100, "centroid")

        # 4D: Diagonal Gaussian (5 seeds)
        for s in self.diag_gaussian_seeds:
            cname = f"diagonal_gaussian_seed_{s}"
            print(f"  Evaluating {cname}...")
            all_results[cname] = evaluate_condition_stream(cname, mask_100, "diagonal_gaussian", seed=s)

        # 4E: PCA Rank Sweep (7 ranks x 3 seeds)
        for r in self.pca_ranks:
            for s in self.pca_rank_seeds:
                cname = f"pca_rank_{r}_seed_{s}"
                print(f"  Evaluating {cname}...")
                all_results[cname] = evaluate_condition_stream(cname, mask_100, "pca_rank", seed=s, rank=r)

        # 4F: Random Rank Sweep (7 ranks x 3 seeds)
        for r in self.pca_ranks:
            for s in self.random_rank_seeds:
                cname = f"random_rank_{r}_seed_{s}"
                print(f"  Evaluating {cname}...")
                all_results[cname] = evaluate_condition_stream(cname, mask_100, "random_rank", seed=s, rank=r)

        # 4G: Isotropic Full (3 seeds)
        for s in self.isotropic_seeds:
            cname = f"isotropic_full_seed_{s}"
            print(f"  Evaluating {cname}...")
            all_results[cname] = evaluate_condition_stream(cname, mask_100, "isotropic_full", seed=s)

        # 4H: Shared Noise (5 seeds)
        for s in self.shared_vs_indep_seeds:
            cname = f"shared_noise_seed_{s}"
            print(f"  Evaluating {cname}...")
            all_results[cname] = evaluate_condition_stream(cname, mask_100, "shared_noise", seed=s)

        # 4I: Independent Noise (5 seeds)
        for s in self.shared_vs_indep_seeds:
            cname = f"independent_noise_seed_{s}"
            print(f"  Evaluating {cname}...")
            all_results[cname] = evaluate_condition_stream(cname, mask_100, "independent_noise", seed=s)

        # 4J: Grouped Diversity Sweep (8 values of K x 3 seeds)
        for k in self.grouped_k:
            for s in self.grouped_seeds:
                cname = f"grouped_k_{k}_seed_{s}"
                print(f"  Evaluating {cname}...")
                all_results[cname] = evaluate_condition_stream(cname, mask_100, "grouped", seed=s, k_val=k)

        # -----------------------------------------------------------------
        # STEP 5: Execute Secondary 75% Replacement Experiment
        # -----------------------------------------------------------------
        print(f"\n[{self.model_name}] Step 5: Executing Secondary 75% Replacement Conditions...")
        secondary_75_results: Dict[str, Dict[str, Any]] = {}

        # 5A: Static Centroid 75%
        print("  Evaluating 75% static centroid...")
        secondary_75_results["static_centroid_75"] = evaluate_condition_stream(
            "static_centroid_75", mask_75, "centroid"
        )

        # 5B: PCA ranks 1, 4, 16 at 75% (seed 11001)
        for r in [1, 4, 16]:
            cname = f"pca_rank_{r}_75"
            print(f"  Evaluating 75% {cname}...")
            secondary_75_results[cname] = evaluate_condition_stream(
                cname, mask_75, "pca_rank", seed=11001, rank=r
            )

        # 5C: Full Gaussian at 75% (seed 11001)
        print("  Evaluating 75% full gaussian...")
        secondary_75_results["full_gaussian_75"] = evaluate_condition_stream(
            "full_gaussian_75", mask_75, "diagonal_gaussian", seed=11001
        )

        # 5D: Grouped K in {1, 8, 196} at 75% (seed 15001)
        for k in [1, 8, 196]:
            cname = f"grouped_k_{k}_75"
            print(f"  Evaluating 75% {cname}...")
            secondary_75_results[cname] = evaluate_condition_stream(
                cname, mask_75, "grouped", seed=15001, k_val=k
            )

        # Check post-run hash
        _, final_hash = load_deit_model(self.model_name, self.device)
        assert initial_hash == final_hash, "Backbone weights were modified during evaluation!"

        elapsed = time.time() - start_time
        peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
        print(f"[{self.model_name}] Completed in {elapsed:.1f}s | Peak VRAM: {peak_vram_mb:.1f} MB")

        return {
            "model_name": self.model_name,
            "embed_dim": embed_dim,
            "pca_result": pca_result,
            "clean_acc": clean_acc,
            "clean_margin_mean": clean_margin_mean,
            "primary_results": all_results,
            "secondary_75_results": secondary_75_results,
            "all_targets": all_targets,
            "elapsed": elapsed,
            "peak_vram_mb": peak_vram_mb,
            "initial_hash": initial_hash,
            "final_hash": final_hash,
            "max_random_ortho_error": max_rand_ortho_err
        }
