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
from patch_fungibility.v0_8_pca import compute_calibration_pca
from patch_fungibility.v0_9_diagnostics import (
    compute_per_image_representation_ranks,
    forward_block_with_v0_9_diagnostics,
    compute_geometric_expansion_diagnostics
)
from patch_fungibility.v0_9_interventions import (
    apply_zero_to_mask,
    apply_centroid_to_mask,
    apply_perturbation_to_mask,
    generate_natural_pca_rank_eps,
    generate_energy_matched_pca_rank_eps,
    generate_pc1_scale_eps,
    generate_single_pc_eps,
    generate_random_1d_eps,
    generate_diagonal_gaussian_eps,
    generate_isotropic_full_eps,
)


class PatchFungibilityV09Pipeline:
    def __init__(
        self,
        model_name: str = "deit_tiny_patch16_224",
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "outputs/fungibility_v0_9",
        ranks: List[int] = [1, 2, 4, 8, 16, 32, 64],
        rank_seeds: List[int] = [16001, 16002, 16003, 16004, 16005],
        scale_seeds: List[int] = [17001, 17002, 17003, 17004, 17005],
        random_1d_seeds: List[int] = [18001, 18002, 18003, 18004, 18005],
        pc_indices: List[int] = [1, 2, 3, 4, 8, 16],
        scale_multipliers: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0],
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.output_dir = output_dir
        self.target_depth = 8
        self.ranks = ranks
        self.rank_seeds = rank_seeds
        self.scale_seeds = scale_seeds
        self.random_1d_seeds = random_1d_seeds
        self.pc_indices = pc_indices
        self.scale_multipliers = scale_multipliers

        os.makedirs(self.output_dir, exist_ok=True)

    def run(
        self,
        calib_loader: DataLoader,
        eval_loader: DataLoader,
        n_samples: int = 1000
    ) -> Dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.time()
        print(f"\n{'='*70}\n[V0.9 Pipeline] Initializing {self.model_name} on {self.device}\n{'='*70}")
        model, initial_hash = load_deit_model(self.model_name, self.device)
        embed_dim = model.embed_dim

        # -----------------------------------------------------------------
        # STEP 1: Compute/Load PCA on Calibration Data (Strictly N=1,000 images)
        # -----------------------------------------------------------------
        pca_cache_file = os.path.join(self.output_dir, f"pca_cache_{self.model_name}.npz")
        if os.path.exists(pca_cache_file):
            print(f"[{self.model_name}] Loading cached calibration PCA from {pca_cache_file}...")
            pca_npz = np.load(pca_cache_file)
            mu_8_np = pca_npz["mu_8"]
            sigma_8_np = pca_npz["sigma_8"]
            E_full = float(pca_npz["E_full"])
            eigenvalues_np = pca_npz["eigenvalues"]
            eigenvectors_np = pca_npz["eigenvectors"]
            ortho_error = float(pca_npz["ortho_error"])
            pca_result = {
                "model_name": self.model_name,
                "n_tokens": int(pca_npz["n_tokens"]),
                "embed_dim": embed_dim,
                "mu_8": mu_8_np,
                "sigma_8": sigma_8_np,
                "E_full": E_full,
                "eigenvalues": eigenvalues_np,
                "eigenvectors": eigenvectors_np,
                "explained_variance_ratio": (eigenvalues_np / eigenvalues_np.sum()).tolist(),
                "ortho_error": ortho_error
            }
        else:
            print(f"[{self.model_name}] Computing PCA on Block-8 calibration patch representations...")
            pca_result = compute_calibration_pca(
                model=model,
                model_name=self.model_name,
                calib_loader=calib_loader,
                target_depth=self.target_depth,
                device=self.device
            )
            np.savez(
                pca_cache_file,
                mu_8=pca_result["mu_8"],
                sigma_8=pca_result["sigma_8"],
                E_full=pca_result["E_full"],
                eigenvalues=pca_result["eigenvalues"],
                eigenvectors=pca_result["eigenvectors"],
                ortho_error=pca_result["ortho_error"],
                n_tokens=pca_result["n_tokens"]
            )

        mu_8 = torch.from_numpy(pca_result["mu_8"]).to(self.device)
        sigma_8 = torch.from_numpy(pca_result["sigma_8"]).to(self.device)
        E_full = pca_result["E_full"]
        eigenvalues = torch.from_numpy(pca_result["eigenvalues"])  # (D,)
        eigenvectors = torch.from_numpy(pca_result["eigenvectors"])  # (D, D)

        lambda_1 = float(eigenvalues[0].item())
        E_MATCH = math.sqrt(E_full / lambda_1)

        print(f"[{self.model_name}] E_full = {E_full:.4f} | lambda_1 = {lambda_1:.4f} | E_MATCH factor = {E_MATCH:.4f}")

        # -----------------------------------------------------------------
        # STEP 2: Cache Evaluation Block-8 Activations
        # -----------------------------------------------------------------
        print(f"[{self.model_name}] Caching Block-8 evaluation features and clean baseline...")
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

        # Primary mask: all 196 spatial patch tokens (indices 1..196)
        mask_100 = list(range(1, 197))
        M = len(mask_100)

        # -----------------------------------------------------------------
        # STEP 3: Precompute Random 1D Unit Vectors (Seeds 18001..18005)
        # -----------------------------------------------------------------
        random_1d_vectors: Dict[int, torch.Tensor] = {}
        max_u_norm_err = 0.0
        for s in self.random_1d_seeds:
            rng_u = np.random.RandomState(s)
            u_raw = rng_u.randn(embed_dim).astype(np.float32)
            u_norm = np.linalg.norm(u_raw)
            u_unit = u_raw / u_norm
            u_t = torch.from_numpy(u_unit)
            norm_err = abs(float(torch.norm(u_t, p=2).item()) - 1.0)
            max_u_norm_err = max(max_u_norm_err, norm_err)
            random_1d_vectors[s] = u_t

        # -----------------------------------------------------------------
        # Helper: Unified Condition Evaluator
        # -----------------------------------------------------------------
        def evaluate_condition_stream(
            cond_name: str,
            gen_type: str,
            seed: Optional[int] = None,
            rank: Optional[int] = None,
            scale: Optional[float] = None,
            pc_idx: Optional[int] = None,
            is_key_diagnostic: bool = False
        ) -> Dict[str, Any]:
            num_batches = (n_eval + self.batch_size - 1) // self.batch_size
            cond_preds: List[int] = []
            cond_corrects: List[int] = []
            cond_margins: List[float] = []

            total_pert_energy = 0.0
            total_tokens_perturbed = 0
            patch_norms: List[float] = []

            # If key diagnostic, track per-image representation ranks across blocks
            rep_diags_per_image = {d: [] for d in [8, 9, 10, 11]} if is_key_diagnostic else {}
            attn_diags = {d: {"key_variance": [], "value_variance": [], "cls_attention_entropy": [], "attention_stable_rank": []}
                          for d in [9, 10, 11]} if is_key_diagnostic else {}
            geom_metrics_list = [] if (is_key_diagnostic and gen_type in ["pc1_natural", "pc1_scale"] and scale == 1.0) else None

            rng = torch.Generator(device="cpu").manual_seed(seed if seed is not None else 42)

            with torch.no_grad():
                for b_idx in range(num_batches):
                    b_start = b_idx * self.batch_size
                    b_end = min(b_start + self.batch_size, n_eval)
                    B_curr = b_end - b_start

                    h_batch = eval_h8[b_start:b_end].to(self.device)
                    b_targets = targets_tensor[b_start:b_end].to(self.device)
                    z_raw = None

                    if gen_type == "clean":
                        h_mod = h_batch
                    elif gen_type == "centroid":
                        h_mod = apply_centroid_to_mask(h_batch, mask_100, mu_8)
                    elif gen_type == "diagonal_gaussian":
                        eps = generate_diagonal_gaussian_eps(B_curr, M, sigma_8, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, mask_100, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                        patch_norms.extend(torch.norm(h_mod[:, 1:, :], p=2, dim=-1).mean(dim=-1).cpu().numpy().tolist())
                    elif gen_type == "isotropic_full":
                        eps = generate_isotropic_full_eps(B_curr, M, embed_dim, E_full, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, mask_100, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                        patch_norms.extend(torch.norm(h_mod[:, 1:, :], p=2, dim=-1).mean(dim=-1).cpu().numpy().tolist())
                    elif gen_type == "natural_pca":
                        assert rank is not None
                        V_R = eigenvectors[:, :rank]
                        e_R = eigenvalues[:rank]
                        eps = generate_natural_pca_rank_eps(B_curr, M, V_R, e_R, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, mask_100, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                        patch_norms.extend(torch.norm(h_mod[:, 1:, :], p=2, dim=-1).mean(dim=-1).cpu().numpy().tolist())
                    elif gen_type == "energy_matched_pca":
                        assert rank is not None
                        V_R = eigenvectors[:, :rank]
                        e_R = eigenvalues[:rank]
                        eps = generate_energy_matched_pca_rank_eps(B_curr, M, V_R, e_R, E_full, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, mask_100, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                        patch_norms.extend(torch.norm(h_mod[:, 1:, :], p=2, dim=-1).mean(dim=-1).cpu().numpy().tolist())
                    elif gen_type == "pc1_scale":
                        assert scale is not None
                        v_1 = eigenvectors[:, 0]
                        eps, z_raw = generate_pc1_scale_eps(B_curr, M, v_1, lambda_1, scale, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, mask_100, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                        patch_norms.extend(torch.norm(h_mod[:, 1:, :], p=2, dim=-1).mean(dim=-1).cpu().numpy().tolist())
                    elif gen_type == "single_pc_natural":
                        assert pc_idx is not None
                        v_k = eigenvectors[:, pc_idx - 1]
                        var_k = float(eigenvalues[pc_idx - 1].item())
                        eps = generate_single_pc_eps(B_curr, M, v_k, var_k, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, mask_100, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                        patch_norms.extend(torch.norm(h_mod[:, 1:, :], p=2, dim=-1).mean(dim=-1).cpu().numpy().tolist())
                    elif gen_type == "single_pc_matched":
                        assert pc_idx is not None
                        v_k = eigenvectors[:, pc_idx - 1]
                        var_k = lambda_1  # norm-controlled to natural PC1 energy
                        eps = generate_single_pc_eps(B_curr, M, v_k, var_k, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, mask_100, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                        patch_norms.extend(torch.norm(h_mod[:, 1:, :], p=2, dim=-1).mean(dim=-1).cpu().numpy().tolist())
                    elif gen_type == "random_1d":
                        assert seed is not None
                        u = random_1d_vectors[seed]
                        eps = generate_random_1d_eps(B_curr, M, u, lambda_1, rng, self.device)
                        h_mod = apply_perturbation_to_mask(h_batch, mask_100, mu_8, eps)
                        total_pert_energy += float((eps ** 2).sum().item())
                        total_tokens_perturbed += B_curr * M
                        patch_norms.extend(torch.norm(h_mod[:, 1:, :], p=2, dim=-1).mean(dim=-1).cpu().numpy().tolist())
                    else:
                        raise ValueError(f"Unknown gen_type: {gen_type}")

                    # Forward pass
                    h_curr = h_mod
                    if is_key_diagnostic:
                        # Depth 8 rank per image
                        r_d8 = compute_per_image_representation_ranks(h_mod[:, 1:, :])
                        rep_diags_per_image[8].append(r_d8)

                        for b in range(self.target_depth + 1, len(model.blocks)):
                            h_curr, a_diag = forward_block_with_v0_9_diagnostics(model.blocks[b], h_curr, compute_attn_svd=True)
                            for k_diag, v_val in a_diag.items():
                                attn_diags[b][k_diag].append(v_val * B_curr)
                            r_db = compute_per_image_representation_ranks(h_curr[:, 1:, :])
                            rep_diags_per_image[b].append(r_db)

                        if geom_metrics_list is not None and z_raw is not None:
                            v_1 = eigenvectors[:, 0]
                            geom_res = compute_geometric_expansion_diagnostics(h_mod, h_curr, v_1, z_raw)
                            geom_metrics_list.append(geom_res)
                    else:
                        for b in range(self.target_depth + 1, len(model.blocks)):
                            h_curr = model.blocks[b](h_curr)

                    logits = model.forward_head(model.norm(h_curr))
                    preds, corrects, _, _, margins = compute_logits_and_margins(logits, b_targets)

                    cond_preds.extend(preds.cpu().numpy().tolist())
                    cond_corrects.extend(corrects.cpu().numpy().tolist())
                    cond_margins.extend(margins.cpu().numpy().tolist())

            # Aggregate diagnostics if key condition
            agg_rep_diags = {}
            if is_key_diagnostic:
                for b_depth in [8, 9, 10, 11]:
                    keys = rep_diags_per_image[b_depth][0].keys()
                    agg_rep_diags[b_depth] = {}
                    for k in keys:
                        all_vals = np.concatenate([batch_dict[k] for batch_dict in rep_diags_per_image[b_depth]])
                        agg_rep_diags[b_depth][k] = float(np.mean(all_vals))
                        agg_rep_diags[b_depth][f"{k}_std"] = float(np.std(all_vals))

            agg_attn_diags = {}
            if is_key_diagnostic:
                for b_depth in [9, 10, 11]:
                    agg_attn_diags[b_depth] = {
                        k: float(np.sum(vals) / n_eval) for k, vals in attn_diags[b_depth].items()
                    }

            mean_realized_energy = float(total_pert_energy / max(total_tokens_perturbed, 1))
            mean_patch_norm = float(np.mean(patch_norms)) if patch_norms else float(torch.norm(mu_8, p=2).item())

            geom_summary = {}
            if geom_metrics_list:
                for gk in geom_metrics_list[0].keys():
                    geom_summary[gk] = float(np.mean([item[gk] for item in geom_metrics_list]))

            return {
                "cond_name": cond_name,
                "gen_type": gen_type,
                "rank": rank,
                "scale": scale,
                "pc_idx": pc_idx,
                "seed": seed,
                "acc": float(np.mean(cond_corrects)),
                "mean_margin": float(np.mean(cond_margins)),
                "preds": cond_preds,
                "corrects": cond_corrects,
                "margins": cond_margins,
                "mean_realized_energy": mean_realized_energy,
                "mean_patch_norm": mean_patch_norm,
                "representation_diagnostics": agg_rep_diags,
                "attention_diagnostics": agg_attn_diags,
                "geometric_expansion": geom_summary
            }

        # -----------------------------------------------------------------
        # STEP 4: Execute Conditions
        # -----------------------------------------------------------------
        all_results: Dict[str, Dict[str, Any]] = {}

        # 4A. References
        print(f"[{self.model_name}] Evaluating reference conditions...")
        all_results["clean"] = evaluate_condition_stream("clean", "clean", is_key_diagnostic=True)
        all_results["static_centroid"] = evaluate_condition_stream("static_centroid", "centroid", is_key_diagnostic=True)
        all_results["diagonal_gaussian_seed_16001"] = evaluate_condition_stream(
            "diagonal_gaussian_seed_16001", "diagonal_gaussian", seed=16001, is_key_diagnostic=True
        )
        all_results["isotropic_full_seed_16001"] = evaluate_condition_stream(
            "isotropic_full_seed_16001", "isotropic_full", seed=16001, is_key_diagnostic=False
        )

        # 4B. Natural PCA Rank Sweep (R in 1..64, seeds 16001..16005)
        print(f"[{self.model_name}] Evaluating Natural PCA Rank Sweep...")
        for r in self.ranks:
            for s in self.rank_seeds:
                cname = f"natural_pca_rank_{r}_seed_{s}"
                is_key = (s == 16001 and r in [1, 2, 4])
                all_results[cname] = evaluate_condition_stream(
                    cname, "natural_pca", seed=s, rank=r, is_key_diagnostic=is_key
                )

        # 4C. Energy-Matched PCA Rank Sweep (R in 1..64, seeds 16001..16005)
        print(f"[{self.model_name}] Evaluating Energy-Matched PCA Rank Sweep...")
        for r in self.ranks:
            for s in self.rank_seeds:
                cname = f"energy_matched_pca_rank_{r}_seed_{s}"
                is_key = (s == 16001 and r == 1)
                all_results[cname] = evaluate_condition_stream(
                    cname, "energy_matched_pca", seed=s, rank=r, is_key_diagnostic=is_key
                )

        # 4D. PC1 Amplitude Sweep (scales in [0.25, 0.5, 1.0, 2.0, 4.0, E_MATCH], seeds 17001..17005)
        print(f"[{self.model_name}] Evaluating PC1 Amplitude Sweep...")
        sweep_scales = [(sc, f"{sc:.2f}") for sc in self.scale_multipliers] + [(E_MATCH, "E_MATCH")]
        for sc_val, sc_label in sweep_scales:
            for s in self.scale_seeds:
                cname = f"pc1_scale_{sc_label}_seed_{s}"
                all_results[cname] = evaluate_condition_stream(
                    cname, "pc1_scale", seed=s, scale=sc_val, is_key_diagnostic=(s == 17001 and sc_label == "1.00")
                )

        # 4E. PC Identity Test (k in [1, 2, 3, 4, 8, 16], seeds 17001..17005)
        print(f"[{self.model_name}] Evaluating PC Identity Test...")
        for k in self.pc_indices:
            for s in self.scale_seeds:
                # Natural PC-k
                cname_nat = f"natural_pc_{k}_seed_{s}"
                all_results[cname_nat] = evaluate_condition_stream(
                    cname_nat, "single_pc_natural", seed=s, pc_idx=k, is_key_diagnostic=False
                )
                # PC1-Energy-Matched PC-k
                cname_mat = f"matched_pc_{k}_seed_{s}"
                all_results[cname_mat] = evaluate_condition_stream(
                    cname_mat, "single_pc_matched", seed=s, pc_idx=k, is_key_diagnostic=False
                )

        # 4F. Matched Random 1D Direction Control (seeds 18001..18005)
        print(f"[{self.model_name}] Evaluating Matched Random 1D Directions...")
        for s in self.random_1d_seeds:
            cname = f"random_1d_seed_{s}"
            all_results[cname] = evaluate_condition_stream(
                cname, "random_1d", seed=s, is_key_diagnostic=False
            )

        # Post-run hash check
        _, final_hash = load_deit_model(self.model_name, self.device)
        assert initial_hash == final_hash, "Backbone parameter hash modified during run!"

        elapsed = time.time() - start_time
        peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
        print(f"[{self.model_name}] Completed in {elapsed:.1f}s | Peak VRAM: {peak_vram_mb:.1f} MB")

        return {
            "model_name": self.model_name,
            "embed_dim": embed_dim,
            "pca_result": pca_result,
            "lambda_1": lambda_1,
            "E_MATCH": E_MATCH,
            "max_u_norm_err": max_u_norm_err,
            "clean_acc": clean_acc,
            "clean_margin_mean": clean_margin_mean,
            "primary_results": all_results,
            "all_targets": all_targets,
            "elapsed": elapsed,
            "peak_vram_mb": peak_vram_mb,
            "initial_hash": initial_hash,
            "final_hash": final_hash
        }
