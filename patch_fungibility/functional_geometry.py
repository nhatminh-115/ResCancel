"""
patch_fungibility/functional_geometry.py

Comprehensive implementation of the Functional Equivalence Geometry Pipeline:
1. Direction library generation: Natural PCA eigenvectors, Isotropic Random directions,
   Downstream Functional Jacobian directions, and Canonical Intervention displacements.
2. Finite-radius perturbation sweep across calibrated scales:
   s in {0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2} * sigma_l.
3. Observables: Margin damage, Logit distance, Prediction flip, KL divergence.
4. Tolerance radius estimation r_l(v; epsilon) and directional fungibility spectrum.
5. Functional metric matrix M_l = E[J^T J], eigenspectrum, and near-null subspace dimensionality.
6. Covariance-functional alignment: Principal angles, projection of PCs onto M_l, and relative spectrum.
7. Decomposition of existing interventions (centroid, perm, sign-flip, PC1, random) into sensitive vs fungible subspaces.
8. Image-wise consistency and token position heterogeneity.
9. Constructive test: Functional-geometry surrogate vs centroid vs Gaussian.
10. Generates publication figures (Figures 1-6) and machine-readable output CSVs.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from patch_fungibility.dense_fraction_models import load_model_and_transform, forward_block_by_block
from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits


# ==============================================================================
# 1. PERTURBATION HOOK & DIRECTION LIBRARY
# ==============================================================================

class DirectionalPerturbationHook:
    """
    Hook that adds a finite perturbation along direction v to selected patch tokens.
    """
    def __init__(
        self,
        target_depth: int,
        direction: torch.Tensor,       # (D,) unit vector on device
        amplitude: float,              # scalar alpha
        mask: Optional[torch.Tensor] = None # (N,) boolean mask or None (all patches)
    ):
        self.target_depth = target_depth
        self.direction = direction / (direction.norm() + 1e-10)
        self.amplitude = amplitude
        self.mask = mask

    def __call__(self, depth: int, h: torch.Tensor) -> torch.Tensor:
        if depth != self.target_depth or self.amplitude == 0.0:
            return h
        
        # h: (B, 1 + N, D)
        h_mod = h.clone()
        cls_tok = h_mod[:, 0:1, :]
        patches = h_mod[:, 1:, :] # (B, N, D)
        
        pert = self.amplitude * self.direction.view(1, 1, -1) # (1, 1, D)
        
        if self.mask is not None:
            # mask: (N,) boolean
            patches[:, self.mask, :] = patches[:, self.mask, :] + pert
        else:
            patches = patches + pert
            
        h_mod[:, 1:, :] = patches
        return h_mod


def construct_direction_library(
    cov_matrix: torch.Tensor,      # (D, D) calibration covariance
    centroid: torch.Tensor,        # (D,) calibration centroid
    mean_grad: torch.Tensor,       # (D,) mean margin gradient direction
    metric_matrix: torch.Tensor,   # (D, D) functional metric M_l
    embed_dim: int,
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Constructs an explicit, standardized library of unit probe directions:
    - Family A: Natural PCA eigenvectors (PC1, PC2, PC4, PC_mid, PC_tail, PC_bottom)
    - Family B: Isotropic Random orthonormal directions
    - Family C: Downstream Jacobian directions (Margin Grad, Top Jacobian, Near-Null Jacobian)
    - Family D: Canonical Intervention displacements (Centroid, Permutation, Sign-Flip, Random 1D)
    """
    directions = {}
    device = cov_matrix.device

    # Family A: Natural PCA of Sigma_l
    evals_cov, evecs_cov = torch.linalg.eigh(cov_matrix)
    # Sort descending
    idx_cov = torch.argsort(evals_cov, descending=True)
    evecs_cov = evecs_cov[:, idx_cov] # columns are eigenvectors

    directions["pc1"] = evecs_cov[:, 0]
    directions["pc2"] = evecs_cov[:, 1]
    directions["pc4"] = evecs_cov[:, min(3, embed_dim - 1)]
    directions["pc_mid"] = evecs_cov[:, embed_dim // 2]
    directions["pc_tail"] = evecs_cov[:, min(embed_dim * 3 // 4, embed_dim - 1)]
    directions["pc_bottom"] = evecs_cov[:, -1]

    # Family B: Isotropic Random directions (mutually orthonormalized via QR)
    torch.manual_seed(seed)
    rand_mat = torch.randn(embed_dim, 4, device=device)
    q_rand, _ = torch.linalg.qr(rand_mat)
    for i in range(4):
        directions[f"rand_{i+1}"] = q_rand[:, i]

    # Family C: Downstream Functional Geometry (Jacobian of M_l)
    if mean_grad.norm() > 1e-8:
        directions["grad_margin"] = mean_grad / mean_grad.norm()
    else:
        directions["grad_margin"] = directions["rand_1"]

    evals_m, evecs_m = torch.linalg.eigh(metric_matrix)
    idx_m = torch.argsort(evals_m, descending=True)
    evecs_m = evecs_m[:, idx_m]

    directions["jac_top"] = evecs_m[:, 0]      # Top functional sensitivity direction
    directions["jac_null"] = evecs_m[:, -1]    # Lowest functional sensitivity / near-null direction

    # Family D: Canonical Intervention Displacements
    if centroid.norm() > 1e-8:
        c_unit = centroid / centroid.norm()
        directions["centroid_dir"] = c_unit
        directions["sign_flip_disp"] = -c_unit  # Displacement direction for sign flip (-mu - mu)

        # Coordinate permutation displacement (permute coordinates)
        torch.manual_seed(seed + 100)
        perm = torch.randperm(embed_dim)
        c_perm = centroid[perm]
        disp_perm = c_perm - centroid
        if disp_perm.norm() > 1e-8:
            directions["perm_disp"] = disp_perm / disp_perm.norm()
        else:
            directions["perm_disp"] = directions["rand_2"]
    else:
        directions["centroid_dir"] = directions["rand_1"]
        directions["sign_flip_disp"] = -directions["rand_1"]
        directions["perm_disp"] = directions["rand_2"]

    # Verify unit norms
    for k, v in directions.items():
        directions[k] = v / (v.norm() + 1e-10)

    return directions


# ==============================================================================
# 2. FUNCTIONAL METRIC ESTIMATION (M_l = E[J^T J])
# ==============================================================================

def estimate_functional_metric(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes local functional metric M_l = (1 / (B * N)) * sum_{i, k} g_{i, k} g_{i, k}^T
    and average margin gradient vector g_mean in R^D.
    """
    B = images.size(0)
    
    with torch.no_grad():
        clean_logits, collected = forward_block_by_block(
            model, model_key, x=images, collect_depths=(depth,)
        )
        top_class = clean_logits.argmax(dim=-1)
        logits_temp = clean_logits.clone()
        logits_temp[torch.arange(B), top_class] = -1e9
        runner_up = logits_temp.argmax(dim=-1)

    h_d = collected[depth] # (B, 1 + N, D)
    h_leaf = h_d.clone().detach().requires_grad_(True)

    logits_down, _ = forward_block_by_block(
        model, model_key, start_depth=depth, h_start=h_leaf
    )
    margins = logits_down[torch.arange(B), top_class] - logits_down[torch.arange(B), runner_up]
    sum_m = margins.sum()

    grad_h = torch.autograd.grad(sum_m, h_leaf)[0]
    grad_p = grad_h[:, 1:, :] # (B, N, D)
    
    # Reshape to (B * N, D)
    grad_flat = grad_p.reshape(-1, grad_p.size(-1)).to(torch.float64) # (M, D)
    M_l = (grad_flat.t() @ grad_flat) / grad_flat.size(0) # (D, D)
    g_mean = grad_flat.mean(dim=0).float() # (D,)

    return M_l.float(), g_mean


# ==============================================================================
# 3. DIRECTIONAL TOLERANCE SWEEP
# ==============================================================================

def evaluate_directional_tolerance_profile(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    direction_library: Dict[str, torch.Tensor],
    sigma_l: float,
    scale_grid: List[float],
    device: torch.device,
    mask_fraction: float = 0.25,
    epsilon_margin: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sweeps finite perturbations along each probe direction in direction_library across scale_grid.
    Returns:
    1. detailed_curves_df: observable values (margin damage, logit dist, flip rate, KL) for all (dir, scale)
    2. tolerance_radii_df: estimated tolerance radius r(v; epsilon) for each direction
    """
    B = images.size(0)
    num_patches = 256 if model_key == "dinov2" else 196
    k_mask = int(round(num_patches * mask_fraction))

    # Deterministic spatial mask (central/contiguous patch subset)
    torch.manual_seed(9101)
    perm_patches = torch.randperm(num_patches)
    mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    mask[perm_patches[:k_mask]] = True

    # 1. Clean forward reference
    with torch.no_grad():
        clean_logits, clean_collected = forward_block_by_block(
            model, model_key, x=images, collect_depths=(depth,)
        )
        h_clean_depth = clean_collected[depth]
        top_class = clean_logits.argmax(dim=-1)
        logits_temp = clean_logits.clone()
        logits_temp[torch.arange(B), top_class] = -1e9
        runner_up = logits_temp.argmax(dim=-1)
        clean_margins = clean_logits[torch.arange(B), top_class] - clean_logits[torch.arange(B), runner_up]
        clean_mean_margin = float(clean_margins.mean().item())
        clean_probs = F.softmax(clean_logits, dim=-1)

    curve_records = []
    radius_records = []

    # Iterate over all directions
    for dir_name, v in direction_library.items():
        v = v.to(device)
        damages = []
        scales_evaluated = []

        for s in scale_grid:
            alpha = s * sigma_l

            hook = DirectionalPerturbationHook(
                target_depth=depth,
                direction=v,
                amplitude=alpha,
                mask=mask
            )
            h_pert = hook(depth, h_clean_depth)

            with torch.no_grad():
                pert_logits, _ = forward_block_by_block(
                    model, model_key, start_depth=depth, h_start=h_pert
                )
                pert_margins = pert_logits[torch.arange(B), top_class] - pert_logits[torch.arange(B), runner_up]
                margin_damage = float((clean_margins - pert_margins).mean().item())

                # Relative logit distance
                logit_dist = float(((pert_logits - clean_logits).norm(dim=-1) / (clean_logits.norm(dim=-1) + 1e-10)).mean().item())

                # Top-1 flip rate
                pert_preds = pert_logits.argmax(dim=-1)
                flip_rate = float((pert_preds != top_class).float().mean().item())

                # KL divergence
                pert_log_probs = F.log_softmax(pert_logits, dim=-1)
                kl_div = float(F.kl_div(pert_log_probs, clean_probs, reduction='batchmean').item())

            curve_records.append({
                "model_key": model_key,
                "depth": depth,
                "direction": dir_name,
                "scale_s": s,
                "amplitude_alpha": alpha,
                "margin_damage": margin_damage,
                "relative_logit_dist": logit_dist,
                "flip_rate": flip_rate,
                "kl_div": kl_div
            })

            damages.append(margin_damage)
            scales_evaluated.append(s)

        # Estimate tolerance radius r(v; epsilon) via linear interpolation
        r_tol = scales_evaluated[-1] # Default to max scale if never exceeded
        for idx in range(len(scales_evaluated) - 1):
            s0, s1 = scales_evaluated[idx], scales_evaluated[idx + 1]
            d0, d1 = damages[idx], damages[idx + 1]
            if d0 <= epsilon_margin and d1 > epsilon_margin:
                # Linear interpolation
                if abs(d1 - d0) > 1e-8:
                    r_tol = s0 + (epsilon_margin - d0) / (d1 - d0) * (s1 - s0)
                else:
                    r_tol = s0
                break
            elif d0 > epsilon_margin and idx == 0:
                # Already damaged at s0
                r_tol = 0.0
                break

        radius_records.append({
            "model_key": model_key,
            "depth": depth,
            "direction": dir_name,
            "tolerance_radius_s": float(r_tol),
            "tolerance_radius_alpha": float(r_tol * sigma_l),
            "max_damage": float(max(damages)),
            "damage_at_s1": float(damages[scale_grid.index(1.0)] if 1.0 in scale_grid else damages[-1])
        })

    curves_df = pd.DataFrame(curve_records)
    radii_df = pd.DataFrame(radius_records)
    return curves_df, radii_df


# ==============================================================================
# 4. CONSTRUCTIVE TEST: FUNCTIONAL-GEOMETRY SURROGATE
# ==============================================================================

def evaluate_constructive_surrogates(
    model: nn.Module,
    model_key: str,
    depth: int,
    images: torch.Tensor,
    cov_matrix: torch.Tensor,
    centroid: torch.Tensor,
    metric_matrix: torch.Tensor,
    device: torch.device,
    mask_fraction: float = 0.25,
    subspace_dim: int = 16,
    seed: int = 42
) -> pd.DataFrame:
    """
    Constructs and evaluates candidate surrogate distributions:
    1. Static Centroid mu_l
    2. Diagonal Gaussian N(mu_l, diag(Sigma_l))
    3. Fungible Subspace Gaussian: mu_l + U_fungible z (sampling only in near-null subspace of M_l)
    4. Sensitive Subspace Gaussian: mu_l + U_sensitive z (sampling in top sensitivity subspace of M_l)
    5. Random Subspace Gaussian: mu_l + U_random z
    """
    B = images.size(0)
    embed_dim = cov_matrix.size(0)
    num_patches = 256 if model_key == "dinov2" else 196
    k_mask = int(round(num_patches * mask_fraction))

    torch.manual_seed(9101)
    perm_patches = torch.randperm(num_patches)
    mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    mask[perm_patches[:k_mask]] = True

    # Clean baseline
    with torch.no_grad():
        clean_logits, clean_collected = forward_block_by_block(
            model, model_key, x=images, collect_depths=(depth,)
        )
        h_clean = clean_collected[depth]
        top_class = clean_logits.argmax(dim=-1)
        logits_temp = clean_logits.clone()
        logits_temp[torch.arange(B), top_class] = -1e9
        runner_up = logits_temp.argmax(dim=-1)
        clean_margins = clean_logits[torch.arange(B), top_class] - clean_logits[torch.arange(B), runner_up]

    # Decompose M_l into eigenvectors
    evals_m, evecs_m = torch.linalg.eigh(metric_matrix)
    idx_m = torch.argsort(evals_m, descending=True)
    evecs_m = evecs_m[:, idx_m]

    U_sensitive = evecs_m[:, :subspace_dim]      # Top sensitive directions
    U_fungible = evecs_m[:, -subspace_dim:]     # Bottom near-null directions

    # Random subspace
    torch.manual_seed(seed + 77)
    rand_mat = torch.randn(embed_dim, subspace_dim, device=device)
    U_random, _ = torch.linalg.qr(rand_mat)

    # Variances from covariance matrix projected onto subspaces
    var_fungible = torch.diag(U_fungible.t() @ cov_matrix @ U_fungible).clamp(min=1e-8)
    var_sensitive = torch.diag(U_sensitive.t() @ cov_matrix @ U_sensitive).clamp(min=1e-8)
    var_random = torch.diag(U_random.t() @ cov_matrix @ U_random).clamp(min=1e-8)
    diag_cov = torch.diag(cov_matrix).clamp(min=1e-8)

    torch.manual_seed(seed + 999)
    # Generate surrogate token sets: shape (B, k_mask, D)
    surrogate_conditions = {}

    # 1. Zero
    surrogate_conditions["zero"] = torch.zeros(B, k_mask, embed_dim, device=device)

    # 2. Static Centroid
    surrogate_conditions["centroid"] = centroid.view(1, 1, embed_dim).expand(B, k_mask, embed_dim).clone()

    # 3. Diagonal Gaussian
    z_diag = torch.randn(B, k_mask, embed_dim, device=device) * torch.sqrt(diag_cov).view(1, 1, embed_dim)
    surrogate_conditions["diagonal_gaussian"] = centroid.view(1, 1, embed_dim) + z_diag

    # 4. Fungible Subspace Gaussian
    z_f = torch.randn(B, k_mask, subspace_dim, device=device) * torch.sqrt(var_fungible).view(1, 1, subspace_dim)
    pert_f = torch.matmul(z_f, U_fungible.t()) # (B, k_mask, D)
    surrogate_conditions["fungible_subspace_gaussian"] = centroid.view(1, 1, embed_dim) + pert_f

    # 5. Sensitive Subspace Gaussian
    z_s = torch.randn(B, k_mask, subspace_dim, device=device) * torch.sqrt(var_sensitive).view(1, 1, subspace_dim)
    pert_s = torch.matmul(z_s, U_sensitive.t())
    surrogate_conditions["sensitive_subspace_gaussian"] = centroid.view(1, 1, embed_dim) + pert_s

    # 6. Random Subspace Gaussian
    z_r = torch.randn(B, k_mask, subspace_dim, device=device) * torch.sqrt(var_random).view(1, 1, subspace_dim)
    pert_r = torch.matmul(z_r, U_random.t())
    surrogate_conditions["random_subspace_gaussian"] = centroid.view(1, 1, embed_dim) + pert_r

    results = []
    for cond_name, surr_tokens in surrogate_conditions.items():
        h_mod = h_clean.clone()
        h_mod[:, 1:, :][:, mask, :] = surr_tokens

        with torch.no_grad():
            out_logits, _ = forward_block_by_block(
                model, model_key, start_depth=depth, h_start=h_mod
            )
            out_margins = out_logits[torch.arange(B), top_class] - out_logits[torch.arange(B), runner_up]
            damage = float((clean_margins - out_margins).mean().item())
            preds = out_logits.argmax(dim=-1)
            acc = float((preds == top_class).float().mean().item())

            results.append({
                "model_key": model_key,
                "depth": depth,
                "condition": cond_name,
                "margin_damage": damage,
                "top1_accuracy": acc,
                "recovery_vs_zero": float((results[0]["margin_damage"] - damage) / (results[0]["margin_damage"] + 1e-10)) if len(results) > 0 else 0.0
            })

    return pd.DataFrame(results)


# ==============================================================================
# 5. MASTER EXECUTION PIPELINE
# ==============================================================================

def run_functional_geometry_experiment(
    n_images: int = 100,
    models_to_evaluate: Tuple[str, ...] = ("deit_small", "vit_base"),
    depths_to_evaluate: Tuple[int, ...] = (5, 7, 8, 10),
    output_dir: str = os.path.join("outputs", "fungibility_functional_geometry"),
    figures_dir: str = os.path.join("figures", "fungibility_functional_geometry")
):
    """
    Executes the comprehensive functional geometry mapping across models and depths.
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Functional Geometry Experiment on {device} (N={n_images} images)")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    scale_grid = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

    all_curves = []
    all_radii = []
    all_metrics = []
    all_alignment = []
    all_consistency = []
    all_interventions = []
    all_surrogates = []

    for m_key in models_to_evaluate:
        print(f"\n=======================================================")
        print(f"Mapping Functional Geometry for: {m_key}")
        print(f"=======================================================")

        model, transform, meta = load_model_and_transform(m_key, device)
        embed_dim = meta["embed_dim"]
        num_patches = meta["num_patches"]

        calib_ds, _, _, _ = get_disjoint_imagenet_splits()
        calib_ds.transform = transform
        calib_subset = torch.utils.data.Subset(calib_ds, list(range(n_images)))
        loader = DataLoader(calib_subset, batch_size=25, shuffle=False)

        # Load all images into tensor on device for fast sweeps
        img_batches, label_batches = [], []
        for x_b, y_b in loader:
            img_batches.append(x_b)
            label_batches.append(y_b)
        all_imgs = torch.cat(img_batches, dim=0).to(device)
        print(f"Loaded {all_imgs.size(0)} images for model {m_key}.")

        for depth in depths_to_evaluate:
            print(f"\n--- Depth {depth} ({m_key}) ---")

            # 1. Compute covariance matrix Sigma_l and centroid mu_l
            with torch.no_grad():
                _, coll = forward_block_by_block(model, m_key, x=all_imgs, collect_depths=(depth,))
                h_depth = coll[depth]
                patches = h_depth[:, 1:, :].reshape(-1, embed_dim).to(torch.float64) # (M, D)
                centroid = patches.mean(dim=0).float()
                centered = patches - centroid.double()
                cov_matrix = ((centered.t() @ centered) / (patches.size(0) - 1)).float()
                sigma_norm = float(torch.sqrt(torch.trace(cov_matrix)).item())
                sigma_l = sigma_norm

            # 2. Estimate Functional Metric M_l = E[J^T J] and mean gradient
            metric_matrix, mean_grad = estimate_functional_metric(
                model, m_key, depth, all_imgs, device
            )

            # Analyze eigenspectrum of M_l
            evals_m, _ = torch.linalg.eigh(metric_matrix)
            evals_m = torch.clamp(evals_m, min=0.0).flip(dims=[0])
            total_sens = evals_m.sum().item()
            null_count = int((evals_m <= 1e-3 * evals_m[0]).sum().item())
            null_frac = null_count / embed_dim

            if total_sens > 0:
                qm = evals_m / total_sens
                qm_nonzero = qm[qm > 1e-12]
                r_eff_m = float(np.exp(-(qm_nonzero * torch.log(qm_nonzero)).sum().item()))
            else:
                r_eff_m = 1.0

            all_metrics.append({
                "model_key": m_key,
                "depth": depth,
                "trace_metric_M": total_sens,
                "effective_rank_M": r_eff_m,
                "top_eigenvalue_M": float(evals_m[0].item()),
                "median_eigenvalue_M": float(evals_m[embed_dim // 2].item()),
                "min_eigenvalue_M": float(evals_m[-1].item()),
                "null_subspace_dimension": null_count,
                "null_subspace_fraction": null_frac,
                "cov_trace_Sigma": float(torch.trace(cov_matrix).item()),
                "sigma_l": sigma_l
            })

            # 3. Construct Direction Library
            dir_library = construct_direction_library(
                cov_matrix, centroid, mean_grad, metric_matrix, embed_dim
            )

            # 4. Directional Tolerance Sweep
            curves_df, radii_df = evaluate_directional_tolerance_profile(
                model, m_key, depth, all_imgs, dir_library, sigma_l, scale_grid, device
            )
            all_curves.append(curves_df)
            all_radii.append(radii_df)

            # Summary metrics of the tolerance spectrum
            radii_vals = radii_df["tolerance_radius_s"].values
            anisotropy = float(np.max(radii_vals) / (np.min(radii_vals) + 1e-6))
            fdim = int((radii_vals >= 0.80).sum())
            fdim_norm = float(fdim / len(radii_vals))
            print(f"  Fungibility Spectrum: Median r={np.median(radii_vals):.2f}, Min={np.min(radii_vals):.2f}, Max={np.max(radii_vals):.2f}, Anisotropy={anisotropy:.1f}, FDim={fdim}/{len(radii_vals)}")

            # 5. Natural Covariance vs Functional Geometry Alignment
            evals_cov, evecs_cov = torch.linalg.eigh(cov_matrix)
            evals_cov = evals_cov.flip(dims=[0])
            evecs_cov = evecs_cov[:, torch.argsort(evals_cov, descending=True)]

            # Eigenspectrum of Sigma^{1/2} M Sigma^{1/2}
            try:
                L_cov = torch.linalg.cholesky(cov_matrix + 1e-6 * torch.eye(embed_dim, device=device))
                sigma_sqrt = L_cov
                gen_mat = sigma_sqrt.t() @ metric_matrix @ sigma_sqrt
                gen_evals, _ = torch.linalg.eigh(gen_mat)
                gen_evals = gen_evals.flip(dims=[0])
                tr_sigma_m = float(gen_evals.sum().item())
            except Exception:
                tr_sigma_m = float(torch.trace(cov_matrix @ metric_matrix).item())

            # Check projection of PC1, PC_mid, PC_bottom onto M_l
            sens_pc1 = float((evecs_cov[:, 0].t() @ metric_matrix @ evecs_cov[:, 0]).item())
            sens_pcmid = float((evecs_cov[:, embed_dim // 2].t() @ metric_matrix @ evecs_cov[:, embed_dim // 2]).item())
            sens_pcbot = float((evecs_cov[:, -1].t() @ metric_matrix @ evecs_cov[:, -1]).item())

            all_alignment.append({
                "model_key": m_key,
                "depth": depth,
                "tr_Sigma_M": tr_sigma_m,
                "sensitivity_PC1": sens_pc1,
                "sensitivity_PC_mid": sens_pcmid,
                "sensitivity_PC_bottom": sens_pcbot,
                "ratio_PC1_to_PCbot": float(sens_pc1 / (sens_pcbot + 1e-10)),
                "anisotropy_ratio": anisotropy,
                "fungible_dimension": fdim,
                "fungible_dim_fraction": fdim_norm
            })

            # 6. Decompose Existing Interventions into Sensitive vs Fungible Subspaces
            U_sensitive = evecs_cov[:, :embed_dim // 4] # Top 25% sensitivity
            U_fungible = evecs_cov[:, embed_dim // 4:]  # Bottom 75%
            
            for d_name in ["centroid_dir", "perm_disp", "sign_flip_disp", "pc1", "rand_1"]:
                v_vec = dir_library[d_name]
                proj_sens = float((v_vec.t() @ metric_matrix @ v_vec).item())
                r_val = float(radii_df[radii_df["direction"] == d_name]["tolerance_radius_s"].iloc[0])
                all_interventions.append({
                    "model_key": m_key,
                    "depth": depth,
                    "intervention": d_name,
                    "functional_curvature_vMv": proj_sens,
                    "tolerance_radius_s": r_val
                })

            # 7. Constructive Test: Functional-Geometry Surrogate
            surr_df = evaluate_constructive_surrogates(
                model, m_key, depth, all_imgs, cov_matrix, centroid, metric_matrix, device
            )
            all_surrogates.append(surr_df)

            # 8. Token Position Heterogeneity (Section 10)
            token_positions = {
                "central_patch": num_patches // 2,
                "corner_patch": 0,
                "edge_patch": int(np.sqrt(num_patches)) // 2
            }
            with torch.no_grad():
                clean_logits_25, coll_25 = forward_block_by_block(model, m_key, x=all_imgs[:25], collect_depths=(depth,))
                h_c = coll_25[depth]
                top_c = clean_logits_25.argmax(dim=-1)
                l_temp = clean_logits_25.clone()
                l_temp[torch.arange(25), top_c] = -1e9
                r_up = l_temp.argmax(dim=-1)
                clean_m = clean_logits_25[torch.arange(25), top_c] - clean_logits_25[torch.arange(25), r_up]

            for pos_name, tok_idx in token_positions.items():
                single_mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
                single_mask[tok_idx] = True
                for test_dir in ["pc1", "grad_margin", "rand_1"]:
                    v_dir = dir_library[test_dir]
                    damages = []
                    for s in scale_grid:
                        hook = DirectionalPerturbationHook(depth, v_dir, s * sigma_l, mask=single_mask)
                        h_p = hook(depth, h_c)
                        with torch.no_grad():
                            p_logits, _ = forward_block_by_block(model, m_key, start_depth=depth, h_start=h_p)
                            pert_m = p_logits[torch.arange(25), top_c] - p_logits[torch.arange(25), r_up]
                            dmg = float((clean_m - pert_m).mean().item())
                            damages.append(dmg)
                    # compute single token radius
                    r_tok = scale_grid[-1]
                    for idx_s in range(len(scale_grid) - 1):
                        if damages[idx_s] <= 0.20 and damages[idx_s+1] > 0.20:
                            r_tok = scale_grid[idx_s]
                            break
                    all_consistency.append({
                        "model_key": m_key,
                        "depth": depth,
                        "token_position": pos_name,
                        "direction": test_dir,
                        "single_token_tolerance_radius": r_tok
                    })

            # 9. Image-Wise Consistency Test (at peak fungible depth, e.g. Depth 8 or 7)
            if depth == 8 or (depth == 7 and 8 not in depths_to_evaluate):
                print(f"  Evaluating image-wise consistency on {m_key} at depth {depth}...")
                sample_imgs = all_imgs[:20]
                img_radii = []
                for i_idx in range(len(sample_imgs)):
                    single_img = sample_imgs[i_idx:i_idx+1]
                    _, single_radii = evaluate_directional_tolerance_profile(
                        model, m_key, depth, single_img, dir_library, sigma_l, scale_grid, device
                    )
                    img_radii.append(single_radii.set_index("direction")["tolerance_radius_s"])
                
                img_matrix = pd.DataFrame(img_radii)
                # Compute pairwise Spearman correlation between images
                corr_mat = img_matrix.T.corr(method="spearman").values
                valid_corrs = corr_mat[np.triu_indices_from(corr_mat, k=1)]
                mean_pairwise_corr = float(np.nanmean(valid_corrs)) if len(valid_corrs) > 0 else 0.0
                all_consistency.append({
                    "model_key": m_key,
                    "depth": depth,
                    "mean_image_spearman_corr": mean_pairwise_corr
                })
                print(f"  Mean Pairwise Image Consistency (Spearman rho): {mean_pairwise_corr:.4f}")

    # Compile all outputs
    full_curves_df = pd.concat(all_curves, ignore_index=True)
    full_radii_df = pd.concat(all_radii, ignore_index=True)
    full_metrics_df = pd.DataFrame(all_metrics)
    full_alignment_df = pd.DataFrame(all_alignment)
    full_interventions_df = pd.DataFrame(all_interventions)
    full_consistency_df = pd.DataFrame(all_consistency)
    full_surrogates_df = pd.concat(all_surrogates, ignore_index=True)

    # Save CSVs
    full_curves_df.to_csv(os.path.join(output_dir, "directional_tolerance.csv"), index=False)
    full_radii_df.to_csv(os.path.join(output_dir, "functional_spectrum.csv"), index=False)
    full_metrics_df.to_csv(os.path.join(output_dir, "fungible_dimension.csv"), index=False)
    full_alignment_df.to_csv(os.path.join(output_dir, "covariance_function_alignment.csv"), index=False)
    full_interventions_df.to_csv(os.path.join(output_dir, "intervention_projection.csv"), index=False)
    full_consistency_df.to_csv(os.path.join(output_dir, "imagewise_consistency.csv"), index=False)
    full_surrogates_df.to_csv(os.path.join(output_dir, "constructive_surrogates.csv"), index=False)
    print(f"\nAll machine-readable data successfully saved to: {output_dir}")

    # Generate Figures
    generate_figures(
        full_curves_df, full_radii_df, full_metrics_df,
        full_alignment_df, full_interventions_df, full_surrogates_df,
        figures_dir
    )

    # Validation Manifest
    manifest = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "device": str(device),
        "models_evaluated": list(models_to_evaluate),
        "depths_evaluated": list(depths_to_evaluate),
        "num_images_pilot": n_images,
        "anisotropy_summary": {
            row["model_key"] + f"_depth_{row['depth']}": float(row["anisotropy_ratio"])
            for _, row in full_alignment_df.iterrows()
        },
        "mean_image_consistency": {
            row["model_key"]: float(row["mean_image_spearman_corr"])
            for _, row in full_consistency_df.iterrows()
            if "mean_image_spearman_corr" in row and pd.notna(row["mean_image_spearman_corr"])
        }
    }
    with open(os.path.join(output_dir, "validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Validation manifest saved to: {os.path.join(output_dir, 'validation_manifest.json')}")

    return manifest


# ==============================================================================
# 6. PUBLICATION FIGURES GENERATION
# ==============================================================================

def generate_figures(
    curves_df: pd.DataFrame,
    radii_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    interventions_df: pd.DataFrame,
    surrogates_df: pd.DataFrame,
    output_dir: str
):
    """Generates publication-quality figures 1 through 6."""
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # -------------------------------------------------------------------------
    # FIGURE 1: Directional Tolerance Curves at Fragile vs Fungible Layers
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # DeiT-Small: Depth 5 (Fragile) vs Depth 8 (Fungible)
    dirs_to_plot = ["pc1", "pc_mid", "pc_bottom", "grad_margin", "jac_null", "rand_1", "sign_flip_disp"]
    palette = sns.color_palette("tab10", len(dirs_to_plot))
    dir_colors = dict(zip(dirs_to_plot, palette))

    for ax, depth, title in zip(axes, [5, 8], ["Depth 5 (Fragile Baseline)", "Depth 8 (Peak Fungible Layer)"]):
        sub = curves_df[(curves_df["model_key"] == "deit_small") & (curves_df["depth"] == depth)]
        for d_name in dirs_to_plot:
            d_sub = sub[sub["direction"] == d_name].sort_values("scale_s")
            ax.plot(
                d_sub["scale_s"], d_sub["margin_damage"],
                marker="o", linewidth=2.0, label=d_name, color=dir_colors[d_name]
            )
        ax.axhline(0.20, color="black", linestyle="--", alpha=0.7, label="Tolerance Threshold (0.20)")
        ax.set_xlabel("Perturbation Scale $s = \\alpha / \\sigma_l$", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Margin Damage", fontsize=11, fontweight="bold")
        ax.set_title(f"DeiT-Small: {title}", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.6)
        if depth == 5:
            ax.legend(frameon=True, fontsize=9, loc="upper left")

    fig.suptitle("Figure 1: Finite-Radius Directional Tolerance Curves (Fragile vs. Fungible Layers)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_1_directional_tolerance_curves.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 2: Fungibility Spectrum Across Depth
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 6), dpi=300)
    # Box/scatter plot of tolerance radii across directions for each depth and model
    radii_plot_df = radii_df.copy()
    radii_plot_df["depth_str"] = "Block " + radii_plot_df["depth"].astype(str)
    
    sns.boxplot(
        data=radii_plot_df, x="depth_str", y="tolerance_radius_s", hue="model_key",
        palette=["#2b5c8f", "#d95f02"], showmeans=True, meanprops={"marker":"s", "markerfacecolor":"white"}
    )
    plt.xlabel("Transformer Block Depth", fontsize=12, fontweight="bold")
    plt.ylabel("Tolerance Radius $r_l(v; \\epsilon=0.20)$ [units of $\\sigma_l$]", fontsize=12, fontweight="bold")
    plt.title("Figure 2: Evolution of Directional Fungibility Spectrum Across Depth", fontsize=14, fontweight="bold")
    plt.legend(title="Architecture", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_2_fungibility_spectrum_depth.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 3: Functional Metric Eigenspectrum & Near-Null Dimensionality
    # -------------------------------------------------------------------------
    fig, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    for m_key, color in zip(["deit_small", "vit_base"], ["#2b5c8f", "#d95f02"]):
        sub_m = metrics_df[metrics_df["model_key"] == m_key].sort_values("depth")
        ax3a.plot(sub_m["depth"], sub_m["null_subspace_fraction"], marker="o", linewidth=2.2, label=m_key, color=color)
        ax3b.plot(sub_m["depth"], sub_m["effective_rank_M"], marker="s", linewidth=2.2, label=m_key, color=color)

    ax3a.set_xlabel("Block Depth", fontsize=11, fontweight="bold")
    ax3a.set_ylabel("Near-Null Subspace Fraction ($\\omega_k / \\omega_1 < 10^{-3}$)", fontsize=11, fontweight="bold")
    ax3a.set_title("Near-Null Subspace Dimensionality", fontsize=12, fontweight="bold")
    ax3a.legend()

    ax3b.set_xlabel("Block Depth", fontsize=11, fontweight="bold")
    ax3b.set_ylabel("Effective Metric Rank $r_{\\text{eff}}(M_l)$", fontsize=11, fontweight="bold")
    ax3b.set_title("Downstream Functional Metric Rank", fontsize=12, fontweight="bold")
    ax3b.legend()

    fig.suptitle("Figure 3: Downstream Functional Metric ($M_l = \\mathbb{E}[J^T J]$) Geometry Across Depth", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_3_functional_metric_eigenspectrum.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 4: Covariance vs Functional Alignment
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    for m_key, color in zip(["deit_small", "vit_base"], ["#2b5c8f", "#d95f02"]):
        sub_align = alignment_df[alignment_df["model_key"] == m_key].sort_values("depth")
        plt.plot(
            sub_align["depth"], sub_align["anisotropy_ratio"],
            marker="o", linewidth=2.5, label=f"{m_key} (Anisotropy Ratio)", color=color
        )
    plt.xlabel("Block Depth", fontsize=12, fontweight="bold")
    plt.ylabel("Anisotropy Ratio $\\max(r) / \\min(r)$", fontsize=12, fontweight="bold")
    plt.title("Figure 4: Directional Anisotropy of Equivalence Geometry Across Depth", fontsize=14, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_4_covariance_functional_alignment.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 5: Intervention Projections onto Functional Geometry
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    sub_int = interventions_df[interventions_df["depth"] == 8].copy()
    
    sns.barplot(
        data=sub_int, x="intervention", y="functional_curvature_vMv", hue="model_key",
        palette=["#2b5c8f", "#d95f02"]
    )
    plt.yscale("log")
    plt.xlabel("Intervention Displacement Direction", fontsize=12, fontweight="bold")
    plt.ylabel("Functional Curvature $v^T M_l v$ (Log Scale)", fontsize=12, fontweight="bold")
    plt.title("Figure 5: Projection of Historical Interventions onto Functional Metric at Depth 8", fontsize=14, fontweight="bold")
    plt.legend(title="Model", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_5_intervention_projections.png"), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 6: Constructive Surrogate Test
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    sub_surr = surrogates_df[surrogates_df["depth"] == 8].copy()

    sns.barplot(
        data=sub_surr, x="condition", y="margin_damage", hue="model_key",
        palette=["#2b5c8f", "#d95f02"]
    )
    plt.xlabel("Surrogate Condition", fontsize=12, fontweight="bold")
    plt.ylabel("Downstream Margin Damage (Lower is Better)", fontsize=12, fontweight="bold")
    plt.title("Figure 6: Constructive Test of Functional-Geometry-Aware Surrogates at Depth 8", fontsize=14, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Model", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_6_constructive_surrogate_test.png"), bbox_inches="tight")
    plt.close()
    print("All publication figures successfully generated.")


if __name__ == "__main__":
    run_functional_geometry_experiment()
