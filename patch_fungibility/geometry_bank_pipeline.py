"""
patch_fungibility/geometry_bank_pipeline.py

Pipeline for GEOMETRY-DIVERSITY BANK POC:
"At the same downstream token budget, can a small set of class-agnostic synthetic tokens
that explicitly preserve learned late-layer geometry and token diversity outperform simple Random Pruning?"

Evaluates:
- Method A: RANDOM_PRUNING (Keep B real patches)
- Method B: REAL_PLUS_CENTROID (Keep B-1 real patches + 1 unweighted centroid)
- Method C: REAL_PLUS_PCA_BANK (Keep B-K real patches + K PCA synthetic tokens)
- Method D: REAL_PLUS_KMEANS_BANK (Keep B-K real patches + K K-means cluster centroids)
- Method E: REAL_PLUS_RANDOM_DIR_BANK (Keep B-K real patches + K random-direction tokens)

Budgets per model:
- DeiT-Tiny: B in {45, 73}
- DeiT-Small: B in {28, 52}
- ViT-B/16: B in {73, 112}
- DINOv2: B in {184, 199}

Bank sizes: K in {4, 8, 16} (subject to K < B).
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits, ParquetImageSubset
from patch_fungibility.dense_fraction_models import load_model_and_transform, forward_block_by_block
from patch_fungibility.dense_fraction_pipeline import extract_activations_and_clean_logits
from patch_fungibility.compression_pipeline import compute_theoretical_flops


MASK_SEEDS = [31001, 31002, 31003, 31004, 31005]
K_VALUES = [4, 8, 16]

MODEL_CONFIGS = [
    {
        "key": "deit_tiny",
        "depth": 8,
        "is_primary": True,
        "n_patches": 196,
        "embed_dim": 192,
        "num_heads": 3,
        "clean_acc": 0.6790,
        "clean_margin": 1.1681,
        "budgets": [45, 73]
    },
    {
        "key": "deit_small",
        "depth": 8,
        "is_primary": True,
        "n_patches": 196,
        "embed_dim": 384,
        "num_heads": 6,
        "clean_acc": 0.7610,
        "clean_margin": 2.2514,
        "budgets": [28, 52]
    },
    {
        "key": "vit_base",
        "depth": 7,
        "is_primary": True,
        "n_patches": 196,
        "embed_dim": 768,
        "num_heads": 12,
        "clean_acc": 0.7610,
        "clean_margin": 3.1923,
        "budgets": [73, 112]
    },
    {
        "key": "dinov2",
        "depth": 9,
        "is_primary": True,
        "n_patches": 256,
        "embed_dim": 384,
        "num_heads": 6,
        "clean_acc": 0.7880,
        "clean_margin": 2.7605,
        "budgets": [184, 199]
    }
]


def forward_downstream_standard(
    model: nn.Module,
    model_key: str,
    start_depth: int,
    seq: torch.Tensor
) -> torch.Tensor:
    """
    Executes downstream Transformer blocks on a sequence of tokens without multiplicity bias.
    seq: (B_batch, 1 + B, D)
    """
    cur = seq
    if model_key in ("deit_tiny", "deit_small", "vit_base"):
        for b_idx in range(start_depth, len(model.blocks)):
            cur = model.blocks[b_idx](cur)
        h_norm = model.norm(cur)
        logits = model.forward_head(h_norm)
        return logits

    elif model_key == "dinov2":
        backbone = model.backbone
        for b_idx in range(start_depth, len(backbone.blocks)):
            cur = backbone.blocks[b_idx](cur)
        h_norm = backbone.norm(cur)
        cls_norm = h_norm[:, 0]
        patch_mean = h_norm[:, 1:].mean(dim=1)
        readout = torch.cat([cls_norm, patch_mean], dim=-1)
        logits = model.linear_head(readout)
        return logits
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def compute_pca_bank(
    all_tokens: torch.Tensor,
    k_list: List[int] = [4, 8, 16]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Computes PCA on calibration patch tokens and constructs symmetric tokens:
    mu +/- sqrt(lambda_i) * v_i for top K/2 directions.
    Returns: (mu, eigenvalues, eigenvectors, dict of {K: bank_tensor_K_D})
    """
    device = all_tokens.device
    embed_dim = all_tokens.size(1)
    mu = all_tokens.mean(dim=0)
    centered = all_tokens - mu.unsqueeze(0)
    cov = torch.cov(centered.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvectors = torch.flip(eigenvectors, dims=[1])
    eigenvalues = torch.clamp(eigenvalues, min=1e-12)

    banks = {}
    for K in k_list:
        n_pairs = K // 2
        tokens = []
        for i in range(n_pairs):
            scale = torch.sqrt(eigenvalues[i])
            vec = eigenvectors[:, i]
            t_pos = mu + scale * vec
            t_neg = mu - scale * vec
            tokens.append(t_pos.unsqueeze(0))
            tokens.append(t_neg.unsqueeze(0))
        banks[K] = torch.cat(tokens, dim=0).to(device)

    return mu, eigenvalues, eigenvectors, banks


def compute_kmeans_bank(
    all_tokens: torch.Tensor,
    k_list: List[int] = [4, 8, 16],
    random_state: int = 42
) -> Dict[int, torch.Tensor]:
    """
    Fits K-means on calibration patch tokens for each K in k_list.
    Returns: dict of {K: centroids_K_D}
    """
    device = all_tokens.device
    tokens_np = all_tokens.detach().cpu().numpy().astype(np.float32)
    banks = {}

    for K in k_list:
        # Fit K-means with fixed seed
        km = KMeans(n_clusters=K, n_init=1, max_iter=50, random_state=random_state)
        km.fit(tokens_np)
        centers_t = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
        banks[K] = centers_t

    return banks


def compute_random_direction_bank(
    mu: torch.Tensor,
    eigenvalues: torch.Tensor,
    k_list: List[int] = [4, 8, 16],
    seed: int = 42
) -> Dict[int, torch.Tensor]:
    """
    Constructs symmetric synthetic tokens along random orthonormal directions:
    mu +/- sqrt(lambda_i) * u_i
    Returns: dict of {K: bank_tensor_K_D}
    """
    device = mu.device
    embed_dim = mu.size(0)
    rng = np.random.RandomState(seed)
    max_k = max(k_list)
    max_pairs = max_k // 2

    # QR decomposition to get orthonormal basis
    mat = rng.randn(embed_dim, max_pairs).astype(np.float32)
    q, _ = np.linalg.qr(mat)
    u_basis = torch.tensor(q, dtype=torch.float32, device=device)

    banks = {}
    for K in k_list:
        n_pairs = K // 2
        tokens = []
        for i in range(n_pairs):
            scale = torch.sqrt(eigenvalues[i])
            vec = u_basis[:, i]
            t_pos = mu + scale * vec
            t_neg = mu - scale * vec
            tokens.append(t_pos.unsqueeze(0))
            tokens.append(t_neg.unsqueeze(0))
        banks[K] = torch.cat(tokens, dim=0).to(device)

    return banks


def evaluate_batch_sequence(
    model: nn.Module,
    model_key: str,
    depth: int,
    seq: torch.Tensor,
    labels_t: torch.Tensor,
    mask_clean_inc: torch.Tensor,
    clean_preds: torch.Tensor,
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Runs forward downstream in batches and returns accuracy, margin, and flip rate.
    """
    N_eval = seq.size(0)
    all_logits = []

    with torch.no_grad():
        for bi in range(0, N_eval, batch_size):
            l_bi = forward_downstream_standard(model, model_key, depth, seq[bi : bi + batch_size])
            all_logits.append(l_bi)

    logits = torch.cat(all_logits, dim=0)
    preds = torch.argmax(logits, dim=-1)
    acc = float((preds == labels_t).float().mean().item())

    true_logits = logits[torch.arange(N_eval), labels_t]
    strongest_inc = torch.max(torch.where(mask_clean_inc, logits, -1e9), dim=1).values
    margins = (true_logits - strongest_inc).cpu().numpy()
    mean_margin = float(np.mean(margins))
    median_margin = float(np.median(margins))
    flip_rate = float((preds != clean_preds).float().mean().item())

    return {
        "top1_acc": acc,
        "mean_margin": mean_margin,
        "median_margin": median_margin,
        "flip_rate": flip_rate
    }


def run_geometry_bank_experiment(
    output_dir: str = "outputs/fungibility_geometry_bank",
    smoke_test: bool = False,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Master runner for GEOMETRY-DIVERSITY BANK POC.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Running Geometry-Diversity Bank POC on device: {device} ===")

    # 1. Load canonical dataset splits
    n_per_split = 64 if smoke_test else 1000
    print(f"[Dataset] Loading canonical ImageNet disjoint splits (N={n_per_split})...")
    calib_ds_raw, eval_ds_raw, calib_manifest, eval_manifest = get_disjoint_imagenet_splits(
        calib_seed=9101, eval_seed=9201, n_per_split=n_per_split
    )
    calib_samples = calib_ds_raw._samples
    eval_samples = eval_ds_raw._samples
    labels = torch.tensor([s[1] for s in eval_samples], dtype=torch.long)

    # 2. Load frozen mask permutations
    with open("outputs/fungibility_dense_fraction/mask_permutations.json") as f:
        perms_meta = json.load(f)
    perms_196 = {int(s): np.array(p) for s, p in perms_meta["perms_196"].items()}
    perms_256 = {int(s): np.array(p) for s, p in perms_meta["perms_256"].items()}

    all_rows = []

    for cfg in MODEL_CONFIGS:
        m_key = cfg["key"]
        depth = cfg["depth"]
        n_patches = cfg["n_patches"]
        embed_dim = cfg["embed_dim"]
        budgets = cfg["budgets"]
        clean_acc = cfg["clean_acc"]
        clean_margin = cfg["clean_margin"]
        perms = perms_196 if n_patches == 196 else perms_256

        print(f"\n========================================================")
        print(f"Model: {m_key} | Depth: {depth} | N_patches: {n_patches} | Embed Dim: {embed_dim}")
        print(f"Target Budgets B: {budgets}")
        print(f"========================================================")

        model, transform, _ = load_model_and_transform(m_key, device=device)
        model.eval()

        # Step 2.1: Extract calibration activations at target depth
        print(f"  Extracting calibration activations (N={n_per_split})...")
        calib_ds = ParquetImageSubset(calib_samples, transform=transform)
        calib_loader = DataLoader(calib_ds, batch_size=64 if embed_dim <= 384 else 32, shuffle=False)

        cached_calib_h, _, _ = extract_activations_and_clean_logits(
            model, m_key, calib_loader, target_depths=(depth,), device=device
        )
        # Spatial patches only (tokens 1 to n_patches)
        h_calib_spatial = cached_calib_h[depth][:, 1:, :].to(device)
        all_calib_tokens = h_calib_spatial.reshape(-1, embed_dim)
        print(f"  Calibration tokens shape: {all_calib_tokens.shape}")

        # Step 2.2: Compute Banks
        print(f"  Computing PCA Bank...")
        mu, eigenvalues, eigenvectors, pca_banks = compute_pca_bank(all_calib_tokens, K_VALUES)

        print(f"  Computing K-Means Bank...")
        kmeans_banks = compute_kmeans_bank(all_calib_tokens, K_VALUES, random_state=42)

        print(f"  Computing Random Direction Bank...")
        rand_banks = compute_random_direction_bank(mu, eigenvalues, K_VALUES, seed=42)

        del cached_calib_h, h_calib_spatial, all_calib_tokens

        # Step 2.3: Extract evaluation activations at target depth
        print(f"  Extracting evaluation activations and clean logits (N={n_per_split})...")
        eval_ds = ParquetImageSubset(eval_samples, transform=transform)
        eval_loader = DataLoader(eval_ds, batch_size=64 if embed_dim <= 384 else 32, shuffle=False)

        cached_eval_h, clean_logits_full, _ = extract_activations_and_clean_logits(
            model, m_key, eval_loader, target_depths=(depth,), device=device
        )
        h_eval = cached_eval_h[depth].to(device)
        clean_logits_t = clean_logits_full.to(device)
        labels_t = labels.to(device)
        N_eval = len(labels)

        # Precompute clean mask and predictions
        clean_preds = torch.argmax(clean_logits_t, dim=-1)
        mask_clean_inc = torch.ones_like(clean_logits_t, dtype=torch.bool)
        mask_clean_inc[torch.arange(N_eval), labels_t] = False

        cls_eval = h_eval[:, :1, :]
        patches_eval = h_eval[:, 1:, :]

        # Precompute FLOPs for budgets
        flops_by_budget = {}
        for B in budgets:
            flops = compute_theoretical_flops(
                n_patches=n_patches,
                embed_dim=embed_dim,
                start_depth=depth,
                total_blocks=12,
                k_replaced=n_patches - B,
                model_key=m_key
            )
            flops_by_budget[B] = flops

        # Step 2.4: Iterate over mask seeds and budgets
        for s_idx, s in enumerate(MASK_SEEDS):
            p_s = perms[s]

            for B in budgets:
                flops = flops_by_budget[B]
                # In the dense sweep convention, when keeping B patches,
                # the surviving B patch indices are the last B elements of p_s:
                surviving_B = p_s[n_patches - B :]

                # ---------------------------------------------------------
                # Condition 1: RANDOM_PRUNING (Keep all B real patches)
                # ---------------------------------------------------------
                real_p_B = patches_eval[:, surviving_B, :]
                seq_pruning = torch.cat([cls_eval, real_p_B], dim=1)
                m_pruning = evaluate_batch_sequence(
                    model, m_key, depth, seq_pruning, labels_t, mask_clean_inc, clean_preds
                )
                acc_pruning = m_pruning["top1_acc"]
                margin_pruning = m_pruning["mean_margin"]

                all_rows.append({
                    "model": m_key,
                    "depth": depth,
                    "mask_seed": s,
                    "budget_B": B,
                    "condition": "RANDOM_PRUNING",
                    "K_synth": 0,
                    "M_real": B,
                    "total_downstream_tokens": 1 + B,
                    "top1_acc": acc_pruning,
                    "clean_relative_acc": acc_pruning / clean_acc,
                    "mean_margin": margin_pruning,
                    "median_margin": m_pruning["median_margin"],
                    "flip_rate": m_pruning["flip_rate"],
                    "delta_acc_vs_pruning": 0.0,
                    "delta_margin_vs_pruning": 0.0,
                    "total_model_gflops": flops["total_comp_gflops"],
                    "flop_reduction_pct": flops["total_reduction_pct"]
                })

                # ---------------------------------------------------------
                # Condition 2: REAL_PLUS_CENTROID (Keep B-1 real + 1 centroid)
                # ---------------------------------------------------------
                real_indices_cent = surviving_B[: B - 1]
                real_p_cent = patches_eval[:, real_indices_cent, :]
                mu_exp = mu.view(1, 1, embed_dim).expand(N_eval, 1, -1)
                seq_cent = torch.cat([cls_eval, real_p_cent, mu_exp], dim=1)
                m_cent = evaluate_batch_sequence(
                    model, m_key, depth, seq_cent, labels_t, mask_clean_inc, clean_preds
                )

                all_rows.append({
                    "model": m_key,
                    "depth": depth,
                    "mask_seed": s,
                    "budget_B": B,
                    "condition": "REAL_PLUS_CENTROID",
                    "K_synth": 1,
                    "M_real": B - 1,
                    "total_downstream_tokens": 1 + B,
                    "top1_acc": m_cent["top1_acc"],
                    "clean_relative_acc": m_cent["top1_acc"] / clean_acc,
                    "mean_margin": m_cent["mean_margin"],
                    "median_margin": m_cent["median_margin"],
                    "flip_rate": m_cent["flip_rate"],
                    "delta_acc_vs_pruning": m_cent["top1_acc"] - acc_pruning,
                    "delta_margin_vs_pruning": m_cent["mean_margin"] - margin_pruning,
                    "total_model_gflops": flops["total_comp_gflops"],
                    "flop_reduction_pct": flops["total_reduction_pct"]
                })

                # ---------------------------------------------------------
                # Conditions 3, 4, 5: Banks for K in {4, 8, 16}
                # ---------------------------------------------------------
                for K in K_VALUES:
                    if K >= B:
                        continue
                    M_real = B - K
                    real_indices_K = surviving_B[: M_real]
                    real_p_K = patches_eval[:, real_indices_K, :]

                    # 3. PCA Bank
                    pca_bank_exp = pca_banks[K].view(1, K, embed_dim).expand(N_eval, K, -1)
                    seq_pca = torch.cat([cls_eval, real_p_K, pca_bank_exp], dim=1)
                    m_pca = evaluate_batch_sequence(
                        model, m_key, depth, seq_pca, labels_t, mask_clean_inc, clean_preds
                    )
                    all_rows.append({
                        "model": m_key,
                        "depth": depth,
                        "mask_seed": s,
                        "budget_B": B,
                        "condition": "REAL_PLUS_PCA_BANK",
                        "K_synth": K,
                        "M_real": M_real,
                        "total_downstream_tokens": 1 + B,
                        "top1_acc": m_pca["top1_acc"],
                        "clean_relative_acc": m_pca["top1_acc"] / clean_acc,
                        "mean_margin": m_pca["mean_margin"],
                        "median_margin": m_pca["median_margin"],
                        "flip_rate": m_pca["flip_rate"],
                        "delta_acc_vs_pruning": m_pca["top1_acc"] - acc_pruning,
                        "delta_margin_vs_pruning": m_pca["mean_margin"] - margin_pruning,
                        "total_model_gflops": flops["total_comp_gflops"],
                        "flop_reduction_pct": flops["total_reduction_pct"]
                    })

                    # 4. K-Means Bank
                    kmeans_bank_exp = kmeans_banks[K].view(1, K, embed_dim).expand(N_eval, K, -1)
                    seq_km = torch.cat([cls_eval, real_p_K, kmeans_bank_exp], dim=1)
                    m_km = evaluate_batch_sequence(
                        model, m_key, depth, seq_km, labels_t, mask_clean_inc, clean_preds
                    )
                    all_rows.append({
                        "model": m_key,
                        "depth": depth,
                        "mask_seed": s,
                        "budget_B": B,
                        "condition": "REAL_PLUS_KMEANS_BANK",
                        "K_synth": K,
                        "M_real": M_real,
                        "total_downstream_tokens": 1 + B,
                        "top1_acc": m_km["top1_acc"],
                        "clean_relative_acc": m_km["top1_acc"] / clean_acc,
                        "mean_margin": m_km["mean_margin"],
                        "median_margin": m_km["median_margin"],
                        "flip_rate": m_km["flip_rate"],
                        "delta_acc_vs_pruning": m_km["top1_acc"] - acc_pruning,
                        "delta_margin_vs_pruning": m_km["mean_margin"] - margin_pruning,
                        "total_model_gflops": flops["total_comp_gflops"],
                        "flop_reduction_pct": flops["total_reduction_pct"]
                    })

                    # 5. Random Direction Bank
                    rand_bank_exp = rand_banks[K].view(1, K, embed_dim).expand(N_eval, K, -1)
                    seq_rand = torch.cat([cls_eval, real_p_K, rand_bank_exp], dim=1)
                    m_rand = evaluate_batch_sequence(
                        model, m_key, depth, seq_rand, labels_t, mask_clean_inc, clean_preds
                    )
                    all_rows.append({
                        "model": m_key,
                        "depth": depth,
                        "mask_seed": s,
                        "budget_B": B,
                        "condition": "REAL_PLUS_RANDOM_DIR_BANK",
                        "K_synth": K,
                        "M_real": M_real,
                        "total_downstream_tokens": 1 + B,
                        "top1_acc": m_rand["top1_acc"],
                        "clean_relative_acc": m_rand["top1_acc"] / clean_acc,
                        "mean_margin": m_rand["mean_margin"],
                        "median_margin": m_rand["median_margin"],
                        "flip_rate": m_rand["flip_rate"],
                        "delta_acc_vs_pruning": m_rand["top1_acc"] - acc_pruning,
                        "delta_margin_vs_pruning": m_rand["mean_margin"] - margin_pruning,
                        "total_model_gflops": flops["total_comp_gflops"],
                        "flop_reduction_pct": flops["total_reduction_pct"]
                    })

        del cached_eval_h, h_eval, clean_logits_t, cls_eval, patches_eval, model
        torch.cuda.empty_cache()

    # 3. Compile DataFrames and Summaries
    df_all = pd.DataFrame(all_rows)
    all_results_path = os.path.join(output_dir, "all_results.csv")
    df_all.to_csv(all_results_path, index=False)
    print(f"\n[Saved] All results saved to {all_results_path} ({len(df_all)} rows)")

    # Aggregated Summary across Mask Seeds
    group_cols = ["model", "depth", "budget_B", "condition", "K_synth", "M_real", "total_downstream_tokens"]
    df_summary = df_all.groupby(group_cols).agg(
        mean_top1_acc=("top1_acc", "mean"),
        std_top1_acc=("top1_acc", "std"),
        mean_clean_relative_acc=("clean_relative_acc", "mean"),
        mean_margin=("mean_margin", "mean"),
        std_margin=("mean_margin", "std"),
        mean_delta_acc=("delta_acc_vs_pruning", "mean"),
        std_delta_acc=("delta_acc_vs_pruning", "std"),
        mean_delta_margin=("delta_margin_vs_pruning", "mean"),
        std_delta_margin=("delta_margin_vs_pruning", "std"),
        mean_flip_rate=("flip_rate", "mean"),
        total_model_gflops=("total_model_gflops", "first"),
        flop_reduction_pct=("flop_reduction_pct", "first")
    ).reset_index()

    summary_path = os.path.join(output_dir, "matched_budget_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"[Saved] Matched-budget summary saved to {summary_path}")

    # Delta vs Pruning Pivot Table
    df_delta = df_summary[df_summary["condition"] != "RANDOM_PRUNING"][[
        "model", "budget_B", "condition", "K_synth", "mean_top1_acc", "mean_delta_acc", "mean_delta_margin"
    ]].copy()
    delta_path = os.path.join(output_dir, "delta_vs_pruning.csv")
    df_delta.to_csv(delta_path, index=False)
    print(f"[Saved] Delta vs pruning table saved to {delta_path}")

    # 4. Audit against Decision Rules
    # Check if any geometry bank (PCA or K-means) beats Random Pruning by >= +1.0% on at least 2 models
    bank_conditions = ["REAL_PLUS_PCA_BANK", "REAL_PLUS_KMEANS_BANK"]
    promising_models = set()
    max_delta_by_model = {}

    for m_cfg in MODEL_CONFIGS:
        mk = m_cfg["key"]
        sub = df_summary[(df_summary["model"] == mk) & (df_summary["condition"].isin(bank_conditions))]
        max_delta = sub["mean_delta_acc"].max() if len(sub) > 0 else -1.0
        max_delta_by_model[mk] = float(max_delta)
        if max_delta >= 0.010:  # >= +1.0 percentage point
            promising_models.add(mk)

    if len(promising_models) >= 2:
        decision_verdict = "PROMISING"
    elif len(promising_models) == 1:
        decision_verdict = "MIXED"
    else:
        decision_verdict = "KILL APPLICATION"

    elapsed = time.time() - start_time
    print(f"\n========================================================")
    print(f"Decision Verdict: {decision_verdict}")
    print(f"Max Delta Acc vs Pruning by Model:")
    for mk, md in max_delta_by_model.items():
        print(f"  - {mk}: {md * 100:+.2f}%")
    print(f"Total Execution Time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    print(f"========================================================")

    val_res = {
        "verdict": decision_verdict,
        "promising_models": list(promising_models),
        "max_delta_by_model": max_delta_by_model,
        "decision_rule": {
            "promising_criterion": "beats Random Pruning by >= +1.0% on >= 2 model families",
            "kill_criterion": "Random Pruning matches or beats geometry banks across tested budgets"
        },
        "latency_benchmarking_triggered": (decision_verdict == "PROMISING"),
        "elapsed_seconds": elapsed
    }
    with open(os.path.join(output_dir, "validation_results.json"), "w") as f:
        json.dump(val_res, f, indent=2)

    manifest = {
        "experiment_name": "GEOMETRY-DIVERSITY BANK POC",
        "date": "2026-09-29",
        "models": [c["key"] for c in MODEL_CONFIGS],
        "mask_seeds": MASK_SEEDS,
        "K_values": K_VALUES,
        "verdict": decision_verdict,
        "smoke_test": smoke_test,
        "elapsed_seconds": elapsed
    }
    with open(os.path.join(output_dir, "experiment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return val_res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/fungibility_geometry_bank")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    run_geometry_bank_experiment(output_dir=args.output_dir, smoke_test=args.smoke_test)
