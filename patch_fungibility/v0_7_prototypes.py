import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_c = a - np.mean(a)
    b_c = b - np.mean(b)
    norm_a = np.linalg.norm(a_c)
    norm_b = np.linalg.norm(b_c)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_c, b_c) / (norm_a * norm_b))


def generate_v0_7_prototypes(
    model_name: str,
    calibration_stats_npz_path: str = "outputs/fungibility_v0_6/calibration_statistics.npz",
    gaussian_seeds: List[int] = [9701, 9702, 9703],
    perm_seeds: List[int] = [9801, 9802, 9803],
    sign_seed: int = 9901,
    cosine_seeds: List[int] = [10001, 10002, 10003],
    target_cosines: List[float] = [0.0, 0.25, 0.50, 0.75],
    scale_factors: List[float] = [0.25, 0.50, 1.00, 2.00, 4.00],
    wrong_depths: List[int] = [5, 6, 7, 9, 10],
) -> Dict[str, Any]:
    """
    Constructs all V0.7 prototype controls for depth 8 interventions:
    - mu_8 (reference calibration mean)
    - sigma_8 (reference calibration std)
    - wrong-depth means (mu_5, mu_6, mu_7, mu_9, mu_10)
    - norm-matched wrong-depth means (mu_norm_5, etc.)
    - coordinate permutations of mu_8 (perm_9801, perm_9802, perm_9803)
    - full sign inversion (-mu_8) and partial sign flips (25%, 50%, 75%)
    - scale sweep (0.25, 0.50, 1.00, 2.00, 4.00)
    - cosine sweep (target cosines 0.0, 0.25, 0.50, 0.75 with seeds 10001, 10002, 10003)
    - variance-normalized rescaled prototype (z_mu_rescaled)
    """
    stats_data = np.load(calibration_stats_npz_path)

    # 1. Load calibration stats for depths 5..10
    depth_means = {}
    depth_sigmas = {}
    for d in [5, 6, 7, 8, 9, 10]:
        mu_key = f"{model_name}_depth_{d}_mu"
        sig_key = f"{model_name}_depth_{d}_sigma"
        assert mu_key in stats_data, f"Missing key {mu_key} in calibration npz!"
        depth_means[d] = stats_data[mu_key].astype(np.float32)
        depth_sigmas[d] = stats_data[sig_key].astype(np.float32)

    mu_8 = depth_means[8]
    sigma_8 = depth_sigmas[8]
    embed_dim = len(mu_8)
    norm_mu8 = float(np.linalg.norm(mu_8))
    hat_mu8 = mu_8 / norm_mu8

    vectors: Dict[str, np.ndarray] = {
        "mu_8": mu_8.copy(),
    }

    # 2. Wrong-depth means (unnormalized and norm-matched)
    for wd in wrong_depths:
        mu_wd = depth_means[wd].copy()
        norm_wd = float(np.linalg.norm(mu_wd))
        mu_wd_norm = (norm_mu8 / norm_wd) * mu_wd

        vectors[f"wrong_depth_{wd}"] = mu_wd
        vectors[f"wrong_depth_norm_{wd}"] = mu_wd_norm

        # Check norm matching invariant
        assert abs(np.linalg.norm(mu_wd_norm) - norm_mu8) < 1e-5, f"Norm matching failed for depth {wd}"

    # 3. Coordinate permutations of mu_8
    for s in perm_seeds:
        rng = np.random.RandomState(s)
        perm = rng.permutation(embed_dim)
        # Bijective permutation check
        assert set(perm) == set(range(embed_dim)), f"Permutation not bijective for seed {s}"
        mu_perm = mu_8[perm].copy()
        assert abs(np.linalg.norm(mu_perm) - norm_mu8) < 1e-5, f"Norm mismatch in perm seed {s}"
        vectors[f"coord_perm_seed_{s}"] = mu_perm

    # 4. Sign-flipped variants
    # Full inversion
    vectors["sign_flip_100%"] = (-mu_8).copy()
    assert abs(np.linalg.norm(vectors["sign_flip_100%"]) - norm_mu8) < 1e-5

    # Partial sign flips (25%, 50%, 75%)
    sign_rng = np.random.RandomState(sign_seed)
    perm_dim = sign_rng.permutation(embed_dim)
    for frac_val, frac_label in [(0.25, "25%"), (0.50, "50%"), (0.75, "75%")]:
        n_flip = int(round(frac_val * embed_dim))
        flip_dims = perm_dim[:n_flip]
        sign_mask = np.ones(embed_dim, dtype=np.float32)
        sign_mask[flip_dims] = -1.0
        mu_partial_flip = (mu_8 * sign_mask).astype(np.float32)

        assert abs(np.linalg.norm(mu_partial_flip) - norm_mu8) < 1e-5
        assert np.sum(sign_mask == -1.0) == n_flip
        vectors[f"sign_flip_{frac_label}"] = mu_partial_flip

    # 5. Scale sweep
    for scale in scale_factors:
        vectors[f"scale_{scale:.2f}x"] = (scale * mu_8).astype(np.float32)

    # 6. Cosine-controlled directional sweep
    for alpha in target_cosines:
        for s in cosine_seeds:
            c_rng = np.random.RandomState(s)
            v = c_rng.randn(embed_dim).astype(np.float32)
            # Orthogonalize against hat_mu8
            v_proj = np.dot(v, hat_mu8) * hat_mu8
            v_perp = v - v_proj
            v_perp_norm = np.linalg.norm(v_perp)
            assert v_perp_norm > 1e-7, f"Degenerate random vector in cosine seed {s}"
            hat_v = v_perp / v_perp_norm

            # Construct unit vector u
            u = alpha * hat_mu8 + np.sqrt(max(0.0, 1.0 - alpha**2)) * hat_v
            r_alpha = (norm_mu8 * u).astype(np.float32)

            actual_cos = cosine_sim(r_alpha, mu_8)
            actual_norm = float(np.linalg.norm(r_alpha))

            assert abs(actual_cos - alpha) < 1e-3, f"Cosine mismatch: target {alpha}, got {actual_cos}"
            assert abs(actual_norm - norm_mu8) < 1e-5, f"Norm mismatch: target {norm_mu8}, got {actual_norm}"

            vectors[f"cosine_{alpha:.2f}_seed_{s}"] = r_alpha

    # 7. Exploratory variance-normalized prototype
    z_raw = (mu_8 / (sigma_8 + 1e-6)).astype(np.float32)
    norm_z = float(np.linalg.norm(z_raw))
    z_rescaled = ((norm_mu8 / norm_z) * z_raw).astype(np.float32)
    assert abs(np.linalg.norm(z_rescaled) - norm_mu8) < 1e-5
    vectors["variance_normalized"] = z_rescaled

    # 8. Compute vector metadata table
    metadata_rows = []
    for vec_name, vec in vectors.items():
        v_norm = float(np.linalg.norm(vec))
        cos_to_mu8 = cosine_sim(vec, mu_8)
        euc_dist_mu8 = float(np.linalg.norm(vec - mu_8))
        corr_mu8 = pearson_corr(vec, mu_8)

        row = {
            "model_name": model_name,
            "vector_name": vec_name,
            "norm": v_norm,
            "cosine_to_mu8": cos_to_mu8,
            "euclidean_dist_to_mu8": euc_dist_mu8,
            "correlation_to_mu8": corr_mu8,
        }
        for d in [5, 6, 7, 8, 9, 10]:
            row[f"cosine_to_mu{d}"] = cosine_sim(vec, depth_means[d])
        metadata_rows.append(row)

    return {
        "model_name": model_name,
        "embed_dim": embed_dim,
        "norm_mu8": norm_mu8,
        "mu_8": mu_8,
        "sigma_8": sigma_8,
        "depth_means": depth_means,
        "depth_sigmas": depth_sigmas,
        "vectors": vectors,
        "metadata_rows": metadata_rows
    }
