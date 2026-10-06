"""
scripts/profile_batched_operator.py

Comprehensive Batched Throughput and Component Profiler:
Evaluates batch sizes BS in {1, 2, 4, 8, 16, 32, 64} (and 96, 128 for small backbones).
Architectures:
- DeiT-Tiny (l=6, 8, B in {98, 49, 32})
- DeiT-Small (l=6, 8, B in {98, 49, 32})
- ViT-Base (l=5, 7, B in {98, 49, 32})
- DINOv2-S/14 (l=6, 8, B in {128, 64, 42})

Generates:
- outputs/fungibility_batched_operator/batch_throughput.csv
- outputs/fungibility_batched_operator/batch_runtime_breakdown.csv
- outputs/fungibility_batched_operator/predictor_batch_ablation.csv
- outputs/fungibility_batched_operator/solver_batch_ablation.csv
- outputs/fungibility_batched_operator/depth_batch_ablation.csv
"""

import os
import sys
import time
import math
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from patch_fungibility.dense_fraction_models import load_model_and_transform
from patch_fungibility.amortized_operator import (
    FactorizedModePredictor,
    extract_oracle_subspace,
    subspace_overlap
)
from patch_fungibility.practical_operator_compression import create_fixed_spatial_grouping
from patch_fungibility.batched_operator_compression import (
    forward_prefix_batched,
    forward_suffix_batched,
    batched_solve_amortized_carrier,
    batched_solve_diagonal_carrier,
    batched_solve_one_step_carrier,
    create_hybrid_dynamic_grouping,
    run_evit_pruning_batched,
    run_tome_bipartite_batched
)


def measure_cuda_latency_ms(fn, n_warmup: int = 10, n_runs: int = 30) -> Tuple[float, float]:
    """
    Measures median and interquartile range (dispersion) of execution time using CUDA events.
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]

    for i in range(n_runs):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    median_t = float(np.median(times))
    iqr_t = float(np.percentile(times, 75) - np.percentile(times, 25))
    return median_t, iqr_t


def compute_theoretical_flops(model_key: str, seq_len: int, depth_start: int, total_depth: int, D: int) -> float:
    """
    Computes approximate FLOPs for suffix transformer blocks with given sequence length.
    Each block has:
    - Self-attention: 2 * seq_len * D^2 (qkv) + 2 * seq_len^2 * D (attn matrix & out) + seq_len * D^2 (proj)
    - MLP (4x): 8 * seq_len * D^2
    Total per block ~ 4 * seq_len * D^2 + 2 * seq_len^2 * D + 8 * seq_len * D^2
    """
    n_suffix_blocks = total_depth - depth_start
    flops_per_block = 12.0 * seq_len * (D ** 2) + 2.0 * (seq_len ** 2) * D
    return n_suffix_blocks * flops_per_block


def run_batched_throughput_profiling(
    models_dict: Dict[str, nn.Module],
    predictors_dict: Dict[Tuple[str, int, int], nn.Module],
    device: torch.device = torch.device("cuda")
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs systematic profiling across batch sizes, depths, budgets, and methods.
    """
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    # Model architectural specifications
    arch_specs = {
        "deit_tiny": {
            "name": "DeiT-Tiny", "N": 196, "D": 192, "total_depth": 12,
            "depths": [6, 8], "budgets": [98, 49, 32],
            "extra_bs": [96, 128]
        },
        "deit_small": {
            "name": "DeiT-Small", "N": 196, "D": 384, "total_depth": 12,
            "depths": [6, 8], "budgets": [98, 49, 32],
            "extra_bs": [96]
        },
        "vit_base": {
            "name": "ViT-Base", "N": 196, "D": 768, "total_depth": 12,
            "depths": [5, 7], "budgets": [98, 49, 32],
            "extra_bs": []
        },
        "dinov2": {
            "name": "DINOv2-S", "N": 256, "D": 384, "total_depth": 12,
            "depths": [6, 8], "budgets": [128, 64, 42],
            "extra_bs": []
        }
    }

    throughput_rows = []
    breakdown_rows = []
    predictor_rows = []
    solver_rows = []
    depth_rows = []

    for model_key, spec in arch_specs.items():
        model = models_dict[model_key].to(device).eval()
        N = spec["N"]
        D = spec["D"]
        total_depth = spec["total_depth"]
        tested_bs = batch_sizes + spec["extra_bs"]

        print(f"\n=======================================================")
        print(f"PROFILING ARCHITECTURE: {spec['name']} ({model_key})")
        print(f"=======================================================")

        # Base clean throughput across batch sizes
        clean_latencies = {}
        for bs in tested_bs:
            dummy_x = torch.randn(bs, 3, 224, 224, device=device)
            try:
                med_t, iqr_t = measure_cuda_latency_ms(lambda: model(dummy_x), n_warmup=10, n_runs=25)
                clean_latencies[bs] = med_t
                vram_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                vram_res = torch.cuda.memory_reserved(device) / (1024 ** 2)
                throughput = (1000.0 * bs) / med_t

                throughput_rows.append({
                    "architecture": model_key,
                    "batch_size": bs,
                    "depth": total_depth,
                    "budget": N,
                    "method": "Clean",
                    "batch_latency_ms": med_t,
                    "batch_latency_iqr_ms": iqr_t,
                    "per_image_latency_ms": med_t / bs,
                    "img_per_sec": throughput,
                    "speedup_vs_clean": 1.0,
                    "vram_allocated_mb": vram_alloc,
                    "vram_reserved_mb": vram_res,
                    "flops_suffix": compute_theoretical_flops(model_key, 1 + N, 0, total_depth, D),
                    "token_count_suffix": 1 + N
                })
                print(f"[{spec['name']}][BS={bs:3d}][Clean] Latency: {med_t:6.2f} ms ({med_t/bs:5.2f} ms/img) | Throughput: {throughput:6.1f} img/s")
            except torch.cuda.OutOfMemoryError:
                print(f"[{spec['name']}][BS={bs:3d}][Clean] OOM encountered, skipping higher batch sizes.")
                break

        # -------------------------------------------------------------
        # Interventions across depths & budgets
        # -------------------------------------------------------------
        for l in spec["depths"]:
            for B in spec["budgets"]:
                # Precompute spatial partition & multiplicities
                grid_sz = int(math.isqrt(N))
                S_sp, mults_sp, _ = create_fixed_spatial_grouping(N, B, grid_h=grid_sz, grid_w=grid_sz, device=device)
                mults_sp_cls = torch.cat([torch.tensor([1.0], device=device), mults_sp])

                predictor = predictors_dict.get((model_key, l, 32))
                if predictor is not None:
                    predictor = predictor.to(device).eval()

                for bs in tested_bs:
                    if bs not in clean_latencies:
                        continue # OOM at clean
                    dummy_x = torch.randn(bs, 3, 224, 224, device=device)

                    # Methods to profile
                    methods = [
                        "Spatial_Group_Mean",
                        "Fast_Amortized_Operator",
                        "Fast_Operator_Diagonal",
                        "Fast_Operator_One_Step",
                        "Hybrid_Group_Mean",
                        "Hybrid_Amortized_Operator",
                        "Attention_Pruning_EViT",
                        "Token_Merging_ToMe"
                    ]

                    def get_v_pred(h_l):
                        if predictor is not None:
                            return predictor(h_l)
                        v = torch.randn(bs, N * D, 32, device=device)
                        v, _ = torch.linalg.qr(v)
                        return v

                    for m_name in methods:
                        def run_pipeline():
                            # 1. Prefix
                            h_l = forward_prefix_batched(model, model_key, l, dummy_x)
                            cls_tok = h_l[:, :1, :]
                            patches = h_l[:, 1:, :]

                            if m_name == "Spatial_Group_Mean":
                                C_mean = torch.bmm(S_sp.unsqueeze(0).expand(bs, -1, -1).transpose(1, 2), patches) / mults_sp.view(1, B, 1)
                                h_comp = torch.cat([cls_tok, C_mean], dim=1)
                                return forward_suffix_batched(model, model_key, l, h_comp, mults_sp_cls, N, route="reference")

                            elif m_name == "Fast_Amortized_Operator":
                                V_pred = get_v_pred(h_l)
                                res = batched_solve_amortized_carrier(patches, S_sp, mults_sp, V_pred)
                                h_comp = torch.cat([cls_tok, res["C_opt"]], dim=1)
                                return forward_suffix_batched(model, model_key, l, h_comp, mults_sp_cls, N, route="reference")

                            elif m_name == "Fast_Operator_Diagonal":
                                V_pred = get_v_pred(h_l)
                                res = batched_solve_diagonal_carrier(patches, S_sp, mults_sp, V_pred)
                                h_comp = torch.cat([cls_tok, res["C_opt"]], dim=1)
                                return forward_suffix_batched(model, model_key, l, h_comp, mults_sp_cls, N, route="reference")

                            elif m_name == "Fast_Operator_One_Step":
                                V_pred = get_v_pred(h_l)
                                res = batched_solve_one_step_carrier(patches, S_sp, mults_sp, V_pred)
                                h_comp = torch.cat([cls_tok, res["C_opt"]], dim=1)
                                return forward_suffix_batched(model, model_key, l, h_comp, mults_sp_cls, N, route="reference")

                            elif m_name == "Hybrid_Group_Mean":
                                S_hyb, mults_hyb = create_hybrid_dynamic_grouping(patches, B=B, K_keep=16, grid_h=grid_sz, grid_w=grid_sz)
                                C_mean_hyb = torch.bmm(S_hyb.transpose(1, 2), patches) / mults_hyb.unsqueeze(-1).clamp(min=1.0)
                                h_comp = torch.cat([cls_tok, C_mean_hyb], dim=1)
                                mults_hyb_cls = torch.cat([torch.ones(bs, 1, device=device), mults_hyb], dim=1)
                                return forward_suffix_batched(model, model_key, l, h_comp, mults_hyb_cls[0], N, route="reference")

                            elif m_name == "Hybrid_Amortized_Operator":
                                S_hyb, mults_hyb = create_hybrid_dynamic_grouping(patches, B=B, K_keep=16, grid_h=grid_sz, grid_w=grid_sz)
                                V_pred = get_v_pred(h_l)
                                res = batched_solve_amortized_carrier(patches, S_hyb, mults_hyb, V_pred)
                                h_comp = torch.cat([cls_tok, res["C_opt"]], dim=1)
                                mults_hyb_cls = torch.cat([torch.ones(bs, 1, device=device), mults_hyb], dim=1)
                                return forward_suffix_batched(model, model_key, l, h_comp, mults_hyb_cls[0], N, route="reference")

                            elif m_name == "Attention_Pruning_EViT":
                                h_evit, mults_evit = run_evit_pruning_batched(h_l, B=B)
                                return forward_suffix_batched(model, model_key, l, h_evit, mults_evit, N, route="unweighted")

                            elif m_name == "Token_Merging_ToMe":
                                h_tome, mults_tome = run_tome_bipartite_batched(h_l, target_tokens=B)
                                return forward_suffix_batched(model, model_key, l, h_tome, mults_tome, N, route="unweighted")

                        try:
                            warmup = 2 if m_name == "Token_Merging_ToMe" else 6
                            runs = 3 if m_name == "Token_Merging_ToMe" else 15
                            med_t, iqr_t = measure_cuda_latency_ms(run_pipeline, n_warmup=warmup, n_runs=runs)
                            vram_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                            vram_res = torch.cuda.memory_reserved(device) / (1024 ** 2)
                            throughput = (1000.0 * bs) / med_t
                            clean_t = clean_latencies[bs]
                            speedup = clean_t / med_t

                            throughput_rows.append({
                                "architecture": model_key,
                                "batch_size": bs,
                                "depth": l,
                                "budget": B,
                                "method": m_name,
                                "batch_latency_ms": med_t,
                                "batch_latency_iqr_ms": iqr_t,
                                "per_image_latency_ms": med_t / bs,
                                "img_per_sec": throughput,
                                "speedup_vs_clean": speedup,
                                "vram_allocated_mb": vram_alloc,
                                "vram_reserved_mb": vram_res,
                                "flops_suffix": compute_theoretical_flops(model_key, 1 + B, l, total_depth, D),
                                "token_count_suffix": 1 + B
                            })
                        except torch.cuda.OutOfMemoryError:
                            print(f"[{spec['name']}][BS={bs}][{m_name}] OOM, skipping.")
                            continue

                # -------------------------------------------------------------
                # 2. Granular Runtime Breakdown (Prefix, Predictor, Grouping, Solve, Suffix)
                # -------------------------------------------------------------
                for bs in [1, 8, 16, 32, 64]:
                    if bs not in clean_latencies:
                        continue
                    dummy_x = torch.randn(bs, 3, 224, 224, device=device)

                    # Stage 1: Prefix
                    t_prefix, _ = measure_cuda_latency_ms(lambda: forward_prefix_batched(model, model_key, l, dummy_x))
                    h_l = forward_prefix_batched(model, model_key, l, dummy_x)
                    cls_tok = h_l[:, :1, :]
                    patches = h_l[:, 1:, :]

                    # Stage 2: Predictor
                    if predictor is not None:
                        t_pred, _ = measure_cuda_latency_ms(lambda: predictor(h_l))
                        V_pred = predictor(h_l)
                    else:
                        t_pred = 0.0
                        v = torch.randn(bs, N * D, 32, device=device)
                        V_pred, _ = torch.linalg.qr(v)

                    # Stage 3: Grouping (Spatial vs Hybrid)
                    t_grp_sp, _ = measure_cuda_latency_ms(lambda: (S_sp.unsqueeze(0).expand(bs, -1, -1).transpose(1, 2) @ patches))
                    t_grp_hyb, _ = measure_cuda_latency_ms(lambda: create_hybrid_dynamic_grouping(patches, B=B, K_keep=16, grid_h=grid_sz, grid_w=grid_sz))

                    # Stage 4: Solvers
                    t_sol_exact, _ = measure_cuda_latency_ms(lambda: batched_solve_amortized_carrier(patches, S_sp, mults_sp, V_pred))
                    t_sol_diag, _ = measure_cuda_latency_ms(lambda: batched_solve_diagonal_carrier(patches, S_sp, mults_sp, V_pred))
                    t_sol_one, _ = measure_cuda_latency_ms(lambda: batched_solve_one_step_carrier(patches, S_sp, mults_sp, V_pred))

                    # Stage 5: Suffix
                    res_exact = batched_solve_amortized_carrier(patches, S_sp, mults_sp, V_pred)
                    h_comp = torch.cat([cls_tok, res_exact["C_opt"]], dim=1)
                    t_suffix, _ = measure_cuda_latency_ms(lambda: forward_suffix_batched(model, model_key, l, h_comp, mults_sp_cls, N, route="reference"))

                    breakdown_rows.append({
                        "architecture": model_key,
                        "batch_size": bs,
                        "depth": l,
                        "budget": B,
                        "prefix_ms": t_prefix,
                        "predictor_ms": t_pred,
                        "spatial_grouping_ms": t_grp_sp,
                        "hybrid_grouping_ms": t_grp_hyb,
                        "exact_solve_ms": t_sol_exact,
                        "diag_solve_ms": t_sol_diag,
                        "onestep_solve_ms": t_sol_one,
                        "suffix_ms": t_suffix,
                        "clean_total_ms": clean_latencies[bs],
                        "operator_total_ms": t_prefix + t_pred + t_grp_sp + t_sol_exact + t_suffix
                    })

                    # Solver ablation record
                    for s_name, s_lat in [("Exact_Cholesky", t_sol_exact), ("Diagonal", t_sol_diag), ("One_Step", t_sol_one)]:
                        solver_rows.append({
                            "architecture": model_key,
                            "batch_size": bs,
                            "depth": l,
                            "budget": B,
                            "solver": s_name,
                            "solve_batch_latency_ms": s_lat,
                            "solve_per_image_ms": s_lat / bs,
                            "relative_speedup_vs_exact": t_sol_exact / max(1e-4, s_lat)
                        })

                # -------------------------------------------------------------
                # 3. Predictor Scaling vs Static (Rank-16 vs Rank-32 vs Static)
                # -------------------------------------------------------------
                pred_r16 = predictors_dict.get((model_key, l, 16))
                for bs in [1, 8, 16, 32, 64]:
                    if bs not in clean_latencies:
                        continue
                    dummy_x = torch.randn(bs, 3, 224, 224, device=device)
                    h_l = forward_prefix_batched(model, model_key, l, dummy_x)

                    # Static rank-32
                    V_static = torch.randn(bs, N * D, 32, device=device)
                    t_static = 0.0 # Static projection matrix precomputed

                    # Predicted rank-16
                    if pred_r16 is not None:
                        t_p16, _ = measure_cuda_latency_ms(lambda: pred_r16(h_l))
                    else:
                        t_p16 = 0.0

                    # Predicted rank-32
                    if predictor is not None:
                        t_p32, _ = measure_cuda_latency_ms(lambda: predictor(h_l))
                    else:
                        t_p32 = 0.0

                    predictor_rows.append({
                        "architecture": model_key,
                        "batch_size": bs,
                        "depth": l,
                        "static_r32_ms": t_static,
                        "pred_r16_ms": t_p16,
                        "pred_r16_per_img_ms": t_p16 / bs,
                        "pred_r32_ms": t_p32,
                        "pred_r32_per_img_ms": t_p32 / bs,
                        "amortization_ratio_64_vs_1": (t_p32 / bs)
                    })

                # Record depth interaction
                for bs in [1, 8, 16, 32, 64]:
                    if bs in clean_latencies:
                        depth_rows.append({
                            "architecture": model_key,
                            "depth": l,
                            "budget": B,
                            "batch_size": bs,
                            "clean_latency_ms": clean_latencies[bs],
                            "suffix_fraction": (total_depth - l) / float(total_depth)
                        })

    df_throughput = pd.DataFrame(throughput_rows)
    df_breakdown = pd.DataFrame(breakdown_rows)
    df_pred = pd.DataFrame(predictor_rows)
    df_solver = pd.DataFrame(solver_rows)
    df_depth = pd.DataFrame(depth_rows)

    return df_throughput, df_breakdown, df_pred, df_solver, df_depth


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Systematic Batched Operator Profiler on {device}...")

    # Load models
    models = {
        "deit_tiny": load_model_and_transform("deit_tiny", device)[0],
        "deit_small": load_model_and_transform("deit_small", device)[0],
        "vit_base": load_model_and_transform("vit_base", device)[0],
        "dinov2": load_model_and_transform("dinov2", device)[0],
    }

    # Load predictors
    predictors = {}
    ckp_dir = os.path.join(REPO_ROOT, "outputs", "fungibility_amortized_operator", "checkpoints")
    for (m_key, l, d) in [("deit_tiny", 6, 192), ("deit_tiny", 8, 192),
                          ("deit_small", 6, 384), ("deit_small", 8, 384),
                          ("vit_base", 5, 768), ("vit_base", 7, 768),
                          ("dinov2", 6, 384), ("dinov2", 8, 384)]:
        for rank in [16, 32]:
            ckp_path = os.path.join(ckp_dir, f"predictor_{m_key}_l{l}_rank{rank}.pt")
            pred = FactorizedModePredictor(N=256 if m_key == "dinov2" else 196, D=d, r=rank)
            if os.path.exists(ckp_path):
                pred.load_state_dict(torch.load(ckp_path, map_location=device))
            pred = pred.to(device).eval()
            predictors[(m_key, l, rank)] = pred

    df_tp, df_bk, df_pred, df_solv, df_dp = run_batched_throughput_profiling(models, predictors, device=device)

    out_dir = os.path.join(REPO_ROOT, "outputs", "fungibility_batched_operator")
    os.makedirs(out_dir, exist_ok=True)

    df_tp.to_csv(os.path.join(out_dir, "batch_throughput.csv"), index=False)
    df_bk.to_csv(os.path.join(out_dir, "batch_runtime_breakdown.csv"), index=False)
    df_pred.to_csv(os.path.join(out_dir, "predictor_batch_ablation.csv"), index=False)
    df_solv.to_csv(os.path.join(out_dir, "solver_batch_ablation.csv"), index=False)
    df_dp.to_csv(os.path.join(out_dir, "depth_batch_ablation.csv"), index=False)

    print(f"\nAll profiling CSVs successfully generated and saved to {out_dir}!")


if __name__ == "__main__":
    main()
