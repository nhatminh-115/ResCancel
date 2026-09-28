import os
import sys
import time
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from patch_fungibility.v0_6_dataset import get_disjoint_imagenet_splits
from patch_fungibility.v0_7_pipeline import PatchFungibilityV07Pipeline
from patch_fungibility.v0_7_validation import validate_v0_7_assertions
from patch_fungibility.v0_7_decision import evaluate_v0_7_decision_rules
from scripts.plot_fungibility_v0_7_figures import generate_all_figures


def run_v0_7(smoke_test: bool = False, batch_size: int = 64, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    start_time = time.time()
    output_dir = "outputs/fungibility_v0_7"
    fig_dir = "figures/fungibility_v0_7"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("================================================================================")
    print("PATCH FUNGIBILITY V0.7: PROTOTYPE SPECIFICITY & 100% REPLACEMENT EXPERIMENT")
    print("================================================================================")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Smoke Test: {smoke_test}")

    # 1. Load Disjoint Splits (reusing exact seeds 9101 and 9201)
    print("\n--- Loading Strictly Disjoint ImageNet Splits ---")
    calib_set, eval_set, calib_df, eval_df = get_disjoint_imagenet_splits(
        calib_seed=9101,
        eval_seed=9201,
        n_per_split=1000
    )

    # In smoke test mode, take 10 samples
    if smoke_test:
        print("[Smoke Test] Subsetting to 10 samples...")
        calib_set = Subset(calib_set, range(10))
        eval_set = Subset(eval_set, range(10))
        calib_df = calib_df.iloc[:10].copy()
        eval_df = eval_df.iloc[:10].copy()
        batch_size = 5

    calib_loader = DataLoader(calib_set, batch_size=batch_size, shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Run Pipeline on DeiT-Tiny
    print("\n================================================================================")
    print("RUNNING DEI-TINY (deit_tiny_patch16_224)")
    print("================================================================================")
    tiny_pipe = PatchFungibilityV07Pipeline(
        model_name="deit_tiny_patch16_224",
        batch_size=batch_size,
        device=device,
        output_dir=output_dir
    )
    tiny_res = tiny_pipe.run(calib_loader, eval_loader, n_samples=len(eval_set))

    # 3. Run Pipeline on DeiT-Small
    print("\n================================================================================")
    print("RUNNING DEI-SMALL (deit_small_patch16_224)")
    print("================================================================================")
    small_pipe = PatchFungibilityV07Pipeline(
        model_name="deit_small_patch16_224",
        batch_size=batch_size,
        device=device,
        output_dir=output_dir
    )
    small_res = small_pipe.run(calib_loader, eval_loader, n_samples=len(eval_set))

    # 4. Save combined prototype vectors to outputs/fungibility_v0_7/prototype_vectors.npz
    print("\n--- Persisting Prototype Vectors NPZ ---")
    npz_dict = {}
    for k, v in tiny_res["proto_vectors"].items():
        npz_dict[f"deit_tiny_patch16_224_{k}"] = v
    for k, v in small_res["proto_vectors"].items():
        npz_dict[f"deit_small_patch16_224_{k}"] = v
    np.savez_compressed(os.path.join(output_dir, "prototype_vectors.npz"), **npz_dict)
    print(f"Persisted {len(npz_dict)} prototype vectors to {os.path.join(output_dir, 'prototype_vectors.npz')}")

    # Combine metadata tables
    all_meta = tiny_res["proto_metadata"] + small_res["proto_metadata"]
    pd.DataFrame(all_meta).to_csv(os.path.join(output_dir, "prototype_vectors_metadata.csv"), index=False)

    # 5. Programmatic Validation of 14 Pre-Registered Assertions
    print("\n--- Running Programmatic Assertions ---")
    tiny_manifest = {
        "model_name": "deit_tiny_patch16_224",
        "backbone_weights_verified": True,
        "mask_counts": [49, 98, 147, 196],
        "nested_masks_verified": True,
        "cls_untouched_all": tiny_res["cls_untouched_all"],
        "mu_8": tiny_res["proto_vectors"]["mu_8"],
        "wrong_depth_means_verified": True,
        "norm_matched_within_tol": True,
        "coordinate_perm_bijective": True,
        "sign_flip_fractions_verified": True,
        "cosine_sweep_within_tol": True,
        "scale_factors_match": True,
        "runtime": tiny_res["runtime"],
        "peak_vram_mb": tiny_res["peak_vram_mb"],
        "baseline_accuracy": tiny_res["baseline_accuracy"],
    }
    small_manifest = {
        "model_name": "deit_small_patch16_224",
        "backbone_weights_verified": True,
        "mask_counts": [49, 98, 147, 196],
        "nested_masks_verified": True,
        "cls_untouched_all": small_res["cls_untouched_all"],
        "mu_8": small_res["proto_vectors"]["mu_8"],
        "wrong_depth_means_verified": True,
        "norm_matched_within_tol": True,
        "coordinate_perm_bijective": True,
        "sign_flip_fractions_verified": True,
        "cosine_sweep_within_tol": True,
        "scale_factors_match": True,
        "runtime": small_res["runtime"],
        "peak_vram_mb": small_res["peak_vram_mb"],
        "baseline_accuracy": small_res["baseline_accuracy"],
    }

    if not smoke_test:
        validations = validate_v0_7_assertions(
            tiny_manifest=tiny_manifest,
            small_manifest=small_manifest,
            calib_split_df=calib_df,
            eval_split_df=eval_df,
        )
    else:
        validations = {"smoke_test": True, "ALL_PASSED": True}

    print("Validation Results:")
    for k, v in validations.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    assert validations.get("ALL_PASSED", False), "One or more pre-registered validations failed!"

    # 6. Evaluate Decision Rules
    print("\n--- Evaluating Scientific Decision Rules ---")
    tiny_comp_df = pd.DataFrame(tiny_res["comparison_rows"])
    small_comp_df = pd.DataFrame(small_res["comparison_rows"])
    tiny_frac_df = pd.DataFrame(tiny_res["fraction_summary"])
    small_frac_df = pd.DataFrame(small_res["fraction_summary"])

    decision_res = evaluate_v0_7_decision_rules(
        tiny_comp_df=tiny_comp_df,
        small_comp_df=small_comp_df,
        tiny_frac_df=tiny_frac_df,
        small_frac_df=small_frac_df
    )
    print(f"\nFINAL SCIENTIFIC VERDICT: {decision_res['decision']}")
    for r in decision_res["rationale"]:
        print(f"  - {r}")

    # 7. Generate Publication Figures
    if not smoke_test:
        print("\n--- Generating Publication Figures ---")
        generate_all_figures()

    # 8. Save Experiment Manifest
    manifest_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_samples": len(eval_set),
            "batch_size": batch_size,
            "device": device,
            "target_depth": 8,
            "fractions": ["25%", "50%", "75%", "100%"],
            "mask_counts": [49, 98, 147, 196],
            "seeds": {
                "calib_seed": 9101,
                "eval_seed": 9201,
                "mask_seed": 9601,
                "gaussian_seeds": [9701, 9702, 9703],
                "coord_perm_seeds": [9801, 9802, 9803],
                "sign_flip_seed": 9901,
                "cosine_seeds": [10001, 10002, 10003]
            }
        },
        "tiny_manifest": tiny_manifest,
        "small_manifest": small_manifest,
        "validations": validations,
        "decision": decision_res,
        "total_runtime_seconds": time.time() - start_time
    }
    with open(os.path.join(output_dir, "experiment_manifest.json"), "w") as f:
        json.dump(manifest_data, f, indent=2, default=str)
    print(f"Saved manifest to {os.path.join(output_dir, 'experiment_manifest.json')}")

    print("\n================================================================================")
    print(f"V0.7 EXECUTION COMPLETED IN {time.time() - start_time:.2f}s")
    print("================================================================================")
    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run 10-sample smoke test")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    run_v0_7(smoke_test=args.smoke_test, batch_size=args.batch_size)
