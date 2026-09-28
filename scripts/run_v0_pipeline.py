import os
import sys
import argparse
import time
import json
import torch
import pandas as pd

# Add repo root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rescancel.pipeline import ResCancelPipeline
from rescancel.dataset import get_imagenet_val_loader
from rescancel.audit_validation import validate_v0_1_results
from scripts.plot_figures import (
    plot_cancellation_landscape,
    plot_geometry_joint_distribution,
    plot_matched_comparison,
    plot_causal_intervention
)


def run_experiment(args):
    print("=" * 75)
    print("  ResCancel V0.1: Repaired Mechanistic Falsification Pipeline")
    print("=" * 75)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Models to evaluate: {args.models}")
    print(f"Sample count: {args.num_samples}, Batch size: {args.batch_size}, Seed: {args.seed}")
    print(f"Output Directory: {args.output_dir}, Figures Directory: {args.figures_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # Initialize data loader (stratified 1 per class for 1000 samples)
    loader = get_imagenet_val_loader(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed
    )

    manifest = {
        "version": "V0.1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_frozen": "docs/V0_PROTOCOL.md",
        "audit_document": "docs/V0_1_AUDIT.md",
        "models": args.models,
        "sample_count": args.num_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "random_control_seeds": [2501, 2502, 2503],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "frozen_criteria": {
            "cos_threshold": -0.60,
            "r_threshold": 0.60,
            "q_threshold": 0.85,
            "c_l1_threshold": 2.00,
            "alphas": [1.0, 0.75, 0.50, 0.25],
            "caliper_x_pct": 0.15,
            "caliper_delta_pct": 0.15,
            "caliper_margin": 0.50
        },
        "package_versions": {
            "torch": torch.__version__,
            "timm": getattr(__import__("timm"), "__version__", "unknown"),
            "pandas": pd.__version__,
            "statsmodels": getattr(__import__("statsmodels"), "__version__", "unknown"),
        },
        "results": {}
    }

    t_start = time.time()

    for model_name in args.models:
        print("\n" + "-" * 75)
        print(f"Processing Model: {model_name}")
        print("-" * 75)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        pipeline = ResCancelPipeline(
            model_name=model_name,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=str(device),
            seed=args.seed,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir
        )

        inst_vit, raw_model = pipeline.load_model()

        # Step 1 & 2: Clean and Perturbed forward passes (separating CLS events from patch summaries)
        img_df, cls_df, patch_df = pipeline.run_clean_and_perturbed_analysis(inst_vit, loader)

        # Save machine-readable tables
        img_csv_path = os.path.join(args.output_dir, f"{model_name}_image_predictions.csv")
        cls_csv_path = os.path.join(args.output_dir, f"{model_name}_cls_residual_events.csv")
        patch_csv_path = os.path.join(args.output_dir, f"{model_name}_patch_summary.csv")
        img_df.to_csv(img_csv_path, index=False)
        cls_df.to_csv(cls_csv_path, index=False)
        patch_df.to_csv(patch_csv_path, index=False)
        print(f"Saved: {img_csv_path}")
        print(f"Saved: {cls_csv_path}")
        print(f"Saved: {patch_csv_path}")

        # Step 3: Confound-Controlled & Matched Analysis
        confound_res = pipeline.analyze_matched_confounds(img_df, cls_df)
        matched_df = confound_res["matched_df"]
        matched_csv_path = os.path.join(args.output_dir, f"{model_name}_matched_controls.csv")
        matched_df.to_csv(matched_csv_path, index=False)
        print(f"Saved: {matched_csv_path}")

        # Step 4: Causal Intervention & Controls (strictly only_extreme=True, seeds 2501, 2502, 2503)
        intervention_df, summary_stats = pipeline.run_causal_falsification(inst_vit, loader)
        intervention_csv_path = os.path.join(args.output_dir, f"{model_name}_intervention_sweep.csv")
        intervention_df.to_csv(intervention_csv_path, index=False)
        print(f"Saved: {intervention_csv_path}")

        # Step 5: Programmatic Audit Validation Checks
        print(f"[{model_name}] Validating protocol compliance assertions...")
        validation_checks = validate_v0_1_results(
            matched_df=matched_df,
            intervention_df=intervention_df,
            reg_results=confound_res["image_level_regression"],
            cls_event_df=cls_df,
            expected_n_images=args.num_samples
        )

        # Step 6: Decision Rule Evaluation
        decision_info = pipeline.evaluate_decision_rule(confound_res, intervention_df)
        print(f"\n[V0.1 DECISION FOR {model_name}]: {decision_info['decision']}")
        for r in decision_info["rationale"]:
            print(f"  -> {r}")

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        print(f"Peak VRAM: {peak_vram_mb:.1f} MB (Budget: <= 8192 MB)")

        # Save model-specific results into manifest
        manifest["results"][model_name] = {
            "clean_accuracy": float(img_df["clean_correct"].mean()),
            "noise_flip_rate": float(img_df["noise_flipped"].mean()),
            "blur_flip_rate": float(img_df["blur_flipped"].mean()),
            "total_cls_events": len(cls_df),
            "extreme_cls_event_count": int(cls_df["is_extreme"].sum()),
            "extreme_cls_event_rate": float(cls_df["is_extreme"].mean()),
            "matching_stats": confound_res["matching_stats"],
            "matched_margin_effect": confound_res["matched_margin_effect"],
            "matched_fragility_effect": confound_res["matched_fragility_effect"],
            "image_level_regression": confound_res["image_level_regression"],
            "causal_intervention_summary": summary_stats,
            "decision": decision_info,
            "validation_checks": validation_checks,
            "peak_vram_mb": peak_vram_mb
        }

        # Step 7: Generate Visualizations
        print(f"Generating publication-quality figures for {model_name}...")
        fig1_path = os.path.join(args.figures_dir, f"{model_name}_cancellation_landscape.png")
        fig2_path = os.path.join(args.figures_dir, f"{model_name}_geometry_joint_distribution.png")
        fig3_path = os.path.join(args.figures_dir, f"{model_name}_confound_matching.png")
        fig4_path = os.path.join(args.figures_dir, f"{model_name}_causal_intervention.png")

        plot_cancellation_landscape(cls_df, model_name, fig1_path)
        plot_geometry_joint_distribution(cls_df, model_name, fig2_path)
        plot_matched_comparison(matched_df, model_name, fig3_path)
        plot_causal_intervention(intervention_df, model_name, fig4_path)

    total_runtime_s = time.time() - t_start
    manifest["total_runtime_seconds"] = total_runtime_s
    manifest_path = os.path.join(args.output_dir, "experiment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest successfully written to: {manifest_path}")
    print(f"Total V0.1 experiment runtime: {total_runtime_s:.2f}s")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResCancel V0.1 Repaired Mechanistic Falsification Runner")
    parser.add_argument("--models", nargs="+", default=["deit_tiny_patch16_224", "deit_small_patch16_224"],
                        help="List of model names to evaluate")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of ImageNet validation samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    parser.add_argument("--output-dir", type=str, default="outputs/v0_1", help="Directory for output tables")
    parser.add_argument("--figures-dir", type=str, default="figures/v0_1", help="Directory for generated plots")
    
    args = parser.parse_args()
    run_experiment(args)
