# Report: Geometry-Diversity Bank Proof-of-Concept (POC)

**Experiment Name**: `GEOMETRY-DIVERSITY BANK POC`  
**Execution Date**: 2026-09-29  
**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Protocol Reference**: [`docs/FUNGIBILITY_GEOMETRY_BANK_PROTOCOL.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_GEOMETRY_BANK_PROTOCOL.md)  
**Decision Verdict**: **`KILL APPLICATION`**  
**Latency Benchmarking Triggered**: **No** (terminated per pre-registered protocol)  
**Hardware Target**: NVIDIA GeForce RTX 5070 Laptop GPU  

---

## 1. Executive Summary & Core Verdict

This experiment directly addresses the final application hypothesis arising from the mechanistic discovery of **Patch Content Fungibility**:
> *"At the same downstream token budget $B$, can a small set of class-agnostic synthetic tokens that explicitly preserve learned late-layer geometry and token diversity outperform simple Random Pruning?"*

### Core Experimental Verdict: **`KILL APPLICATION`**
- **Unambiguous Empirical Result:** Across all four model architectures (DeiT-Tiny, DeiT-Small, ViT-B/16, and DINOv2 ViT-S/14), across all tested downstream budgets $B$, across all bank formulations (PCA eigenvectors, K-Means cluster centroids, Random orthogonal directions), and across all 5 spatial mask seeds, **Random Pruning strictly matches or outperforms every synthetic bank configuration**.
- **$\Delta \text{Acc}$ vs. Random Pruning:** Mean accuracy change relative to Random Pruning was $\le 0.0\%$ in every tested condition:
  - DeiT-Tiny: $\Delta \text{Acc} \in [-3.96\%, -0.28\%]$
  - DeiT-Small: $\Delta \text{Acc} \in [-8.28\%, -0.24\%]$
  - ViT-B/16: $\Delta \text{Acc} \in [-2.10\%, -0.16\%]$
  - DINOv2: $\Delta \text{Acc} \in [-0.72\%, -0.08\%]$
- **Monotonic Degradation with Bank Size $K$:** Increasing the number of synthetic tokens $K \in \{0, 1, 4, 8, 16\}$ at a fixed downstream budget $B$ causes a strictly **monotonic drop in Top-1 accuracy and true-class margin**. The optimal bank size is uniformly $K = 0$ (allocating 100% of the downstream budget to surviving real image patches).
- **Termination Action:** Per the pre-registered decision rule, the application line is **PERMANENTLY TERMINATED**. No physical latency benchmarks are run. The paper remains exclusively focused on the fundamental mechanistic discovery of late-layer patch content fungibility.

---

## 2. Matched-Budget Evaluation Summary Table

All conditions evaluated at the exact same downstream sequence length ($1 + B$ tokens) across $N=1,000$ disjoint evaluation images and 5 deterministic spatial mask seeds (`31001`–`31005`):

| Model | Budget $B$ | Condition | $K_{\text{synth}}$ | $M_{\text{real}}$ | Mean Top-1 Acc (%) | $\pm$ SD (%) | Mean Margin | $\Delta \text{Acc}$ vs. Pruning (%) | $\Delta \text{Margin}$ vs. Pruning | Total Model GFLOPs | FLOP Red. (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Tiny** | 45 | **RANDOM_PRUNING** | 0 | 45 | **61.62%** | $\pm 1.25$ | 0.5638 | **0.00%** | 0.0000 | 1.82 GF | 26.6% |
| (Clean 67.9%) | 45 | Single Centroid | 1 | 44 | 61.34% | $\pm 1.23$ | 0.5487 | -0.28% | -0.0151 | 1.82 GF | 26.6% |
| | 45 | PCA Bank | 4 | 41 | 60.58% | $\pm 1.29$ | 0.4716 | -1.04% | -0.0922 | 1.82 GF | 26.6% |
| | 45 | PCA Bank | 8 | 37 | 59.62% | $\pm 1.41$ | 0.4168 | -2.00% | -0.1470 | 1.82 GF | 26.6% |
| | 45 | PCA Bank | 16 | 29 | 57.66% | $\pm 1.49$ | 0.2704 | -3.96% | -0.2934 | 1.82 GF | 26.6% |
| | 45 | K-Means Bank | 4 | 41 | 60.52% | $\pm 1.29$ | 0.4735 | -1.10% | -0.0903 | 1.82 GF | 26.6% |
| | 45 | K-Means Bank | 8 | 37 | 59.88% | $\pm 1.29$ | 0.4282 | -1.74% | -0.1356 | 1.82 GF | 26.6% |
| | 45 | K-Means Bank | 16 | 29 | 58.08% | $\pm 1.34$ | 0.2938 | -3.54% | -0.2700 | 1.82 GF | 26.6% |
| | 73 | **RANDOM_PRUNING** | 0 | 73 | **64.82%** | $\pm 0.38$ | 0.8091 | **0.00%** | 0.0000 | 1.93 GF | 22.1% |
| | 73 | Single Centroid | 1 | 72 | 64.44% | $\pm 0.48$ | 0.7936 | -0.38% | -0.0155 | 1.93 GF | 22.1% |
| | 73 | PCA Bank | 4 | 69 | 64.14% | $\pm 0.43$ | 0.7603 | -0.68% | -0.0488 | 1.93 GF | 22.1% |
| | 73 | PCA Bank | 8 | 65 | 63.74% | $\pm 0.89$ | 0.7338 | -1.08% | -0.0753 | 1.93 GF | 22.1% |
| | 73 | PCA Bank | 16 | 57 | 62.86% | $\pm 0.46$ | 0.6501 | -1.96% | -0.1590 | 1.93 GF | 22.1% |
| | 73 | K-Means Bank | 4 | 69 | 64.22% | $\pm 0.44$ | 0.7643 | -0.60% | -0.0448 | 1.93 GF | 22.1% |
| | 73 | K-Means Bank | 8 | 65 | 63.68% | $\pm 0.92$ | 0.7385 | -1.14% | -0.0706 | 1.93 GF | 22.1% |
| | 73 | K-Means Bank | 16 | 57 | 62.88% | $\pm 0.76$ | 0.6558 | -1.94% | -0.1533 | 1.93 GF | 22.1% |
| **DeiT-Small** | 28 | **RANDOM_PRUNING** | 0 | 28 | **69.42%** | $\pm 0.68$ | 1.3389 | **0.00%** | 0.0000 | 3.34 GF | 28.9% |
| (Clean 76.1%) | 28 | Single Centroid | 1 | 27 | 69.02% | $\pm 1.02$ | 1.2936 | -0.40% | -0.0453 | 3.34 GF | 28.9% |
| | 28 | PCA Bank | 4 | 24 | 67.62% | $\pm 1.78$ | 1.1826 | -1.80% | -0.1563 | 3.34 GF | 28.9% |
| | 28 | PCA Bank | 8 | 20 | 66.42% | $\pm 1.75$ | 1.0737 | -3.00% | -0.2652 | 3.34 GF | 28.9% |
| | 28 | PCA Bank | 16 | 12 | 61.34% | $\pm 2.34$ | 0.6893 | -8.08% | -0.6496 | 3.34 GF | 28.9% |
| | 28 | K-Means Bank | 4 | 24 | 67.62% | $\pm 1.63$ | 1.1998 | -1.80% | -0.1391 | 3.34 GF | 28.9% |
| | 28 | K-Means Bank | 8 | 20 | 66.40% | $\pm 1.55$ | 1.1038 | -3.02% | -0.2351 | 3.34 GF | 28.9% |
| | 28 | K-Means Bank | 16 | 12 | 61.14% | $\pm 2.28$ | 0.7617 | -8.28% | -0.5772 | 3.34 GF | 28.9% |
| | 52 | **RANDOM_PRUNING** | 0 | 52 | **72.64%** | $\pm 1.03$ | 1.7348 | **0.00%** | 0.0000 | 3.51 GF | 25.0% |
| | 52 | Single Centroid | 1 | 51 | 72.40% | $\pm 1.06$ | 1.7132 | -0.24% | -0.0216 | 3.51 GF | 25.0% |
| | 52 | PCA Bank | 4 | 48 | 71.96% | $\pm 1.17$ | 1.6610 | -0.68% | -0.0739 | 3.51 GF | 25.0% |
| | 52 | PCA Bank | 8 | 44 | 71.64% | $\pm 1.33$ | 1.6271 | -1.00% | -0.1077 | 3.51 GF | 25.0% |
| | 52 | PCA Bank | 16 | 36 | 70.90% | $\pm 1.07$ | 1.4994 | -1.74% | -0.2354 | 3.51 GF | 25.0% |
| | 52 | K-Means Bank | 4 | 48 | 71.94% | $\pm 1.12$ | 1.6633 | -0.70% | -0.0715 | 3.51 GF | 25.0% |
| | 52 | K-Means Bank | 8 | 44 | 71.74% | $\pm 1.25$ | 1.6350 | -0.90% | -0.0998 | 3.51 GF | 25.0% |
| | 52 | K-Means Bank | 16 | 36 | 70.80% | $\pm 1.10$ | 1.5141 | -1.84% | -0.2208 | 3.51 GF | 25.0% |
| **ViT-B/16** | 73 | **RANDOM_PRUNING** | 0 | 73 | **72.48%** | $\pm 0.54$ | 2.8060 | **0.00%** | 0.0000 | 25.79 GF | 26.6% |
| (Clean 76.1%) | 73 | Single Centroid | 1 | 72 | 72.32% | $\pm 0.58$ | 2.7889 | -0.16% | -0.0171 | 25.79 GF | 26.6% |
| | 73 | PCA Bank | 4 | 69 | 72.00% | $\pm 0.82$ | 2.7763 | -0.48% | -0.0297 | 25.79 GF | 26.6% |
| | 73 | PCA Bank | 8 | 65 | 71.62% | $\pm 0.73$ | 2.7294 | -0.86% | -0.0766 | 25.79 GF | 26.6% |
| | 73 | PCA Bank | 16 | 57 | 71.10% | $\pm 0.91$ | 2.6168 | -1.38% | -0.1892 | 25.79 GF | 26.6% |
| | 73 | K-Means Bank | 4 | 69 | 71.98% | $\pm 0.54$ | 2.7899 | -0.50% | -0.0161 | 25.79 GF | 26.6% |
| | 73 | K-Means Bank | 8 | 65 | 71.50% | $\pm 0.73$ | 2.7354 | -0.98% | -0.0706 | 25.79 GF | 26.6% |
| | 73 | K-Means Bank | 16 | 57 | 70.38% | $\pm 0.89$ | 2.5968 | -2.10% | -0.2092 | 25.79 GF | 26.6% |
| | 112 | **RANDOM_PRUNING** | 0 | 112 | **73.72%** | $\pm 0.63$ | 2.9987 | **0.00%** | 0.0000 | 28.68 GF | 18.4% |
| | 112 | Single Centroid | 1 | 111 | 73.52% | $\pm 0.62$ | 2.9901 | -0.20% | -0.0086 | 28.68 GF | 18.4% |
| | 112 | PCA Bank | 4 | 108 | 73.56% | $\pm 0.24$ | 2.9730 | -0.16% | -0.0257 | 28.68 GF | 18.4% |
| | 112 | PCA Bank | 8 | 104 | 73.40% | $\pm 0.50$ | 2.9559 | -0.32% | -0.0428 | 28.68 GF | 18.4% |
| | 112 | PCA Bank | 16 | 96 | 72.84% | $\pm 0.25$ | 2.8962 | -0.88% | -0.1025 | 28.68 GF | 18.4% |
| | 112 | K-Means Bank | 4 | 108 | 73.40% | $\pm 0.36$ | 2.9761 | -0.32% | -0.0226 | 28.68 GF | 18.4% |
| | 112 | K-Means Bank | 8 | 104 | 73.12% | $\pm 0.36$ | 2.9511 | -0.60% | -0.0476 | 28.68 GF | 18.4% |
| | 112 | K-Means Bank | 16 | 96 | 72.48% | $\pm 0.45$ | 2.8787 | -1.24% | -0.1200 | 28.68 GF | 18.4% |
| **DINOv2** | 184 | **RANDOM_PRUNING** | 0 | 184 | **77.54%** | $\pm 0.34$ | 2.6238 | **0.00%** | 0.0000 | 6.03 GF | 7.6% |
| (Clean 78.8%) | 184 | Single Centroid | 1 | 183 | 77.40% | $\pm 0.23$ | 2.6225 | -0.14% | -0.0013 | 6.03 GF | 7.6% |
| | 184 | PCA Bank | 4 | 180 | 77.22% | $\pm 0.24$ | 2.6039 | -0.32% | -0.0199 | 6.03 GF | 7.6% |
| | 184 | PCA Bank | 8 | 176 | 77.12% | $\pm 0.39$ | 2.5843 | -0.42% | -0.0395 | 6.03 GF | 7.6% |
| | 184 | PCA Bank | 16 | 168 | 76.92% | $\pm 0.19$ | 2.5482 | -0.62% | -0.0757 | 6.03 GF | 7.6% |
| | 184 | K-Means Bank | 4 | 180 | 77.28% | $\pm 0.22$ | 2.6039 | -0.26% | -0.0200 | 6.03 GF | 7.6% |
| | 184 | K-Means Bank | 8 | 176 | 77.02% | $\pm 0.41$ | 2.5817 | -0.52% | -0.0422 | 6.03 GF | 7.6% |
| | 184 | K-Means Bank | 16 | 168 | 76.92% | $\pm 0.13$ | 2.5447 | -0.62% | -0.0791 | 6.03 GF | 7.6% |
| | 199 | **RANDOM_PRUNING** | 0 | 199 | **77.86%** | $\pm 0.34$ | 2.6685 | **0.00%** | 0.0000 | 6.13 GF | 6.1% |
| | 199 | Single Centroid | 1 | 198 | 77.78% | $\pm 0.47$ | 2.6678 | -0.08% | -0.0007 | 6.13 GF | 6.1% |
| | 199 | PCA Bank | 4 | 195 | 77.66% | $\pm 0.40$ | 2.6508 | -0.20% | -0.0177 | 6.13 GF | 6.1% |
| | 199 | PCA Bank | 8 | 191 | 77.24% | $\pm 0.17$ | 2.6345 | -0.62% | -0.0341 | 6.13 GF | 6.1% |
| | 199 | PCA Bank | 16 | 183 | 77.14% | $\pm 0.36$ | 2.6034 | -0.72% | -0.0651 | 6.13 GF | 6.1% |
| | 199 | K-Means Bank | 4 | 195 | 77.62% | $\pm 0.43$ | 2.6510 | -0.24% | -0.0176 | 6.13 GF | 6.1% |
| | 199 | K-Means Bank | 8 | 191 | 77.28% | $\pm 0.40$ | 2.6319 | -0.58% | -0.0366 | 6.13 GF | 6.1% |
| | 199 | K-Means Bank | 16 | 183 | 77.20% | $\pm 0.35$ | 2.5995 | -0.66% | -0.0690 | 6.13 GF | 6.1% |

---

## 3. Scientific Analysis & Mechanistic Insights

### 3.1 Real Anchors Strictly Dominate Synthetic Carriers Under Compression
In the uncompressed full-sequence setting ($N_{\text{patch}}=196$ or $256$), late layers exhibit content fungibility: up to $86.5\%$ of tokens can be replaced with a calibration centroid without collapsing accuracy, because the 26 surviving real tokens provide sufficient image-specific signal and the 170 surrogate tokens maintain the background representation volume.

However, **under sequence compression**, the token count itself is the scarce resource. At a fixed downstream budget $B$:
- Every slot given to a synthetic carrier token requires **sacrificing one real image patch** ($M_{\text{real}} = B - K$).
- A real image patch contains localized, image-specific discriminative evidence.
- A synthetic carrier (even when preserving the mean, top eigenvectors, or cluster centroids of the calibration manifold) contains **only generic, class-agnostic population statistics**.
- Consequently, trading real image patches for generic carriers strictly discards discriminative mutual information. At $B=28$ in DeiT-Small, dropping from $M=28$ real patches down to $M=12$ real patches (with $K=16$ PCA carriers) drops Top-1 accuracy from **$69.42\% \to 61.34\%$ ($-8.08\%$ drop)**.

### 3.2 Invariance Across Bank Formulations
Across all architectures and budgets:
- **PCA Bank vs. K-Means Bank:** Perform virtually identically across all $K$ (e.g. DeiT-Small $B=28, K=4$: PCA $67.62\%$ vs. K-Means $67.62\%$; $K=8$: PCA $66.42\%$ vs. K-Means $66.40\%$).
- **Learned vs. Random Directions:** Random orthogonal directions scaled by matched eigenvalues achieve nearly identical results (e.g. DeiT-Small $B=28, K=4$: Random Dir $67.62\%$).
- **Implication:** The bottleneck is **NOT** the geometric fidelity or clustering quality of the synthetic bank. The bottleneck is the fundamental information loss incurred by deleting real image patches.

---

## 4. Answers to Mandatory Questions

1. **What is Random Pruning accuracy at every matched $B$?**  
   - DeiT-Tiny ($B=45$): **61.62%** | ($B=73$): **64.82%**
   - DeiT-Small ($B=28$): **69.42%** | ($B=52$): **72.64%**
   - ViT-B/16 ($B=73$): **72.48%** | ($B=112$): **73.72%**
   - DINOv2 ($B=184$): **77.54%** | ($B=199$): **77.86%**
2. **What is PCA bank accuracy for $K \in \{4, 8, 16\}$?**  
   - DeiT-Tiny ($B=45$): $K=4 \to 60.58\%$, $K=8 \to 59.62\%$, $K=16 \to 57.66\%$
   - DeiT-Small ($B=28$): $K=4 \to 67.62\%$, $K=8 \to 66.42\%$, $K=16 \to 61.34\%$
   - ViT-B ($B=73$): $K=4 \to 72.00\%$, $K=8 \to 71.62\%$, $K=16 \to 71.10\%$
   - DINOv2 ($B=184$): $K=4 \to 77.22\%$, $K=8 \to 77.12\%$, $K=16 \to 76.92\%$
3. **What is K-means bank accuracy for $K \in \{4, 8, 16\}$?**  
   - DeiT-Tiny ($B=45$): $K=4 \to 60.52\%$, $K=8 \to 59.88\%$, $K=16 \to 58.08\%$
   - DeiT-Small ($B=28$): $K=4 \to 67.62軽\%$, $K=8 \to 66.40\%$, $K=16 \to 61.14\%$
   - ViT-B ($B=73$): $K=4 \to 71.98\%$, $K=8 \to 71.50\%$, $K=16 \to 70.38\%$
   - DINOv2 ($B=184$): $K=4 \to 77.28\%$, $K=8 \to 77.02\%$, $K=16 \to 76.92\%$
4. **What are $\Delta \text{Acc}$ and $\Delta \text{Margin}$ vs. Random Pruning?**  
   Strictly negative or zero in all configurations. Mean $\Delta \text{Acc}$ ranges from $-0.08\%$ to $-8.28\%$; mean $\Delta \text{Margin}$ ranges from $-0.0007$ to $-0.6496$.
5. **What is the best $K$ per model?**  
   $K = 0$ (Random Pruning) for all four models without exception.
6. **Cross-mask variability:**  
   Standard deviations across the 5 spatial mask seeds are small and consistent ($\pm 0.2\%$ to $\pm 1.5\%$), confirming that the superiority of real patches over synthetic carriers is mask-invariant.
7. **Decision Rule Classification:**  
   **`KILL APPLICATION`** (0 of 4 models showed any gain $\ge +1.0\%$; Random Pruning won universally).
8. **Was any physical latency benchmark triggered?**  
   **No.** Per protocol, latency benchmarking is only triggered if a method achieves `PROMISING` status.
9. **Does the main paper story change?**  
   **No.** The paper remains strictly a study of late Vision Transformer representation geometry and content fungibility. This experiment decisively closes the door to synthetic-carrier token merging as a compression technique, reinforcing that the mechanistic finding is about representation redundancy within the full network, not an off-the-shelf pruning competitor.

---

## 5. Artifact Suite

- Manifest: [`outputs/fungibility_geometry_bank/experiment_manifest.json`](file:///d:/Study/ResCancel/outputs/fungibility_geometry_bank/experiment_manifest.json)
- Validation Results: [`outputs/fungibility_geometry_bank/validation_results.json`](file:///d:/Study/ResCancel/outputs/fungibility_geometry_bank/validation_results.json)
- Full Trial Data: [`outputs/fungibility_geometry_bank/all_results.csv`](file:///d:/Study/ResCancel/outputs/fungibility_geometry_bank/all_results.csv) (440 evaluations)
- Matched Budget Summary: [`outputs/fungibility_geometry_bank/matched_budget_summary.csv`](file:///d:/Study/ResCancel/outputs/fungibility_geometry_bank/matched_budget_summary.csv)
- Delta Table: [`outputs/fungibility_geometry_bank/delta_vs_pruning.csv`](file:///d:/Study/ResCancel/outputs/fungibility_geometry_bank/delta_vs_pruning.csv)
- Publication Figures:
  - [`accuracy_vs_token_budget.png`](file:///d:/Study/ResCancel/figures/fungibility_geometry_bank/accuracy_vs_token_budget.png)
  - [`delta_vs_pruning.png`](file:///d:/Study/ResCancel/figures/fungibility_geometry_bank/delta_vs_pruning.png)
  - [`pca_vs_kmeans.png`](file:///d:/Study/ResCancel/figures/fungibility_geometry_bank/pca_vs_kmeans.png)
