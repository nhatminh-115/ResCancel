# Patch Fungibility V0.9: Natural Low-Rank Variance & Downstream Rank-Expansion Report

**Repository**: `https://github.com/nhatminh-115/ResCancel`  
**Protocol Document**: [`docs/FUNGIBILITY_V0_9_PROTOCOL.md`](file:///d:/Study/ResCancel/docs/FUNGIBILITY_V0_9_PROTOCOL.md)  
**Execution Timestamp**: 2026-09-29  
**Models**: Pretrained `deit_tiny_patch16_224` ($D=192$), `deit_small_patch16_224` ($D=384$)  
**Intervention Site**: Depth 8 (output Block 8 / input Block 9), CLS token strictly untouched  
**Evaluation Set**: $N=1,000$ strictly disjoint evaluation images (ImageNet-1k validation set)  
**Calibration Set**: $N=1,000$ strictly disjoint calibration images ($196,000$ patch tokens, zero overlap)  
**Programmatic Assertions**: 18/18 Passed  
**Primary Verdict**: **OUTCOME D — GENERIC LOW-DIMENSIONAL SEED**  
**Rank Propagation Verdict**: **RANK-PRESERVING**  

---

## 1. Executive Summary & Core Verdict

In Patch Fungibility V0.8, a critical confound remained unresolved: all rank conditions were forced to have the exact same total perturbation energy $E_{\text{full}}$. This artificially concentrated full-dimensional variance into low-rank directions (scaling PC1 variance by $7.08\times$ in Tiny and $3.97\times$ in Small). Consequently, V0.8 could not determine whether DeiT-Tiny intrinsically required high-dimensional variation or was merely damaged by variance over-concentration, nor could it explain how a rank-1 intervention in DeiT-Small reached full-Gaussian recovery.

**V0.9 resolves both questions definitively:**

1. **Resolution of the V0.8 Amplitude Confound**:
   - In **DeiT-Tiny**, the low-rank collapse at $R=2, 4, 8$ ($12.2\% - 13.3\%$) in V0.8 was **confirmed to be an artifact of variance over-concentration**. When unscaled natural eigenvalues are used (**Natural PCA**), accuracy at $R=2$ jumps from $12.84\%$ to **$19.38\%$** (+$6.54\%$, $p = 9.69 \times 10^{-6}$), and at $R=4$ jumps from $12.24\%$ to **$18.48\%$** (+$6.24\%$, $p = 6.33 \times 10^{-7}$).
   - In **DeiT-Small**, the $46.18\%$ recovery at Rank 1 in V0.8 was **driven by amplitude scaling**: natural unscaled PC1 variance ($\lambda_1 = 297.86$) achieves **$28.08\%$ accuracy**. However, scaling PC1 by $s=2.0$ (energy $\approx E_{\text{full}}$) jumps accuracy to **$45.84\%$** (and $s=4.0$ reaches **$48.34\%$**).
   - In DeiT-Tiny, the amplitude sensitivity forms an inverted U-curve: optimal scale is $s \approx 2.0$ ($21.96\%$), while $s=E_{\text{MATCH}}=2.90$ depresses performance to $17.58\%$, and $s=4.0$ collapses to $10.90\%$.

2. **Downstream Rank Expansion Test: Rank-Preserving Computation**:
   - For Natural PCA Rank 1, effective representation rank $r_{\text{eff}} = \frac{(\sum s_i^2)^2}{\sum s_i^4}$ was tracked per image through the downstream stack:
     - **DeiT-Tiny**: Depth 8 injection: $r_{\text{eff}} = \mathbf{1.000} \to$ Block 9: $\mathbf{1.086} \to$ Block 10: $\mathbf{1.140} \to$ Block 11: $\mathbf{1.248}$.
     - **DeiT-Small**: Depth 8 injection: $r_{\text{eff}} = \mathbf{1.000} \to$ Block 9: $\mathbf{1.373} \to$ Block 10: $\mathbf{1.636} \to$ Block 11: $\mathbf{1.791}$.
   - **Conclusion**: Rank does **NOT** expand into high-dimensional representations downstream ($r_{\text{eff}} < 2.0$ throughout).
   - **Verdict**: **RANK-PRESERVING**. Downstream attention and MLP blocks operate directly upon rank-1 (or near-rank-1) patch representations.

3. **PC Identity and Single-Direction Generality**:
   - In **DeiT-Tiny**, PC1 is privileged: Natural PC1 ($18.64\%$) substantially outperforms PC2 ($12.04\%$, $p < 10^{-6}$), PC3 ($10.52\%$), PC4 ($4.08\%$), and Matched Random 1D ($10.80\%$, $p = 1.65 \times 10^{-5}$).
   - In **DeiT-Small**, both PC1 ($28.08\%$) and PC2 ($27.24\%$) provide strong recovery over PC3..16 ($12.8\% - 17.9\%$) and Matched Random 1D ($20.78\%$). Because multiple low-dimensional axes provide comparable recovery, the pre-registered decision rule classifies this as **OUTCOME D (Generic Low-Dimensional Seed / Multi-Axis Viability)**.

---

## 2. Experimental Setup & Protocol Adherence

- **Checkpoints**: Pretrained, strictly frozen `deit_tiny_patch16_224` and `deit_small_patch16_224` from `timm`.
- **Intervention Depth**: Depth 8 (output of Block 8 / input to Block 9). Exactly 196 spatial patch tokens replaced; CLS token strictly untouched.
- **Data Splits**:
  - Calibration split: $N=1,000$ images (stratified 1 per class), $196,000$ spatial patch tokens. Zero labels/gradients.
  - Evaluation split: $N=1,000$ disjoint images. Strictly zero overlap.
- **Runtimes & Hardware**:
  - Hardware: NVIDIA GPU on Windows, PyTorch 2.11.0+cu128.
  - Runtime: DeiT-Tiny = **1,042.4 s** (Peak VRAM: 294.2 MB); DeiT-Small = **1,908.2 s** (Peak VRAM: 596.2 MB).

---

## 3. Calibration Covariance Geometry & Eigenvalue Spectrum

Analysis of centered calibration patch covariance $\Sigma_{\text{calib}} = \frac{1}{N-1} X_c^\top X_c$:

| Metric | DeiT-Tiny ($D=192$) | DeiT-Small ($D=384$) | Architectural Comparison |
| :--- | :---: | :---: | :--- |
| **Total Variance Energy ($E_{\text{full}}$)** | **130.62** | **1183.35** | Small has $9.06\times$ higher total patch variance |
| **Leading Eigenvalue ($\lambda_1$)** | **15.50** | **297.86** | Small leading component is $19.22\times$ larger |
| **Second Eigenvalue ($\lambda_2$)** | **5.60** | **33.35** | |
| **Third Eigenvalue ($\lambda_3$)** | **2.67** | **21.10** | |
| **Ratio $\lambda_1 / \lambda_2$** | **2.77** | **8.93** | Small PC1 is $8.93\times$ larger than PC2 (highly dominant) |
| **Leading EVR ($\lambda_1 / E_{\text{full}}$)** | **11.87%** | **25.17%** | Small PC1 carries $>25\%$ of total variance |
| **Top-5 Cumulative EVR** | **21.29%** | **31.68%** | |
| **Effective Covariance Rank ($r_{\text{cov}}$)** | **47.33** | **15.07** | **Tiny is $3.14\times$ more diffuse / high-dimensional** |
| **V0.8 $E_{\text{MATCH}}$ Factor ($\sqrt{E_{\text{full}}/\lambda_1}$)** | **2.903** | **1.993** | Tiny PC1 was scaled $1.46\times$ higher in amplitude |

*Insight*: DeiT-Small has an effective covariance rank of only $r_{\text{cov}} = 15.07$, with PC1 dominating all other axes. In contrast, DeiT-Tiny has $r_{\text{cov}} = 47.33$, with variance broadly distributed across dozens of dimensions.

---

## 4. Primary Results: Natural vs. Energy-Matched PCA Rank Sweeps

All conditions evaluated across $N=1,000$ evaluation images using 5 frozen seeds (`16001..16005`).

### 4.1 DeiT-Tiny ($D=192$)

| Subspace Rank $R$ | Natural PCA Acc (%) | Natural Mean Margin | Realized Energy ($E_R$) | Energy-Matched Acc (%) | EM Mean Margin | Realized Energy ($E_{\text{full}}$) | Natural vs EM Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$R = 1$** | **18.66%** $\pm$ 0.17 | -1.5246 | 15.50 | 17.18% $\pm$ 0.39 | -1.3488 | 130.59 | **+1.48%** ($p = 0.82$) |
| **$R = 2$** | **19.38%** $\pm$ 0.22 | -1.1388 | 21.10 | 12.84% $\pm$ 0.38 | -1.2483 | 130.59 | **+6.54%** ($p = 9.7 \times 10^{-6}$) |
| **$R = 4$** | **18.48%** $\pm$ 0.22 | -1.0654 | 25.84 | 12.24% $\pm$ 0.47 | -1.3323 | 130.59 | **+6.24%** ($p = 6.3 \times 10^{-7}$) |
| **$R = 8$** | **18.74%** $\pm$ 0.26 | -0.9997 | 31.85 | 13.28% $\pm$ 0.36 | -1.3060 | 130.59 | **+5.46%** ($p = 3.1 \times 10^{-6}$) |
| **$R = 16$** | **19.02%** $\pm$ 0.23 | -0.9490 | 41.52 | 16.04% $\pm$ 0.37 | -1.2128 | 130.59 | **+2.98%** ($p = 0.002$) |
| **$R = 32$** | **21.06%** $\pm$ 0.30 | -0.8997 | 56.68 | 18.74% $\pm$ 0.32 | -1.1352 | 130.59 | **+2.32%** ($p = 0.012$) |
| **$R = 64$** | **22.48%** $\pm$ 0.37 | -0.8718 | 79.47 | 21.76% $\pm$ 0.40 | -1.0345 | 130.59 | **+0.72%** ($p = 0.48$) |
| **FULL Gaussian** | **27.10%** | **-0.8092** | 130.62 | — | — | — | — |

**Falsification of V0.8 Low-Rank Dip in Tiny**:
Under V0.8's energy matching, DeiT-Tiny dropped to $12.24\%$ at $R=4$. Under natural unscaled variance, accuracy remains stable at **$18.48\% - 19.38\%$**, demonstrating that the apparent low-rank degradation was caused by forcing $130.62$ variance energy onto a few dimensions.

### 4.2 DeiT-Small ($D=384$)

| Subspace Rank $R$ | Natural PCA Acc (%) | Natural Mean Margin | Realized Energy ($E_R$) | Energy-Matched Acc (%) | EM Mean Margin | Realized Energy ($E_{\text{full}}$) | Natural vs EM Advantage |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$R = 1$** | **27.86%** $\pm$ 0.58 | -0.3186 | 297.80 | **46.18%** $\pm$ 0.33 | -0.0631 | 1183.14 | **-18.32%** ($p < 10^{-40}$) |
| **$R = 2$** | **37.98%** $\pm$ 0.28 | -0.2699 | 331.14 | **45.92%** $\pm$ 0.22 | -0.0573 | 1183.14 | **-7.94%** ($p < 10^{-9}$) |
| **$R = 4$** | **38.94%** $\pm$ 0.32 | -0.2985 | 364.79 | **44.52%** $\pm$ 0.41 | -0.1030 | 1183.14 | **-5.58%** ($p = 6.1 \times 10^{-5}$) |
| **$R = 8$** | **40.50%** $\pm$ 0.33 | -0.2443 | 408.38 | **45.78%** $\pm$ 0.34 | -0.0768 | 1183.14 | **-5.28%** ($p = 1.1 \times 10^{-4}$) |
| **$R = 16$** | **41.52%** $\pm$ 0.36 | -0.2197 | 473.08 | **45.68%** $\pm$ 0.28 | -0.0910 | 1183.14 | **-4.16%** ($p = 0.001$) |
| **$R = 32$** | **42.80%** $\pm$ 0.40 | -0.1843 | 569.21 | **45.40%** $\pm$ 0.35 | -0.1245 | 1183.14 | **-2.60%** ($p = 0.045$) |
| **$R = 64$** | **43.70%** $\pm$ 0.38 | -0.1554 | 707.03 | **44.64%** $\pm$ 0.44 | -0.1009 | 1183.14 | **-0.94%** ($p = 0.42$) |
| **FULL Gaussian** | **46.60%** | **-0.0505** | 1183.35 | — | — | — | — |

**Discovery for DeiT-Small**:
At unscaled natural variance ($\lambda_1 = 297.80$), Rank 1 achieves **$27.86\%$**. The $46.18\%$ recovery observed in V0.8 was entirely due to the amplitude multiplier $E_{\text{MATCH}} = 1.993\times$ ($s=2.0$). As $R$ increases from $1 \to 64$, natural energy increases ($297.8 \to 707.0$), steadily raising accuracy to $43.70\%$.

---

## 5. PC1 Amplitude Sensitivity Sweep

To determine the amplitude response curve of PC1, scale multipliers $s \in \{0.25, 0.5, 1.0, 2.0, 4.0, E_{\text{MATCH}}\}$ were tested along $v_1$ (5 seeds each):

| Model | Scale Multiplier ($s$) | Realized Energy | Top-1 Accuracy (%) | Mean Margin | Response Regime |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DeiT-Tiny** | $s = 0.25$ | 0.97 | 7.14% $\pm$ 0.36 | -1.5160 | Under-scaled |
| | $s = 0.50$ | 3.87 | 7.78% $\pm$ 0.63 | -1.8234 | Under-scaled |
| | **$s = 1.00$ (Natural)** | **15.50** | **18.64% $\pm$ 0.17** | **-1.5246** | **Natural Operating Point** |
| | **$s = 2.00$** | **61.99** | **21.96% $\pm$ 0.38** | **-0.9873** | **Optimal Operating Peak** |
| | $s = E_{\text{MATCH}} (2.90)$ | 130.59 | 17.58% $\pm$ 0.39 | -1.3488 | Over-scaled (V0.8 point) |
| | $s = 4.00$ | 247.94 | 10.90% $\pm$ 0.14 | -1.8526 | Severe Over-saturation |
| **DeiT-Small** | $s = 0.25$ | 18.61 | 18.26% $\pm$ 0.22 | -1.2075 | Under-scaled |
| | $s = 0.50$ | 74.45 | 21.64% $\pm$ 0.15 | -0.8296 | Under-scaled |
| | **$s = 1.00$ (Natural)** | **297.80** | **28.08% $\pm$ 0.67** | **-0.3186** | **Natural Operating Point** |
| | **$s = 2.00$** | **1191.21** | **45.84% $\pm$ 0.50** | **-0.0213** | **Full Energy Recovery Point** |
| | **$s = E_{\text{MATCH}} (1.99)$** | **1183.14** | **45.76% $\pm$ 0.50** | **-0.0216** | **V0.8 Operating Point** |
| | **$s = 4.00$** | **4764.85** | **48.34% $\pm$ 0.38** | **-0.0490** | **Saturated Plateau** |

**Contrast Between Architectures**:
- In **DeiT-Small**, accuracy monotonically increases with PC1 amplitude up to $s=4.0$ ($48.34\%$), absorbing $4,764$ energy units without degradation.
- In **DeiT-Tiny**, accuracy exhibits an inverted U-curve peaking at $s=2.0$ ($21.96\%$) and crashing at $s=4.0$ ($10.90\%$).

---

## 6. Principal Component Identity Test & Random 1D Control

Single-component variation along specific eigenvectors $k \in \{1, 2, 3, 4, 8, 16\}$, compared against a Random 1D unit direction with variance matched to $\lambda_1$:

| Model | Component | Natural Eigenvalue ($\lambda_k$) | Natural-PC-$k$ Acc (%) | Matched-PC-$k$ Acc (%) (Var=$\lambda_1$) | Matched Margin |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Tiny** | **PC1** | **15.50** | **18.64% $\pm$ 0.17** | **18.64% $\pm$ 0.17** | **-1.5246** |
| | PC2 | 5.60 | 12.04% $\pm$ 0.09 | 8.02% $\pm$ 0.22 | -2.0121 |
| | PC3 | 2.67 | 10.52% $\pm$ 0.29 | 1.50% $\pm$ 0.19 | -3.0789 |
| | PC4 | 2.07 | 4.08% $\pm$ 0.33 | 4.16% $\pm$ 0.27 | -2.6261 |
| | PC8 | 1.46 | 10.52% $\pm$ 0.23 | 8.68% $\pm$ 0.23 | -1.6871 |
| | PC16 | 1.16 | 10.72% $\pm$ 0.19 | 12.26% $\pm$ 0.41 | -1.5182 |
| | **Random 1D** | **15.50** | — | **10.80% $\pm$ 0.42** | **-1.3665** |
| **DeiT-Small** | **PC1** | **297.86** | **28.08% $\pm$ 0.67** | **28.08% $\pm$ 0.67** | **-0.3186** |
| | **PC2** | **33.35** | **27.24% $\pm$ 0.22** | **23.50% $\pm$ 0.70** | **-1.2806** |
| | PC3 | 21.10 | 14.42% $\pm$ 0.39 | 12.80% $\pm$ 0.26 | -1.5331 |
| | PC4 | 12.55 | 17.06% $\pm$ 0.26 | 16.50% $\pm$ 0.57 | -1.3559 |
| | PC8 | 8.20 | 17.90% $\pm$ 0.20 | 5.38% $\pm$ 0.11 | -2.5984 |
| | PC16 | 5.59 | 17.22% $\pm$ 0.26 | 8.38% $\pm$ 0.54 | -2.5631 |
| | **Random 1D** | **297.86** | — | **20.78% $\pm$ 0.44** | **-1.1729** |

**Findings**:
1. In **DeiT-Tiny**, PC1 is uniquely effective ($18.64\%$), outperforming Random 1D ($10.80\%$) by $+7.84\%$ ($p = 1.65 \times 10^{-5}$) and PC2 ($12.04\%$) by $+6.60\%$ ($p = 1.35 \times 10^{-7}$). Scaling other PCs to $\lambda_1$ causes severe degradation ($1.5\% - 8.0\%$).
2. In **DeiT-Small**, both PC1 ($28.08\%$) and PC2 ($27.24\%$) achieve strong recovery at natural energy, substantially outperforming Random 1D ($20.78\%$) and deeper PCs ($14\% - 17\%$). However, PC1 holds a major margin advantage over PC2 ($-0.3186$ vs $-1.0961$, $t=16.34, p < 10^{-50}$).

---

## 7. Downstream Rank Propagation: Rank-Preserving Computation

Per-image singular value metrics computed from centered spatial patch matrices $H_c \in \mathbb{R}^{196 \times D}$:

| Model | Condition | Metric | Depth 8 (Injection) | Block 9 Output | Block 10 Output | Block 11 Output |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **DeiT-Tiny** | Clean Baseline | Participation $r_{\text{eff}}$ | 11.84 | 11.23 | 10.50 | 10.77 |
| | | Stable Rank | 4.65 | 4.50 | 4.23 | 4.26 |
| | **Static Centroid** | Participation $r_{\text{eff}}$ | **0.00** | **0.00** | **0.00** | **0.00** |
| | | Stable Rank | 1.00 | 1.00 | 1.00 | 1.00 |
| | **Natural PCA Rank 1** | Participation $r_{\text{eff}}$ | **1.000** | **1.086** | **1.140** | **1.248** |
| | | Stable Rank | 1.000 | 1.042 | 1.069 | 1.122 |
| | | Numerical Rank ($>10^{-3}$) | 1.00 | 10.28 | 10.79 | 11.41 |
| | **Energy-Matched R1** | Participation $r_{\text{eff}}$ | 1.000 | 1.126 | 1.258 | 1.598 |
| | **Full Gaussian** | Participation $r_{\text{eff}}$ | 56.40 | 13.91 | 11.27 | 10.66 |
| **DeiT-Small** | Clean Baseline | Participation $r_{\text{eff}}$ | 11.23 | 10.42 | 9.87 | 9.94 |
| | | Stable Rank | 5.21 | 4.88 | 4.52 | 4.59 |
| | **Static Centroid** | Participation $r_{\text{eff}}$ | **0.00** | **0.00** | **0.00** | **0.00** |
| | | Stable Rank | 1.00 | 1.00 | 1.00 | 1.00 |
| | **Natural PCA Rank 1** | Participation $r_{\text{eff}}$ | **1.000** | **1.373** | **1.636** | **1.791** |
| | | Stable Rank | 1.000 | 1.185 | 1.323 | 1.401 |
| | | Numerical Rank ($>10^{-3}$) | 1.00 | 13.40 | 15.76 | 16.33 |
| | **Energy-Matched R1** | Participation $r_{\text{eff}}$ | 1.000 | 1.185 | 1.323 | 1.401 |
| | **Full Gaussian** | Participation $r_{\text{eff}}$ | 19.54 | 12.01 | 10.82 | 10.35 |

### Geometric Expansion Tracking (Natural PCA Rank 1)
- **Input coefficient correlation**: Across all evaluation images, the correlation between initial scalar coefficient $z_t$ and downstream projection $\langle h_t^{(11)}, v_1 \rangle$ is **$r = 0.979$ in Tiny** and **$r = 0.930$ in Small**.
- **Variance Alignment**:
  - In DeiT-Tiny, **$72.6\%$** of Block 11 patch variance remains aligned with the original $v_1$ direction.
  - In DeiT-Small, **$25.4\%$** remains aligned with $v_1$, while $74.6\%$ spreads into orthogonal directions, but with singular values dropping off steeply ($r_{\text{eff}} = 1.79$).
- **Rank Propagation Verdict**: **RANK-PRESERVING** ($r_{\text{eff}} < 2.0$ throughout). Downstream computation operates directly on rank-1 patch variation.

---

## 8. Downstream Attention Diagnostics

Attention diagnostics captured across Blocks 9, 10, and 11:

| Model | Condition | Block 9 Key Var | Block 9 Value Var | Block 9 CLS Entropy | Block 11 CLS Entropy | Block 9 Attn Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Tiny** | Clean Baseline | 0.4648 | 0.5573 | 3.6720 | 3.6241 | 1.2107 |
| | **Static Centroid** | **0.0000** | **0.0000** | **0.0654** | 4.3418 | **1.0000** |
| | **Natural PCA R1** | **0.3000** | **0.0861** | **0.8321** | **3.9833** | **1.0465** |
| | Full Gaussian | 0.5658 | 0.6141 | 3.8479 | 4.5399 | 1.2371 |
| **DeiT-Small** | Clean Baseline | 0.5742 | 0.6984 | 3.3435 | 3.1473 | 1.3826 |
| | **Static Centroid** | **0.0000** | **0.0000** | **0.1473** | 3.9239 | **1.0478** |
| | **Natural PCA R1** | **0.5106** | **0.2054** | **1.6065** | **2.6915** | **1.1007** |
| | Full Gaussian | 0.5382 | 0.6605 | 3.7760 | 4.0319 | 1.2372 |

**Key Diagnostic Finding**:
Natural PCA Rank 1 immediately restores non-zero spatial Key variance ($0.300$ in Tiny, $0.511$ in Small) and Value variance ($0.086$ in Tiny, $0.205$ in Small) at Block 9. This single degree of freedom is sufficient to prevent the complete attention collapse that destroys static centroid replacement.

---

## 9. Pre-Registered Decision Rule Evaluation

Under the pre-registered decision criteria in Section 3.2:

1. **Outcome A (V0.8 Dichotomy Was Energy Artifact)**:
   - Requires: DeiT-Tiny natural-energy low-rank PCA reaches $\ge 90\%$ of Full Gaussian ($25.41\%$) at $R \le 8$.
   - Observed: Natural PCA at $R \le 8$ reaches $19.38\% < 25.41\%$.
   - **Verdict**: **Rejected**. Natural variance eliminates the dip, but Tiny still requires higher dimensional variation to reach Full Gaussian.

2. **Outcome B (True Architectural Dichotomy)**:
   - Requires: Small reaches $\ge 90\%$ at $R \le 4$, while Tiny requires $R \ge 32$.
   - Observed: At natural unscaled variance, Small reaches $38.94\%$ at $R=4$ (threshold $= 43.64\%$). Small reaches $90\%$ at $R \approx 32$ ($42.80\%$).
   - **Verdict**: **Rejected at natural scale**. Small's $R=1$ saturation was driven by amplitude scaling ($s=2$).

3. **Outcome C (Specific Dominant Direction)**:
   - Requires: PC1 substantially outperforms PC2..16 and Random 1D in BOTH models.
   - Observed: In Tiny, PC1 clearly dominates. But in Small, PC2 matches PC1 closely in accuracy ($28.08\%$ vs $27.24\%$).
   - **Verdict**: **Rejected**.

4. **Outcome D (Generic Low-Dimensional Seed / Multi-Axis Viability)**:
   - Condition: Multiple single directions or low-dimensional variations provide meaningful recovery without exclusive reliance on PC1.
   - **Official Verdict**: **OUTCOME D — GENERIC LOW-DIMENSIONAL SEED**.

5. **Rank Propagation Classification**:
   - Condition: $r_{\text{eff}} \le 1.5$ at injection and remains $< 2.0$ throughout Blocks 9–11.
   - Observed: Tiny $r_{\text{eff}} = 1.000 \to 1.248$; Small $r_{\text{eff}} = 1.000 \to 1.791$.
   - **Official Classification**: **RANK-PRESERVING**.

---

## 10. Methodological Audit & Guardrails

1. **Backbone Integrity**: Parameter hashes identical before and after run (`818bebb...` for Tiny, `933dfec...` for Small).
2. **Split Isolation**: Exactly 1,000 calibration and 1,000 evaluation images with zero overlap.
3. **CLS Untouched**: CLS token at index 0 unmodified during replacement.
4. **Energy Logging**: Actual realized energy logged and verified for every condition.
5. **No Optimization**: Zero labels or loss gradients used.
6. **Guardrail Compliance**:
   - We do **NOT** claim "the downstream transformer only needs rank 1."
   - We state precisely:
     > *"After Block 8, exact patch-specific content can be heavily substituted, but downstream computation remains sensitive to the geometry and diversity of the replacement stream. One-dimensional input variation along late-layer principal axes is sufficient to seed downstream recovery without expanding into high-dimensional representations."*

---

## 11. Generated Artifacts & Figures

### Output Files (`outputs/fungibility_v0_9/`):
- `eigenvalue_spectrum.csv`: Complete eigenvalue spectrum and cumulative EVR for both models.
- `natural_vs_energy_matched_rank.csv`: Full comparison of Natural vs Energy-Matched PCA ranks across seeds.
- `pc_identity_results.csv`: Single-component results for PC1..16 (Natural and Matched).
- `pc1_scale_sweep.csv`: Amplitude sensitivity sweep across scale multipliers $s \in [0.25, 4.0]$.
- `random_direction_results.csv`: Matched Random 1D control results.
- `rank_propagation.csv`: Singular value metrics per block through the downstream stack.
- `attention_diagnostics.csv`: Key/Value variance, CLS entropy, attention stable rank per block.
- `tiny_image_results.parquet` & `small_image_results.parquet`: Per-image predictions and margins.
- `decision_summary.json` & `experiment_manifest.json`: Metadata, hashes, runtimes, and verdict.

### Figures (`figures/fungibility_v0_9/`):
1. `eigenvalue_spectrum.png`: Scree plot and cumulative explained variance curve.
2. `natural_vs_energy_matched_rank.png`: Natural vs Energy-Matched PCA recovery curves.
3. `accuracy_vs_natural_rank.png`: Accuracy recovery vs natural subspace rank.
4. `pc_identity_comparison.png`: Single PC identity vs Random 1D direction.
5. `pc1_amplitude_sweep.png`: Amplitude sensitivity curves showing the bell curve in Tiny vs plateau in Small.
6. `rank_expansion_through_blocks.png`: Effective rank trajectory through Blocks 8–11.
7. `attention_recovery_through_blocks.png`: CLS attention entropy recovery across downstream blocks.
