# Confirmatory Benchmark Protocol: Operator-Aware Token Compression

**Status:** PRE-REGISTERED & FROZEN BEFORE EVALUATION  
**Date:** October 2026  
**Repository:** [nhatminh-115/Patch-Content-Fungibility](https://github.com/nhatminh-115/Patch-Content-Fungibility)  
**Related Protocol:** [FUNGIBILITY_V0_REPORT.md](file:///d:/Study/ResCancel/docs/FUNGIBILITY_V0_REPORT.md)  
**Implementation Modules:**  
- [operator_compression_confirmatory.py](file:///d:/Study/ResCancel/patch_fungibility/operator_compression_confirmatory.py)  
- [run_operator_compression_confirmatory.py](file:///d:/Study/ResCancel/scripts/run_operator_compression_confirmatory.py)

---

## 1. Executive Summary & Objective

Pilot experiments established that Vision Transformer patch token streams exhibit extreme downstream transmission anisotropy. Rather than minimizing ordinary representation error $\|E\|_F^2 = \|P - S C\|_F^2$, an operator-aware compression method steers error into the downstream-invisible linear subspace by solving:
$$\min_{C} \|J_{l \to L} \text{vec}((P - S C)^\top)\|_2^2 + \lambda \|P - S C\|_F^2$$
where $J_{l \to L} = \frac{\partial z_{\text{readout}}}{\partial \text{vec}(P_l^\top)}$ is the multi-block downstream Jacobian from intervention depth $l$ to the pre-classifier readout.

While pilot results indicated substantial damage reductions over standard pruning and Euclidean mean-merging, pilot evaluations were performed on small sample subsets. This protocol defines the **Strict Confirmatory Benchmark** to evaluate whether Operator-Aware Compression robustly shifts the accuracy-token frontier on held-out data across architectures, budgets, and random seeds, with **all algorithmic parameters frozen prior to confirmatory execution**.

---

## 2. Pre-Registered & Frozen Method Specifications

All hyperparameter choices, solver formulations, grouping methods, and intervention depths are strictly frozen as specified below. No tuning is permitted on the confirmatory evaluation split.

### 2.1 Primary Method: Operator-Aware Oracle Compression
- **Operator:** Per-image exact downstream Jacobian $J_{l \to L} \in \mathbb{R}^{D_{\text{readout}} \times ND}$.
- **Terminology:** Designated formally as **Operator-Aware Oracle Compression** due to the requirement of backward VJP computation per image.
- **Regularization Parameter:**
  $$\lambda = \lambda_{\text{factor}} \cdot \frac{\text{Tr}(\Sigma)}{D_{\text{readout}}}, \quad \lambda_{\text{factor}} = 10.0$$
  where $\Sigma = \sum_{j=1}^B \frac{1}{m_j} K_j K_j^\top \in \mathbb{R}^{D_{\text{readout}} \times D_{\text{readout}}}$ is the carrier Gram matrix and $K_j = \sum_{i \in G_j} J_{i} \in \mathbb{R}^{D_{\text{readout}} \times D}$.
  *Rationale:* Frozen based on pilot $\lambda$-sweep showing that $\lambda_{\text{factor}} = 10.0$ optimally balances linear transmission cancellation against high-order manifold distortion.
- **Numerical Solver:** Exact closed-form linear system solve via Cholesky / LU decomposition:
  $$\tilde{\alpha} = (\Sigma + \lambda I)^{-1} r_{\text{mean}}$$
  $$C_j^* = C_{\text{mean}, j} + \frac{1}{m_j} K_j^\top \tilde{\alpha}$$
  where $r_{\text{mean}} = J_{l \to L} \text{vec}((P - S C_{\text{mean}})^\top)$.

### 2.2 Primary Grouping Strategy
- **Grouping Rule:** Deterministic **Feature-Similarity Grouping**.
- **Specification:** Tokens are normalized to unit sphere ($P / \|P\|_2$). Seed medoid chosen deterministically (token 0), subsequent medoids selected via greedy farthest-point sampling under cosine distance. All tokens assigned to nearest medoid by cosine similarity.
- **Controlled Grouping Ablations (Same-Group Controls):**
  1. *Random Grouping:* Uniform partition of tokens using canonical seeds.
  2. *Spatial Grouping:* 2D regular grid partitioning on the $14 \times 14$ (or $16 \times 16$) patch lattice.

### 2.3 Intervention Depths
Established late intervention depths from prior fungibility audits:
- **DeiT-Tiny:** Block index $l = 8$ (of 12, remaining: 4 blocks)
- **DeiT-Small:** Block index $l = 8$ (of 12, remaining: 4 blocks)
- **ViT-B/16 AugReg:** Block index $l = 7$ (of 12, remaining: 5 blocks)
- **DINOv2 ViT-S/14:** Block index $l = 8$ (of 12, remaining: 4 blocks)

### 2.4 Multiplicity-Aware Sequence Shortening & Collapse Parity
- **Attention Bias:** In all downstream blocks, attention weights are corrected by injecting log-multiplicity bias:
  $$\text{attn\_bias}_j = \log(m_j)$$
  $$\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d}} + \log(m)\right) V$$
- **DINOv2 Output Readout:** Readout is $[h_{\text{norm}, \text{cls}}, \bar{h}_{\text{patch}}]$ where:
  $$\bar{h}_{\text{patch}} = \frac{1}{N} \sum_{j=1}^B m_j h_{\text{norm}, j}$$
- **Numerical Parity Criterion:** Maximum absolute logit difference between the sequence-shortened model ($1 + B$ tokens) and the full-length surrogate model ($1 + N$ repeated tokens) must satisfy $\|z_{\text{comp}} - z_{\text{full}}\|_\infty < 10^{-5}$.

---

## 3. Matched-Budget Baseline Suite

To ensure an unyielding confirmatory benchmark, Operator-Aware Compression is evaluated against 7 distinct training-free baselines at exactly matched downstream sequence lengths ($B$ tokens):

1. **Random Pruning (Baseline A):** Uniformly randomly selects $B$ patches to retain; remaining patches are dropped without carriers.
2. **Norm-Based Pruning (Baseline B):** Retains the top $B$ patches with largest feature $L_2$ norm $\|P_i\|_2$.
3. **Attention-Based Pruning (Baseline C):** Retains the top $B$ patches receiving the highest attention from the $[CLS]$ token at layer $l$:
   $$w_i = \frac{1}{H} \sum_{h=1}^H \text{Softmax}\left(\frac{q_{\text{cls}, h} k_{i, h}^\top}{\sqrt{d_h}}\right)$$
4. **Group-Mean Merging (Baseline D):** Feature-similarity grouping into $B$ clusters; carrier is the standard Euclidean centroid $C_{\text{mean}, j} = \frac{1}{m_j} \sum_{i \in G_j} P_i$ with multiplicity-aware attention correction.
5. **Medoid Merging (Baseline E):** Feature-similarity grouping into $B$ clusters; carrier is the actual patch token closest to the cluster mean: $C_{\text{medoid}, j} = \arg\min_{P_i \in G_j} \|P_i - C_{\text{mean}, j}\|_2$.
6. **Unweighted Centroid Carrier (Baseline F):** Cluster centroids used as carriers, but downstream attention ignores multiplicities ($\text{mult}_j = 1$).
7. **ToMe-Style Bipartite Soft Matching (Baseline G):** Faithful training-free Bipartite Soft Matching (Bolya et al., 2022). Tokens are partitioned into sets $\mathcal{A}$ and $\mathcal{B}$; cosine similarities determine the top $r_{\text{merge}} = N - B$ pairs to merge, updating carriers and token weights iteratively.

---

## 4. Evaluation Split & Token Budgets

### 4.1 Canonical Held-Out Split
- **Dataset:** ImageNet-1k Validation Set.
- **Evaluation Subset:** Exactly $N = 1,000$ images (stratified 1 sample per class, random seed 9201).
- **Strict Disjointness:** Zero overlap with the calibration split (seed 9101, 1,000 images). No hyperparameter tuning or method adjustments permitted on this split.

### 4.2 Token Budget Grids
Budgets span moderate compression (75% retained) to aggressive compression (16% retained):
- **For $N = 196$ (DeiT-Tiny, DeiT-Small, ViT-Base):**
  - $B = 147$ ($75.0\%$ tokens, 1.33× compression)
  - $B = 98$ ($50.0\%$ tokens, 2.00× compression)
  - $B = 72$ ($36.7\%$ tokens, 2.72× compression)
  - $B = 49$ ($25.0\%$ tokens, 4.00× compression)
  - $B = 32$ ($16.3\%$ tokens, 6.13× compression)
- **For $N = 256$ (DINOv2 ViT-S/14):**
  - $B = 192$ ($75.0\%$ tokens, 1.33× compression)
  - $B = 128$ ($50.0\%$ tokens, 2.00× compression)
  - $B = 94$ ($36.7\%$ tokens, 2.72× compression)
  - $B = 64$ ($25.0\%$ tokens, 4.00× compression)
  - $B = 42$ ($16.4\%$ tokens, 6.10× compression)

### 4.3 Stochastic Seeds
For stochastic baselines (Random Pruning, Random Grouping), results are evaluated across 5 canonical seeds:
$$\mathcal{S} = \{31001, 31002, 31003, 31004, 31005\}$$
Reporting includes mean, standard deviation, and paired image-level statistics.

---

## 5. Secondary Controlled Ablations

1. **Low-Rank Operator Truncation:**
   Truncate $J_{l \to L}$ using top $r$ singular vectors of $J J^\top$:
   $$r \in \{16, 32, 64\}$$
   Examines whether a tiny downstream-visible subspace captures the full compression advantage.
2. **Cumulative Joint Value-Key Operator ($C_{l \to L}$):**
   Evaluate whether the analytical Value-Key cumulative operator achieves comparable accuracy without backward passes.
3. **Calibration-Averaged Operator ($\bar{J}$):**
   Evaluate a static, dataset-level average operator $\bar{J} = \mathbb{E}[J_{l \to L}]$ to confirm the necessity of dynamic image-conditioned geometry.

---

## 6. Metrics & Statistical Testing

### 6.1 Primary Endpoint
- **Top-1 Accuracy:** Matched-budget classification accuracy on the 1,000 held-out images.
- **Normalized Retention:**
  $$\rho(B) = \frac{\text{Top1}_{\text{method}}(B)}{\text{Top1}_{\text{clean}}}$$
- **Frontier AUC (AUC-Token):** Area under the accuracy vs. remaining token fraction curve:
  $$\text{AUC} = \int_{f_{\min}}^{1.0} \text{Top1}(f) \, df$$

### 6.2 Secondary Mechanistic & Functional Metrics
- **Clean Prediction Agreement:** Fraction of samples where compressed prediction matches clean uncompressed model.
- **Prediction Flip Rate:** Fraction of samples where the compressed model flips the top-1 prediction.
- **True-Class Margin Damage:** $\Delta M = (z_{\text{clean}}[y^*] - \max_{c \ne y^*} z_{\text{clean}}[c]) - (z_{\text{comp}}[y^*] - \max_{c \ne y^*} z_{\text{comp}}[c])$.
- **Logit $L_2$ Distance:** $\|z_{\text{comp}} - z_{\text{clean}}\|_2$.
- **Kullback-Leibler Divergence:** $D_{\text{KL}}(p_{\text{clean}} \parallel p_{\text{comp}})$.
- **Downstream Operator Residual:** $\|J_{l \to L} \text{vec}(E^\top)\|_2$.
- **Carrier Displacement Norm:** $\|\delta C\|_F = \|C_{\text{opt}} - C_{\text{mean}}\|_F$.

### 6.3 Paired Statistical Tests
For all direct comparisons (Operator vs. Best Pruning, Operator vs. Best Merging, Operator vs. Group-Mean):
1. **McNemar's Test:** Exact test on paired discordant classifications ($b$ vs. $c$).
2. **Paired $t$-test & Wilcoxon Signed-Rank Test:** On paired true-class margin damage.
3. **Bootstrap 95% Confidence Intervals:** 2,000 resamples for accuracy and margin differences.
4. **Cohen's $d_z$ Effect Size:** Standardized mean difference of paired image damages.

---

## 7. Confirmatory Decision Levels

The confirmatory outcome will be categorized strictly into one of four descriptive levels:

- **LEVEL A — Strong Constructive Result:**  
  Operator-Aware Compression consistently shifts the accuracy-token frontier above both the strongest pruning baseline and the strongest merging baseline across all 4 architectures and across the budget sweep.
- **LEVEL B — Mechanism-Specific Improvement:**  
  Operator-Aware carriers consistently and statistically significantly outperform Group-Mean merging under identical token groupings ($S$), but performance relative to the global pruning/merging frontier is mixed.
- **LEVEL C — Model- or Budget-Conditional Advantage:**  
  Statistically significant frontier improvement is observed only in specific architectures (e.g. DeiT) or under aggressive token budgets ($B \le 49$).
- **LEVEL D — Pilot Fails to Replicate:**  
  On the full $N=1,000$ held-out benchmark, Operator-Aware Compression shows no statistically significant advantage over standard group-mean merging or matched-budget baselines.
