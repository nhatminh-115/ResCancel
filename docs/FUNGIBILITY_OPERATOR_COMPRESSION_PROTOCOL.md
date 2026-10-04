# Fungibility Operator Compression Protocol

## 1. Scientific Objective

In previous stages of the patch-content fungibility program, we established:
1. **Single-Block & Multi-Block Downstream Geometry:** The downstream Jacobian $J_{l \to L} \in \mathbb{R}^{D_{\text{readout}} \times (ND)}$ and cumulative Joint Value-Key operator $C_{l \to L}$ predict held-out perturbation damage ($r > 0.98$) and characterize a massive downstream low-transmission subspace ($\ge 98.98\%$).
2. **Prior Compression Failure:** Historical compression attempts using generic centroids, image-mean carriers, PCA, and geometry banks consistently failed to outperform simple matched-budget pruning.
3. **The Constructive Hypothesis:** Previous carrier constructions failed because they attempted to minimize raw Euclidean representation error $\|P - \hat{P}\|_F^2$ or used image-agnostic prototypes. In contrast, our mechanistic theory establishes that **not all representation error matters equally**: errors lying in the null space of the downstream transmission operator $J_{l \to L}$ (or $C_{l \to L}$) are invisible to the downstream network.

The objective of this protocol is to construct, evaluate, and benchmark **Operator-Aware Token Compression**:
Can the discovered downstream fungibility geometry be used to construct compressed carrier states $C \in \mathbb{R}^{B \times D}$ ($B \ll N$) that place the inevitable compression error $E = P - S C$ directly into the downstream-invisible subspace, outperforming geometry-agnostic merging and pruning?

---

## 2. Mathematical Formulation of Operator-Aware Compression

### 2.1 The Representation-Compression Setup
Let $P \in \mathbb{R}^{N \times D}$ be the clean spatial patch token activations at intervention layer $l$.
We wish to compress the $N$ patches into $B$ carrier tokens:
$$C \in \mathbb{R}^{B \times D}, \quad B \ll N.$$
Let $S \in \{0, 1\}^{N \times B}$ be an assignment matrix where $S_{i, j} = 1$ if patch $i$ is assigned to carrier $j$ (with $\sum_{j=1}^B S_{i, j} = 1$).
The reconstructed full-length surrogate stream is:
$$\hat{P} = S C \in \mathbb{R}^{N \times D}.$$
The representation error induced by compression is:
$$E = P - \hat{P} = P - S C \in \mathbb{R}^{N \times D}.$$

### 2.2 The Operator-Aware Least-Squares Objective
Ordinary Euclidean merging (such as k-means or ToMe) minimizes the raw Frobenius error:
$$\min_C \|P - S C\|_F^2 \implies C_{\text{mean}, j} = \frac{1}{m_j} \sum_{i \in G_j} P_i$$
where $m_j = |G_j| = \sum_{i=1}^N S_{i, j}$ is the multiplicity (group size) of carrier $j$.

In contrast, the **Operator-Aware Carrier Objective** minimizes downstream transmission:
$$\min_C \mathcal{L}(C) = \| J_{l \to L} \, \text{vec}((P - S C)^\top) \|_2^2 + \lambda \|P - S C\|_F^2$$
where $J_{l \to L} \in \mathbb{R}^{D_{\text{readout}} \times (ND)}$ is the downstream Jacobian, and $\lambda \ge 0$ is a Tikhonov regularizer preventing carrier states from departing the natural representation manifold.

### 2.3 Exact Closed-Form Linear Least-Squares Solution
Let $y = \text{vec}(P^\top) \in \mathbb{R}^{ND}$ and $x = \text{vec}(C^\top) \in \mathbb{R}^{BD}$.
Using the Kronecker identity $\text{vec}(C^\top S^\top) = (S \otimes I_D) \text{vec}(C^\top)$:
$$\text{vec}((S C)^\top) = (S \otimes I_D) x.$$
Let $K \equiv J_{l \to L} (S \otimes I_D) \in \mathbb{R}^{D_{\text{readout}} \times (BD)}$ and $b \equiv J_{l \to L} y \in \mathbb{R}^{D_{\text{readout}}}$.
The downstream transmission is:
$$J_{l \to L} \text{vec}(E^\top) = b - K x.$$

At the group mean $C_{\text{mean}}$, the mean residual is:
$$r_{\text{mean}} \equiv J_{l \to L} \text{vec}((P - S C_{\text{mean}})^\top) \in \mathbb{R}^{D_{\text{readout}}}.$$
Writing $C_j = C_{\text{mean}, j} + \tilde{C}_j$ (so $x = x_{\text{mean}} + \tilde{x}$):
$$b - K x = r_{\text{mean}} - K \tilde{x}.$$
The regularizer around the group mean is:
$$\|P - S C\|_F^2 = \|P - S C_{\text{mean}}\|_F^2 + \tilde{x}^\top (S^\top S \otimes I_D) \tilde{x}.$$
Because $S^\top S = \text{diag}(m_1, \dots, m_B)$, the weight matrix is $W = \text{diag}(m_1 I_D, \dots, m_B I_D)$.

Let $J_i \in \mathbb{R}^{D_{\text{readout}} \times D}$ denote the $i$-th patch column block of $J_{l \to L}$.
For group $j$, the $j$-th block of $K$ is the group column sum:
$$K_j = \sum_{i \in G_j} J_i \in \mathbb{R}^{D_{\text{readout}} \times D}.$$
The Gram matrix $K W^{-1} K^\top$ is:
$$\Sigma \equiv \sum_{j=1}^B \frac{1}{m_j} K_j K_j^\top \in \mathbb{R}^{D_{\text{readout}} \times D_{\text{readout}}}.$$

The exact closed-form solution for the carrier adjustment is:
$$\tilde{\alpha} = (\Sigma + \lambda I)^{-1} r_{\text{mean}} \in \mathbb{R}^{D_{\text{readout}}}$$
$$\tilde{C}_j = \frac{1}{m_j} K_j^\top \tilde{\alpha} = \frac{1}{m_j} \left( \sum_{i \in G_j} J_i \right)^\top \tilde{\alpha} \in \mathbb{R}^D$$
$$C_j^* = C_{\text{mean}, j} + \tilde{C}_j.$$

This solution is non-iterative, closed-form, and computes in $<1$ millisecond on GPU because $D_{\text{readout}} \le 384 \ll BD$.

---

## 3. Sequence Shortening via Multiplicity-Aware Exact Carrier

Compression requires physical sequence reduction from $N$ tokens to $B$ tokens.
To ensure mathematical fidelity:
1. **Full-Length Surrogate:** $\hat{P} = S C^* \in \mathbb{R}^{N \times D}$.
2. **Physical Sequence Shortening:** The $N$ patches are replaced by the $B$ carrier tokens $C^* \in \mathbb{R}^{B \times D}$, plus the 1 CLS token (total length $1 + B$).
3. **Multiplicity Attention Correction:** Each carrier $j$ is assigned multiplicity $m_j = |G_j|$.
   For self-attention, the log-multiplicity bias $+\log(m_k)$ is added to the attention logits for keys:
   $$\text{attn\_bias}_k = \log(m_k), \quad k \in \{0, 1, \dots, B\}$$
   with $m_0 = 1$ for CLS.
4. **Mandatory Collapse Parity Test:**
   $$\text{Parity Error} = \| \text{logits}(\text{Full Repeated Surrogate}) - \text{logits}(\text{Collapsed } B\text{-Carrier}) \|_{\infty} \le 10^{-4}.$$

---

## 4. Controlled Benchmark Suite

### 4.1 Token Budgets
For $N = 196$ patches:
$$B \in \{147, 98, 72, 49, 32\}$$
corresponding to compression ratios $1.33\times$ ($75\%$), $2.0\times$ ($50\%$), $2.72\times$ ($36.7\%$), $4.0\times$ ($25\%$), and $6.12\times$ ($16.3\%$).

### 4.2 Grouping Strategies ($S$)
1. **Spatial Grouping:** Contiguous spatial grid patches.
2. **Feature-Similarity Grouping:** Nearest-neighbor cosine clustering on clean activations.
3. **Random Grouping:** Uniform random partition of patches (isolation control).

### 4.3 Matched-Budget Baselines
At every budget $B$ under identical conditions:
1. `pure_random_pruning`: Randomly drop $N - B$ tokens.
2. `norm_based_pruning`: Retain the $B$ tokens with highest activation norm.
3. `group_mean_merging`: $C_j = \text{mean}_{i \in G_j} P_i$ with multiplicity-aware attention.
4. `medoid_merging`: Real token in $G_j$ closest to the mean.
5. `unweighted_centroid_carrier`: Collapse to unweighted carrier ($s=1$).
6. `operator_aware_J`: Closed-form regularized carrier using $J_{l \to L}$.
7. `operator_aware_C_cum`: Closed-form regularized carrier using multi-block Value-Key operator $C_{l \to L}$.
8. `calibration_averaged_operator`: Closed-form carrier using calibration $\bar{J}$ (zero test-time backprop).

---

## 5. Experimental Deliverables

1. **Protocol Document:** `docs/FUNGIBILITY_OPERATOR_COMPRESSION_PROTOCOL.md`
2. **Scientific Report:** `docs/FUNGIBILITY_OPERATOR_COMPRESSION_REPORT.md`
3. **Implementation:** `patch_fungibility/operator_compression.py`, `scripts/run_operator_compression.py`
4. **Output CSVs & Manifest:** `outputs/fungibility_operator_compression/`
   - `reconstruction_objectives.csv`
   - `full_length_surrogate_results.csv`
   - `collapse_parity.csv`
   - `matched_budget_results.csv`
   - `residual_vs_damage.csv`
   - `budget_frontier.csv`
   - `grouping_ablation.csv`
   - `operator_ablation.csv`
   - `low_rank_ablation.csv`
   - `calibration_operator_results.csv`
   - `replication_summary.csv`
   - `validation_manifest.json`
5. **Publication Figures:** `figures/fungibility_operator_compression/`
   - `figure_a_reconstruction_objective.png`
   - `figure_b_matched_budget_accuracy.png`
   - `figure_c_accuracy_token_frontier.png`
   - `figure_d_operator_residual_vs_damage.png`
   - `figure_e_grouping_ablation.png`
   - `figure_f_operator_ablation.png`
   - `figure_g_low_rank_tradeoff.png`
   - `figure_h_cross_architecture_replication.png`

---

## 6. Execution Rules
- Strictly preserve `PAPER_DRAFT.md` untouched.
- Never kill background tasks without explicit instruction.
- Respect native dual pooling for DINOv2.
