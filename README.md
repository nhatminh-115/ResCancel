# ResCancel: Mechanistic Falsification of Residual Cancellation in Pretrained ViTs

[![Protocol: Pre-Registered](https://img.shields.io/badge/protocol-pre--registered-blue.svg)](docs/V0_PROTOCOL.md)
[![Hardware: <= 8GB VRAM](https://img.shields.io/badge/VRAM-%3C%3D_8GB-green.svg)](#hardware-budget)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

---

## 1. Central Research Question

> **In pretrained Vision Transformers, are extreme residual-cancellation events merely a normal and useful part of standard computation, or is there a subset that causally contributes to predictive fragility?**

This is a **mechanistic falsification experiment**, **NOT** a method-development experiment.
In V0, we deliberately do **not** propose a new architecture, router, gate, regularizer, training objective, or fine-tuning procedure. The goal is to determine rigorously and cheaply whether residual cancellation represents a harmful pathology justifying training-time intervention, or whether it constitutes productive, necessary computation.

---

## 2. Theoretical Motivation & Cancellation Metrics

In a transformer block, residual connections compute updates across self-attention and MLP:
$$x_{\text{new}} = x + \delta$$

When update vector $\delta$ opposes feature representation $x$, geometric cancellation and coordinate opposition occur. We instrument four core geometric and algebraic quantities:

1. **Coordinate Cancellation ($C_{L1}$):**
   $$C_{L1} = \frac{\|x\|_1 + \|\delta\|_1}{\|x + \delta\|_1 + \epsilon}$$
   Connects to ResLRP and measures elementwise sign opposition.
2. **Geometric Anti-Alignment ($\cos(x, \delta)$):**
   $$\cos(x, \delta) = \frac{\langle x, \delta \rangle}{\|x\|_2 \|\delta\|_2 + \epsilon}$$
3. **Relative Update Magnitude ($r$):**
   $$r = \frac{\|\delta\|_2}{\|x\|_2 + \epsilon}$$
4. **Residual Contraction ($q$):**
   $$q = \frac{\|x + \delta\|_2}{\|x\|_2 + \epsilon}$$

An **extreme candidate cancellation event** is strictly defined by the conjunction of:
- Strong negative cosine: $\cos(x, \delta) \le -0.60$
- Delta magnitude comparable to $x$: $r \ge 0.60$
- Substantial contraction after addition: $q \le 0.85$
- Coordinate cancellation: $C_{L1} \ge 2.00$

*(We do not define pathology using negative cosine alone).*

---

## 3. Confound Controls & Causal Intervention

To distinguish causal effects from confounds (hard images, update magnitude):
1. **Matched Controls:**
   Every high-cancellation event is paired with a control matched on:
   - Layer index ($0 \dots 11$)
   - Residual site (`attn` vs `mlp`)
   - Token type (`cls` vs `patch`)
   - $\|x\|_2$ ($\pm 15\%$)
   - $\|\delta\|_2$ ($\pm 15\%$)
   - Clean logit margin $\Delta z$ ($\pm 0.5$)
2. **Causal Intervention:**
   Decompose $\delta = \delta_\parallel + \delta_\perp$ relative to $x$. When $\delta_\parallel$ opposes $x$, selectively weaken only the opposing component:
   $$\delta' = \alpha \delta_\parallel + \delta_\perp, \quad \alpha \in \{1.0, 0.75, 0.50, 0.25\}$$
   Evaluated against matched random-direction perturbations and uniform scaling controls with identical norm shifts.

---

## 4. Repository Structure

```
ResCancel/
├── rescancel/                 # Core research package
│   ├── __init__.py
│   ├── metrics.py             # Residual geometry & prediction metrics (C_L1, cos, r, q, margin)
│   ├── instrumentation.py     # Clean block wrappers & causal intervention hooks
│   ├── dataset.py             # Reproducible ImageNet validation subset loader
│   ├── corruptions.py         # Controlled mild Gaussian noise & Gaussian blur
│   ├── confound_matching.py   # Caliper matching & multivariable logistic regression
│   ├── intervention.py        # Causal intervention & matched control sweeps
│   └── pipeline.py            # End-to-end falsification orchestrator
├── scripts/
│   ├── run_v0_pipeline.py     # Main CLI experiment runner
│   └── plot_figures.py        # Publication-ready figure generator
├── docs/
│   ├── V0_PROTOCOL.md         # Pre-registered and frozen experimental protocol
│   └── V0_REPORT.md           # Comprehensive scientific report & GO/KILL decision
├── outputs/                   # Machine-readable tables & experiment manifest
│   ├── experiment_manifest.json
│   └── *.csv
├── figures/                   # High-resolution plots
│   └── *.png
├── LICENSE
└── README.md
```

---

## 5. Quickstart & Reproducibility

### Installation
```bash
pip install torch torchvision timm pandas scipy matplotlib seaborn scikit-learn
```

### Running the Full V0 Pipeline
Run the pre-registered experiment on 1,000 stratified ImageNet validation samples:
```bash
python scripts/run_v0_pipeline.py --models deit_tiny_patch16_224 deit_small_patch16_224 --num-samples 1000 --batch-size 32
```

### Hardware Budget
- Target hardware budget: **$\le 8$ GB VRAM**.
- Peak VRAM observed: **$< 500$ MB** using streaming statistics and batch size 32.