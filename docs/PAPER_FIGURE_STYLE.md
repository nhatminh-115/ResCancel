# Paper Figure Style Guide

This document defines the final visual language for the Patch Content Fungibility manuscript. Historical experiment figures remain untouched; the submission uses the regenerated figures under `figures/paper_final/`.

## Core principles

1. **Message-first:** every main figure should make one causal claim visible without reading the caption.
2. **One visual language:** identical typography, line weights, panel labels, spacing, and semantic colors across all plots.
3. **No decorative chart titles:** panel titles identify the model or comparison; the manuscript caption carries the full claim.
4. **No debug annotations:** seed IDs, long condition names, protocol prose, and internal experiment labels are removed from main figures.
5. **Same condition, same encoding:** zero, centroid, Gaussian, geometry-destroying controls, and learned directions keep the same color/marker meaning across figures.
6. **Supplement is still publication-grade:** supplementary figures use the same style, but may carry denser legends and more conditions.

## Typography and geometry

- Font family: sans-serif (DejaVu Sans / arXiv-safe default).
- Base text: 9.2 pt.
- Axis labels: 9.5 pt.
- Panel titles: 10.2 pt, semibold.
- Legend: 8.2–8.5 pt.
- Panel labels: bold 11 pt, `(a)`, `(b)`, etc.
- Main double-column width: approximately 7.2 in.
- Main 2x2 figures: approximately 7.2 x 5.1 in.
- Line width: 1.8 pt primary, 1.1 pt secondary.
- Marker size: 4–5 pt.
- Grid: light gray, y-axis only unless both axes materially aid reading.
- Top/right spines removed.
- Figures saved as both 400-DPI PNG and vector PDF.

## Semantic palette

- Clean reference: dark slate `#4C566A`
- Zero / destructive control: red `#C43C39`
- Calibration centroid: blue `#2F6FB0`
- Diagonal Gaussian / independent diversity: green `#3A8E5B`
- Coordinate permutation: amber `#D08C28`
- Sign inversion: purple `#8E5AA7`
- Static/shared collapse: muted red-gray `#A65A5A`
- Learned PC1: blue `#2F6FB0`
- Learned PC2: light blue `#79A7D3`
- Random direction: neutral gray `#7A7A7A`

Color is never the only cue: conditions also differ by line style, marker, or hatch where applicable.

## Final figure map

### Main manuscript

1. **Conceptual overview** — intervention design, geometry constraint, diversity constraint, and compression boundary.
2. **Depth emergence** — four model families, 25% replacement.
3. **Dense replacement dose-response** — four models, five spatial masks.
4. **Geometry constraint** — centroid vs coordinate permutation, sign inversion, and zero.
5. **Diversity constraint** — shared vs independent replacement, plus margin-gain panel.
6. **Direction-specific low-dimensional variation** — learned PC directions vs random direction.

### Supplementary

S1. Mask-seed robustness.
S2. Accuracy-retention thresholds.
S3. Natural vs energy-matched PCA rank.
S4. PC1 amplitude sensitivity.
S5. Effective-rank propagation.
S6. Exact carrier-equivalence audit.
S7. Accuracy vs downstream token count.
S8. Accuracy vs measured GPU latency.
S9. Synthetic geometry-bank delta vs random pruning.

## Interpretation guardrails reflected visually

- DINOv2 complete replacement is shown honestly at its accuracy floor; its diversity evidence is carried by true-class margin rather than an implied accuracy rescue.
- Geometry plots use actual feature-coordinate permutation and sign-inversion controls, not generic “OOD” placeholders.
- Low-dimensional plots distinguish learned directions from random directions and do not label the result as generic rank-one sufficiency.
- Compression figures remain supplementary and are framed as a boundary result, not as a new acceleration method.
