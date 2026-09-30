# Reviewer-Facing Manuscript Audit

**Paper:** Patch Content Fungibility in Vision Transformers: Geometric and Diversity Constraints in Late-Layer Representations  
**Status:** Full draft after final experimental freeze  
**Canonical evidence:** `docs/PAPER_EVIDENCE_TABLE.md`

## 1. Main contribution hierarchy

1. **Primary novelty:** ordinary late spatial patch activations become content-fungible under fixed-sequence causal substitution.
2. **Primary mechanistic decomposition:** valid substitutions remain constrained by learned feature geometry and, under complete stream replacement, token-to-token diversity.
3. **Robustness/generalization:** replicated across DeiT-Tiny, DeiT-Small, supervised ViT-B/16 AugReg, and self-supervised DINOv2; dense dose-response is robust across five independently sampled spatial masks.
4. **Boundary result:** representational replaceability does not imply competitive compression. Duplicate centroid surrogates admit exact multiplicity-aware collapse, but matched-budget pruning and real patches dominate generic synthetic carriers.

Low-dimensional PC experiments support Contribution 2 but are not positioned as a standalone novelty claim.

## 2. Claims that must remain qualified

- Do not claim that visual content is unnecessary; interventions occur only after substantial upstream image processing.
- Do not claim a universal block index. Peak fungibility depth differs by architecture.
- Do not call the five-mask result spatial invariance; use "robust across the five tested spatial permutations."
- Do not claim DINOv2 accuracy recovery from diversity at 100% replacement; the support is margin-level because the official readout consumes patch mean.
- Do not claim that the natural late representation is rank-one. Learned 1D directions are useful probes and remain direction/amplitude dependent.
- Do not call diagonal Gaussian sampling manifold matching.
- Do not present the weighted carrier or geometry banks as a competitive acceleration method.
- Do not claim centroid replacement itself as novel. CCViT uses centroid replacement at the input/pre-training level; the novelty here is internal late-layer causal substitution in frozen models.

## 3. Closest prior art and distinction

- **Parodi et al., CVPRW 2026:** closest methodological precedent; shows exact content of specialized DINO register tokens is less necessary than zero ablation suggests. Our object is ordinary spatial patch tokens and the study additionally resolves depth, fraction, geometry, diversity, and low-dimensional constraints.
- **DynamicViT / ToMe:** remove or merge tokens for efficiency. Our primary experiments preserve sequence length and intervene on token content.
- **CAAP (2026):** activation patching for attribution. Its estimand is causal contribution/localization; ours is necessity of exact internal content and the properties that remain required.
- **CCViT (2023):** centroid replacement exists at input/pre-training level. It does not study frozen internal late-layer patch substitutions.
- **Wang et al., CVPR 2026:** late visual-token redundancy in VLLMs via pruning/information measures. Different model class and estimand.

## 4. Current main figures

1. Conceptual overview: `figures/paper_final/main/fig01_conceptual_overview.png`
2. Depth-dependent emergence: `figures/paper_final/main/fig02_depth_emergence.png`
3. Dense fraction robustness: `figures/paper_final/main/fig03_dense_fraction.png`
4. Geometry controls: `figures/paper_final/main/fig04_geometry_constraint.png`
5. Shared vs independent diversity: `figures/paper_final/main/fig05_diversity_constraint.png`
6. Learned vs random 1D: `figures/paper_final/main/fig06_lowdim_direction.png`

The six main figures share one publication style and are regenerated from frozen outputs by `scripts/render_paper_figures_final.py`. Compression-equivalence, latency, and geometry-bank negative results remain supplementary/boundary evidence.

## 5. Current remaining manuscript tasks

- Convert author-year prose citations to the target journal bibliography style after venue selection.
- Add final author affiliations/acknowledgments/data-access wording.
- Main-text figure set is frozen at six figures, including the conceptual overview; dense-fraction accuracy is the primary fraction figure. Nine additional plots are in `PAPER_SUPPLEMENTARY_DRAFT.md`.
- Final PNG and vector/PDF figure variants are generated under `figures/paper_final/`; preserve this visual language in venue-specific typesetting.
- Supplementary draft is complete at `docs/PAPER_SUPPLEMENTARY_DRAFT.md`; preserve its corrected final interpretations rather than copying stronger historical wording from stage reports.
- Run final numerical cross-check against `PAPER_EVIDENCE_TABLE.md` after any typesetting conversion.
- Perform final reference duplication/preprint-versus-proceedings audit before submission.

## 6. Reviewer stress-test verdict

The strongest defensible paper is a **mechanistic representation paper**, not an efficiency paper. The evidence chain is coherent: content substitution succeeds late; arbitrary geometry fails; diversity becomes causal under complete replacement; learned directions outperform arbitrary directions; the pattern generalizes across architectures and masks; compression attempts establish a clear boundary rather than an engineering win.
