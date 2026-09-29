# Paper Submission Assets

## Working title
**Patch Content Fungibility in Vision Transformers: Geometric and Diversity Constraints in Late-Layer Representations**

## Keywords
Vision Transformer; patch tokens; causal intervention; activation substitution; representation geometry; token diversity

## Highlights
- Late spatial patch activations can lose much of their exact image-specific content while downstream ViT classification remains functional.
- Valid class-agnostic substitutions are constrained by learned feature geometry rather than by activation magnitude alone.
- Complete patch-stream replacement reveals a causal requirement for token-to-token diversity in CLS-readout Vision Transformers.
- The phenomenon replicates across DeiT, supervised ViT-B, and self-supervised DINOv2 and is robust across dense replacement fractions and multiple spatial masks.
- Exact surrogate redundancy can be compressed computationally, but generic synthetic carriers do not outperform retaining real patches at matched token budgets.

## One-sentence contribution
We identify and causally decompose a late Vision Transformer regime in which ordinary spatial patch representations become content-fungible while remaining constrained by learned feature geometry and token-set diversity.

## Short significance statement
Token pruning shows that some late visual tokens can be removed, but removal confounds representational redundancy with sequence reduction. By holding token slots and downstream computation fixed while directly substituting internal patch activations, this study separates exact content necessity from structural requirements. The resulting evidence shows that late ViT computation can tolerate the loss of image-specific patch content while still requiring compatible feature geometry and, under complete replacement, diversity across tokens.

## Graphical-abstract concept
A three-panel schematic:
1. **Content:** after a late ViT block, replace many real patch activations with a class-agnostic centroid or Gaussian surrogate; prediction is largely preserved.
2. **Constraints:** permuting feature coordinates or making the entire replacement stream identical causes failure; correct geometry and token diversity restore function.
3. **Boundary:** collapsing duplicate centroid states reduces sequence cost, but at the same token budget real patches outperform generic carriers, illustrating “replaceable does not imply compressible.”

## Submission positioning
The manuscript should be positioned as a **mechanistic representation study**, not as an acceleration method. The primary novelty is ordinary late spatial patch content fungibility and its causal decomposition. Cross-model robustness and compression-boundary experiments strengthen the empirical scope but should not displace the central representation question.
