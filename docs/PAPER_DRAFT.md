# Patch Content Fungibility in Vision Transformers: Geometric and Diversity Constraints in Late-Layer Representations

> Draft status: manuscript construction started after experimental freeze at V1.  
> Canonical numerical source: `docs/PAPER_EVIDENCE_TABLE.md`.  
> Claim boundaries: `docs/PAPER_CLAIMS_AUDIT.md`.  
> Do not introduce numerical values not present in the canonical evidence table.

## Abstract

Vision Transformers preserve a spatial stream of patch tokens throughout inference, but it is unclear how much image-specific information those tokens must retain once computation reaches late layers. We study this question through causal activation substitution: after a selected Transformer block, we replace subsets of ordinary spatial patch activations while leaving the pretrained model, classification head, and remaining forward computation unchanged. Across DeiT-Tiny, DeiT-Small, supervised ViT-B/16 AugReg, and self-supervised DINOv2 ViT-S/14, late patch representations become strikingly content-fungible. Replacing image-specific patch activations with class-agnostic calibration centroids or coarse held-out Gaussian samples recovers 81.8--97.5% of the margin damage caused by zero replacement at each model's late fungibility window. This tolerance is not arbitrary: coordinate permutation and sign inversion of otherwise scale-matched replacements sharply degrade performance, showing that valid substitutions remain constrained by learned feature geometry. Under complete patch-stream replacement, making replacement tokens independent rather than identical produces large accuracy gains in CLS-readout models, with supporting margin-level evidence in DINOv2, revealing a causal dependence on token-to-token diversity. Finally, learned one-dimensional variation along late-layer principal directions substantially outperforms matched random directions, although the effect depends on direction and amplitude. These results identify a late-stage representational regime in which exact patch-specific content becomes dispensable while geometric alignment and spatial diversity remain computationally important. The findings provide a mechanistic account of late ViT patch representations and motivate future geometry-aware token compression methods without itself modifying sequence length or inference cost.

## 1. Introduction

Vision Transformers maintain a sequence of spatial patch tokens across many layers, even after those tokens have repeatedly exchanged information through self-attention. Early in the network, this design is intuitive: each patch token begins as a representation of a particular image region, and preserving its content appears necessary for visual processing. Later in the network, however, repeated global interaction makes the semantic role of an individual patch less obvious. A late patch representation may still occupy a spatial slot, yet its downstream function need not depend on retaining the exact image-specific content that originally entered that slot. This raises a basic mechanistic question: **what information must remain in late spatial patch activations for downstream Vision Transformer classification to function?**

Existing work has largely approached patch redundancy from an efficiency or attribution perspective. Token pruning and token merging ask which tokens can be removed or combined while preserving predictions, whereas attribution methods estimate which existing activations are important for a particular output. These approaches are valuable, but they do not directly answer whether a retained spatial token must preserve its own image-specific representation. Removing a token changes sequence length and downstream computation; measuring sensitivity leaves the representation itself unchanged. We instead hold the token slots and downstream architecture fixed and intervene directly on the content carried by ordinary spatial patch activations.

Our central experiment is activation substitution. For a frozen pretrained Vision Transformer, we run an image normally up to a chosen intermediate depth, keep the classification token untouched, and replace a controlled subset of spatial patch activations before continuing the remaining blocks. Replacement vectors are derived only from a disjoint calibration set and do not use the evaluation image, its label, or gradients. This setup lets us separate several questions that are often conflated. Does downstream computation require the exact image-specific patch vector, or merely a plausible late-layer representation? If a generic substitute works, must it align with the learned feature coordinates? If the entire real patch stream is removed, does downstream computation require the replacement tokens to remain distinct from one another?

The first result is that **late patch content becomes strongly fungible**. In DeiT-Tiny and DeiT-Small, replacing 25% of late patch activations with a single calibration centroid recovers 93.8% and 97.5% of the margin damage induced by zero replacement at Depth 8. The same phenomenon generalizes beyond the DeiT family: ViT-B/16 AugReg recovers 81.8% of zero damage at its strongest late window, while DINOv2 ViT-S/14 recovers 93.4% at Depth 9. In DINOv2, the contrast is particularly stark: zeroing only 25% of late patches drops Top-1 accuracy from 78.8% to 4.5%, whereas substituting a class-agnostic calibration centroid restores it to 74.1%. Thus, the downstream network can be highly sensitive to the presence and form of the patch stream while remaining surprisingly insensitive to the exact image-specific content of many individual patch activations.

This fungibility is not equivalent to arbitrary replacement. Geometry-preserving controls reveal that the surrogate activation must remain aligned with the learned late-layer feature space. In ViT-B, replacing 75% of patches with the correct centroid retains 65.4% Top-1 accuracy, but permuting the centroid's feature coordinates reduces accuracy to 38.0%, and sign inversion reduces it to 27.9%. In DINOv2, replacing 25% of patches with the correct centroid retains 74.6% accuracy, whereas coordinate permutation drops to 31.8% and sign inversion to 0.1%. These interventions preserve broad scale statistics while destroying feature-coordinate identity or orientation. The result is a more precise characterization than simple redundancy: late patch content is fungible only within a constrained region of learned representation geometry.

A second boundary emerges when the entire spatial stream is replaced. If every replacement token is identical, downstream computation collapses. Holding the marginal replacement distribution fixed while sampling tokens independently instead of broadcasting a shared vector produces large gains in CLS-readout models: +25.32 percentage points in DeiT-Tiny, +37.08 points in DeiT-Small, and +11.74 points in ViT-B. DINOv2 differs because its official classifier directly consumes the mean final patch representation; under 100% replacement, Top-1 accuracy remains at floor for both shared and independent replacements. Even there, independent variation improves true-class margin by 1.124 with a paired effect size of (d_z=0.301). We therefore treat token-to-token diversity as a strong causal requirement under complete replacement in CLS-readout models, with supporting margin-level evidence under DINOv2's pooled readout rather than claiming universal accuracy rescue.

The structure of this required variation is also informative. A single learned principal direction can provide substantial recovery relative to a matched random one-dimensional direction: 18.64% versus 10.80% in DeiT-Tiny, 28.08% versus 20.78% in DeiT-Small, and 40.10% versus 6.70% in ViT-B. However, earlier controls show that this effect depends on both direction and amplitude, so we do not claim that arbitrary rank-one variation is sufficient. Instead, the result suggests that downstream blocks are selectively compatible with low-dimensional variation aligned to learned late-layer geometry.

Together, these findings support a simple picture of late Vision Transformer computation. Patch tokens begin as image-specific spatial representations, but after repeated global interaction their exact content becomes progressively less necessary for classification. What remains necessary is not the original patch identity itself, but compatibility with the learned feature geometry and, under complete stream replacement, sufficient token-to-token variation to sustain downstream computation. We refer to this regime as **patch content fungibility**.

This paper makes four contributions. First, we introduce a causal substitution framework for testing what information must remain in ordinary spatial patch activations while keeping sequence length and downstream computation fixed. Second, we show that late patch-content fungibility generalizes across DeiT, supervised ViT-B, and self-supervised DINOv2. Third, we demonstrate that valid substitutions are geometry-constrained and that complete stream replacement exposes a causal dependence on token diversity, with the strength of the observable effect depending on classifier readout topology. Fourth, we show that learned low-dimensional variation can partially restore downstream computation, while direction and amplitude remain essential. Our study is mechanistic rather than an acceleration method: sequence length is unchanged throughout. Nevertheless, the results identify concrete constraints that future token compression or merging methods may exploit.

The remainder of the paper is organized as follows. Section 2 positions patch content fungibility relative to token pruning, token merging, attribution, register-token analyses, and representation redundancy. Section 3 defines the intervention framework and calibration protocol. Section 4 characterizes the emergence of content fungibility with depth. Section 5 isolates geometric constraints on valid replacements. Section 6 studies token-to-token diversity under complete stream replacement. Section 7 examines low-dimensional replacement structure. Section 8 evaluates cross-architecture and cross-training-regime generalization. Section 9 discusses limitations and implications for representation analysis and future compression methods.


## 2. Related Work

### 2.1 Vision Transformer patch representations

Vision Transformers represent an image as a sequence of patch embeddings and propagate those tokens through a stack of self-attention blocks, typically with a dedicated classification token used for image-level prediction (Dosovitskiy et al., 2021). DeiT showed that this architecture can be trained effectively on ImageNet with data-efficient recipes and distillation (Touvron et al., 2021), while DINOv2 demonstrated that self-supervised ViT backbones can learn highly transferable visual representations at scale (Oquab et al., 2023). These works establish patch tokens as the basic spatial computational units of modern ViTs, but they do not ask whether a late patch slot must preserve the exact image-specific representation that arrived at that slot.

Recent analyses have made the distinction between global and local token roles increasingly explicit. Darcet et al. (2023) identified high-norm artifact tokens in ViT feature maps that are often located in low-information background regions and appear to be repurposed for internal computation, motivating the introduction of dedicated register tokens. Marouani et al. (2026) further studied the interaction between [CLS] and patch tokens and found that standard ViTs implicitly differentiate their processing despite sharing nominally identical architectural operations. Their results support the broader view that patch-token function evolves with depth and need not remain equivalent to local image content. Our question is narrower and intervention-based: once a model has reached a late layer, what properties of the ordinary spatial patch activations must actually be preserved for the remaining computation to succeed?

### 2.2 Token pruning, merging, and late-token redundancy

A large efficiency literature demonstrates that not all visual tokens need to be propagated at full cost. DynamicViT learns input-dependent token importance and progressively prunes tokens, obtaining substantial FLOP and throughput reductions with little accuracy loss (Rao et al., 2021). Token Merging (ToMe) instead combines similar tokens without retraining, reducing sequence length while preserving aggregate information and improving throughput across image, video, and audio transformers (Bolya et al., 2023). Related dynamic-computation methods similarly adapt token count or computation to image difficulty.

More recently, analyses of vision-language models have reported strong depth dependence in visual-token utility. Wang et al. (2026) describe an “information horizon” beyond which visual-token information becomes increasingly uniform or redundant and show that random late-layer token pruning can be competitive with importance-based pruning in that regime. Although this result concerns VLLMs rather than standalone image-classification ViTs, it reinforces a recurring observation: the functional role of visual tokens changes substantially with depth.

Our study differs from pruning and merging in a key experimental respect. We **do not remove, skip, or combine token slots** in the core experiments. Sequence length and downstream Transformer structure are held fixed. Instead, we overwrite the *content* of selected spatial activations and ask which representational properties remain necessary. This separation matters because a token can be redundant in content while its slot, geometry, or token-to-token variation remains computationally useful.

### 2.3 Causal interventions on internal visual representations

Attribution methods traditionally estimate which image regions or tokens influence a prediction using gradients, attention, relevance propagation, or input perturbations. Causal Attribution via Activation Patching (CAAP; Izadi et al., 2026) moves closer to the interventionist perspective by directly transplanting internal patch representations between source and neutral target contexts to measure patch-level causal contribution. CAAP is designed to produce faithful spatial attribution maps: it asks *which contextualized patch representation contributes to a target prediction?*

Patch content fungibility asks a different causal question. Rather than identifying the importance of a specific source patch, we hold the evaluation image and token slots fixed and replace selected activations with class-agnostic calibration-derived surrogates. The dependent variable is not patch attribution but the downstream model's ability to continue computing under controlled loss of image-specific content. Thus, the two paradigms share activation-level intervention but differ in estimand: CAAP measures causal contribution of patch-associated evidence, whereas our substitutions test the **necessity of exact representational content** and isolate the geometric and diversity constraints that remain after that content is removed.

### 2.4 Registers and the limits of zero ablation

The closest methodological precedent is recent work on register-token ablation. Darcet et al. (2023) introduced register tokens after observing that some ViT patch tokens become high-norm computational artifacts. Parodi et al. (2026) subsequently showed that zero-ablation can substantially overstate dependence on exact register content: replacing DINO register activations with means, noise, or cross-image register values preserves performance far better than replacing them with zeros. Their results demonstrate that a large zero-ablation effect need not imply dependence on the precise image-specific activation value.

Our work adopts the same general methodological lesson—**zero is a destructive control, not a sufficient estimate of content necessity**—but studies a different object and extends the intervention logic substantially. We operate on ordinary spatial patch tokens rather than auxiliary registers, including models with no register tokens. We vary replacement fraction and depth, test coordinate permutation and sign inversion to isolate feature-space geometry, contrast shared and independent replacements to isolate token-to-token diversity, and examine learned low-dimensional variation. The resulting question is therefore not whether specialized registers require exact content, but what computational constraints remain when image-specific information is removed from the ordinary late spatial stream.

### 2.5 Positioning

The literature above establishes four adjacent facts: ViT patch roles evolve with depth; many visual tokens can be pruned or merged; internal activation interventions can measure causal patch effects; and zero-ablation can exaggerate content dependence in specialized register tokens. Patch content fungibility connects these observations but targets a distinct representational object. Our core contribution is a controlled substitution framework that keeps token slots and downstream computation fixed while separating **exact content**, **feature geometry**, and **token diversity** as experimentally manipulable factors in late spatial patch representations.

## 3. Patch Content Fungibility Framework

### 3.1 Problem formulation

Consider a pretrained Vision Transformer with (L) Transformer blocks. For an input image (x), let the token sequence after block (ell) be

[
H^{(ell)}(x) = [,c^{(ell)}(x);, p_1^{(ell)}(x),ldots,p_N^{(ell)}(x),] in mathbb{R}^{(N+1)	imes D},
]

where (c^{(ell)}) is the classification token and (p_i^{(ell)}) is the activation of spatial patch position (i). The remaining network from this intervention point to the classifier is denoted (F_{>ell}).

Our goal is not to determine whether patch position (i) is important in isolation. Instead, we ask whether the remaining computation requires the **exact image-specific value** (p_i^{(ell)}(x)). We therefore intervene after a normal forward pass through block (ell), overwrite selected patch activations, and then resume the untouched pretrained model:

[
	ilde{y} = F_{>ell}(	ilde{H}^{(ell)}).
]

For a spatial mask (Msubseteq{1,ldots,N}),

[
	ilde{p}_i^{(ell)} =
egin{cases}
R_i^{(ell)}, & iin M,\
p_i^{(ell)}(x), & i
otin M,
end{cases}
]

while

[
	ilde{c}^{(ell)} = c^{(ell)}(x)
]

in all primary patch interventions. Thus, all upstream image processing through block (ell) is preserved, the [CLS] state remains image-specific, sequence length is unchanged, and only the content supplied by selected spatial slots to subsequent blocks is manipulated.

This distinction is central to interpretation. A successful late-layer substitution does **not** imply that the original image or earlier patch computation was unnecessary. It implies only that, conditional on the computation already performed up to depth (ell), the downstream network does not require the exact activation values of the replaced patch slots.

### 3.2 Calibration-derived surrogate representations

Replacement vectors are constructed from a calibration set that is strictly disjoint from evaluation images. Let (mathcal{C}_{ell}) denote the collection of spatial patch activations extracted at depth (ell) from the calibration set. No evaluation activations, labels, or gradients are used to construct the replacements.

We define the calibration centroid

[
mu_{ell} = rac{1}{|mathcal{C}_{ell}|}sum_{pinmathcal{C}_{ell}} p.
]

**Centroid substitution** replaces every selected patch with the same class-agnostic vector:

[
R_i^{(ell)} = mu_{ell}.
]

This is a stringent test of exact-content necessity because (mu_{ell}) is not the representation of the current image, nor of any particular calibration image. It is a dataset-level prototype of a late patch activation.

For **diagonal-Gaussian substitution**, we additionally estimate the per-coordinate calibration standard deviation (sigma_{ell}) and independently sample

[
R_i^{(ell)} sim
mathcal{N}!left(
mu_{ell},
operatorname{diag}(sigma_{ell}^2)
ight).
]

This preserves coarse coordinate-wise first- and second-order statistics without preserving image-specific patch content, cross-coordinate covariance, or the original patch identity. We deliberately describe this as moment matching rather than “manifold matching.”

### 3.3 Destructive and geometric controls

Zero replacement,

[
R_i^{(ell)} = 0,
]

serves as a destructive baseline. A central motivation of the study is that zero replacement can confound content removal with severe out-of-distribution activation shift. Accordingly, claims of fungibility are based on the contrast between zero and calibration-derived replacements rather than on zero sensitivity alone.

To test whether any nonzero vector with similar global scale is sufficient, we use geometry-destroying controls. A **coordinate-permuted centroid** applies a frozen bijection (pi) to the feature dimensions,

[
R_i^{(ell)} = pi(mu_{ell}),
]

preserving the multiset of coordinate values and hence scalar mean, variance, and (L_2) norm while destroying feature-coordinate correspondence. A **sign-inverted centroid** uses

[
R_i^{(ell)} = -mu_{ell},
]

preserving norm while reversing orientation. Large performance differences between (mu_{ell}) and these controls indicate that successful substitution depends on learned feature geometry rather than merely nonzero activation magnitude.

### 3.4 Isolating token-to-token diversity

When every spatial patch is replaced, centroid substitution removes all variation across patch slots. To isolate whether downstream computation requires token-to-token diversity independently of the marginal replacement distribution, we compare two Gaussian interventions with matched per-token statistics.

In the **shared** condition, one Gaussian sample (epsilon) is drawn for an image and broadcast to every replaced spatial position:

[
R_i^{(ell)} = mu_{ell} + epsilon
quadorall i.
]

In the **independent** condition, each spatial token receives its own sample from the same distribution:

[
R_i^{(ell)} = mu_{ell} + epsilon_i,
qquad
epsilon_i overset{	ext{i.i.d.}}{sim}
mathcal{N}(0,operatorname{diag}(sigma_{ell}^2)).
]

The expected per-token norm and marginal distribution are therefore matched; the principal manipulated variable is whether different spatial slots carry distinct values. This comparison provides the cleanest causal test of token-to-token diversity under complete stream replacement.

### 3.5 Low-dimensional replacement variation

To determine whether useful diversity must occupy the full feature space, we eigendecompose the covariance of centered calibration patch activations. Let (v_k) and (lambda_k) denote eigenvectors and eigenvalues. A natural one-dimensional replacement along component (k) is

[
R_i^{(ell)}
=
mu_{ell}
+
z_isqrt{lambda_k},v_k,
qquad z_isimmathcal{N}(0,1).
]

We compare learned principal directions with random unit directions under matched variance. These interventions test whether downstream compatibility depends only on having non-identical tokens or also on the orientation and amplitude of their variation. Because the experiments show architecture-specific amplitude sensitivity, we treat low-dimensional recovery as direction- and scale-dependent rather than claiming generic rank-one sufficiency.

### 3.6 Spatial masks and replacement fraction

For fraction-based interventions, spatial positions are selected independently of image content, labels, attention scores, activation norms, or saliency. A seeded random permutation of the (N) spatial patch positions defines a nested family of masks: larger replacement fractions extend the same ordered prefix rather than selecting an unrelated subset. This design ensures that changes across fractions reflect progressively replacing more positions under a fixed spatial ordering.

The discovery and mechanism experiments use frozen deterministic masks specified by their corresponding protocols. A final dense-fraction robustness analysis is being run separately to quantify sensitivity to the random spatial ordering across multiple mask seeds; it will be integrated only after its results are audited. No result from that pending analysis is assumed in the present draft.

### 3.7 Models, data isolation, and forward validation

The study spans four pretrained Vision Transformer settings: DeiT-Tiny, DeiT-Small, supervised ViT-B/16 AugReg, and self-supervised DINOv2 ViT-S/14 with the no-register checkpoint and official ImageNet linear head. The first three use 196 spatial patches at the evaluated image resolution, while DINOv2 uses 256. DeiT and ViT-B classification is read from [CLS], whereas the evaluated DINOv2 head consumes the concatenation of normalized [CLS] and the mean normalized patch representation. This difference is explicitly accounted for when interpreting complete-stream replacement.

Calibration and evaluation each use 1,000 ImageNet-1k validation images with zero overlap. Calibration statistics are computed only from the calibration split. All model weights and classifier heads remain frozen. For ViT-B and DINOv2, the manual block-by-block forward paths used for intervention were verified against the untouched official forward implementation on held-out images, with exact prediction agreement before any scientific result was issued.

### 3.8 Evaluation metrics

For each image we record the true-class logit, strongest incorrect logit, true-class margin, Top-1 correctness, prediction flips, and directional correctness changes. Let (m_{mathrm{clean}}(x)) and (m_R(x)) be the true-class margins under clean and replacement conditions. We define mean margin damage as

[
D_R =
mathbb{E}_x[
m_{mathrm{clean}}(x)-m_R(x)
].
]

When zero replacement produces meaningful damage, we summarize how much of that destructive effect is avoided by a replacement (R) using

[
operatorname{Recovery}(R)
=
rac{D_{mathrm{zero}}-D_R}
{D_{mathrm{zero}}}.
]

Recovery is not reported as zero when the denominator is negligible; such cases are treated as not applicable. Statistical comparisons use the evaluation image as the independent unit, with paired margin comparisons, bootstrap confidence intervals, paired parametric and nonparametric tests, effect sizes, and exact McNemar tests for paired accuracy changes. Multiple stochastic replacement seeds are summarized before inferential comparisons rather than being pooled as independent images.

