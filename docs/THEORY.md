# 2. Theory

## 2.1 Problem Formulation

An SVG document is a rooted, ordered tree *T* whose nodes carry a tag, a set of attributes, and — for shape elements — heavy geometry strings (path data `d`, polygon `points`). An edit request is a natural-language instruction *I*. The editing task is to produce a document *T′* such that the rendering of *T′* satisfies *I* while preserving everything *I* does not mention. Two properties make this harder than generic text editing: correctness is judged **visually** (through the renderer) but the edit must be expressed **symbolically** (through the XML); and the overwhelming majority of the document's characters — the geometry — must survive the edit byte-for-byte.

The model's output space distinguishes the two families of systems studied here. A *full-rewrite* system implements *f*: (*T*, *I*) → *T′* directly in text space: the model emits the entire new document. A *patch* system factors the same function through a constrained edit language: the model emits a program *P* drawn from a small operation vocabulary (`set_attributes`, `remove_attributes`, `insert_primitive`, `remove_element`), each operation addressing nodes by stable identifiers *n0, n1, …* assigned by a deterministic preorder walk, and a symbolic executor applies *P* to *T*. The output space shrinks from *all strings* to *all valid programs over the current tree*.

## 2.2 The Patch Hypothesis: Constrained Edits over Regeneration

Let *N* be the token length of the document and *k* the token length of the minimal edit description, with *k* ≪ *N* (in SVGEditBench, *N* is dominated by path data; a typical gold patch is tens of tokens against thousands). A generative model with per-token reliability *p* reproduces the unedited content of a rewrite only with probability on the order of *p*^(*N*−*k*): every emitted token is a fresh opportunity to corrupt geometry that should never have changed. A patch system does not re-emit unchanged content at all; the model's exposure is *p*^*k*. The hypothesis is not that small models cannot produce long correct outputs — it is that content which must not change should not pass through the model.

Constraining the output space has a second, equally important consequence: it converts silent corruption into detectable failure. Because a patch is a program rather than prose, it can be checked *before* it is applied — schema validity, per-task attribute allowlists, target scoping, cardinality bounds, and the protected-geometry invariant (SHA-256 hashes over `d`/`points` must be unchanged after execution). A rewrite that mangles a path yields a plausible-looking but wrong SVG; a patch that violates policy is rejected outright and scored as a failure. Verifiability, not merely brevity, is the point of the representation.

## 2.3 The Skeleton: Task-Sufficient Lossy Compression

The model does not need to see what it is forbidden to change. The skeleton *S*(*T*) is a lossy encoding of the document in which geometry attributes are replaced by (hash, length) witnesses, style properties are resolved through CSS inheritance so that every node exposes its *effective* fill, stroke, and opacity, and tree structure (parent, depth, sibling index) is preserved exactly.

The skeleton is *task-sufficient*: for edits restricted to the editable attribute set, the patch computable from (*S*(*T*), *I*) is identical to the patch computable from (*T*, *I*) — raw geometry informs no permitted decision — while the token cost of the context falls by roughly the share of the file that geometry occupies. This is a rate–distortion choice: context tokens are spent only on decision-relevant structure, and the discarded information is exactly the information the validator guarantees immutable.

## 2.4 Semantic Grounding: Closing the Referent Gap

Instructions refer to *appearances* — "the red part", "the small square in the corner" — while patches address *node IDs*. The residual problem after §2.3 is grounding a perceptual referring expression to XML nodes, and it is the dominant error source in selection-heavy tasks. The skeleton grounds color words (via resolved fill) but not spatial or size language: nothing in it says *where a node paints*.

A node's visible contribution is defined by *generate-and-compare*: render the document, re-render it with node *n* suppressed, and take the difference of the rasters,

> *V*(*n*) = *R*(*T*) − *R*(*T* ∖ *n*).

*V*(*n*) is occlusion-aware — a node fully covered by later siblings has *V*(*n*) = ∅ — and inheritance-faithful, which rendering *n* in isolation is not. *V*(*n*) is then reduced to a handful of sufficient statistics for reference resolution: bounding box in viewBox coordinates, area share, coarse position on a 3×3 grid, dominant rendered color, and a visibility flag. At roughly twenty tokens per node, these statistics form a **symbolic visual interface** that a frozen language model can read directly, with no training and no embedding projection.

This is the cheap path of visual grounding. The learned path encodes *V*(*n*) with a frozen ViT and scores nodes with a GNN over the DOM graph. Both instantiate the same principle: *semantic context is the per-node answer to "what do you look like and where are you," obtained by comparing renders with and without the component.* The analytic path costs *n* + 1 rasterizations per document and is cacheable indefinitely because the benchmark inputs are frozen; it should therefore be exhausted before the learned path is invoked, and it provides the baseline against which the learned path must justify its cost.

## 2.5 Compound Edits as Operator Composition

The basic tasks form a vocabulary of atomic, individually verifiable operators. A compound instruction is modeled as a composition *B_k* ∘ ⋯ ∘ *B_1* applied to evolving state *T_i* = apply(*B_i*, *T*_{i−1}), with the skeleton recomputed at every step so each operator sees the current document. Decomposition trades one hard inference for *k* easy ones at the price of error propagation: per-step success rates *q_i* compound to ∏ *q_i*. Decomposition therefore wins exactly when single-shot success on the compound edit is worse than the product of per-step successes — the empirical regime for small models, whose failure mode on compound requests is joint overload rather than per-primitive incapacity.

The decomposer is trained in two phases: supervised fine-tuning on synthetically chained examples (compositions of gold basic-task cases over shared source emoji, with paraphrased compound instructions), then GRPO refinement. GRPO fits because the reward — end-state quality measured by failure-aware MSE, gold-step recall, geometry preservation, and an over-decomposition penalty — is non-differentiable and is estimated group-relative across sampled decompositions, avoiding a separate value model.

## 2.6 Measurement Principles

Three commitments shape the evaluation. **Failures are priced in**: every case is scored, and invalid or unrenderable outputs receive the worst raster score (failure-aware MSE = 1), so reliability appears in every average rather than being filtered out of them. **Structure is separated from pixels**: patch precision/recall against a deterministically derived gold patch, and the protected-geometry rate, say *what changed*; raster MSE says *how it looks* — conflating them hides whether an architecture fails at selection or at rendering-relevant values. **Bottlenecks are isolated by ablation**: oracle-target injection removes the selection problem, the two-stage architecture separates selection from value generation, and all architectures run on identical frozen cases, so paired per-task deltas attribute improvement to the representation change that caused it. The evaluation is designed to answer *where error lives*, not only how much of it there is.
