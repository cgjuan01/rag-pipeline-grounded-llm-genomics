# Causally-anchored multi-omic deep learning for exercise and ageing

Juan CG, Ntasis L. Causally-anchored multi-omic deep learning recovers
exercise-responsive and ageing-causal genes from human physical activity.
medRxiv 2026. doi:10.64898/2025.12.26.25343061

## Aim

Physical activity is among the most robust epidemiological correlates of reduced
mortality and multi-morbidity, but the molecular mechanisms remain incompletely
resolved. This study combines causally-anchored multi-omic Mendelian
randomisation with graph-based deep learning for gene prioritisation, using
accelerometer-derived vigorous physical activity in UK Biobank as the exposure.
It asks whether causal inference and deep learning together can recover
exercise-responsive and potentially ageing-causal genes that were independently
identified in prior studies.

## Design

Five stages. An exome-wide association study identifies protein-coding variants
associated with habitual vigorous physical activity, providing genetic
instruments. LD-aware, overlap-aware, pleiotropy-filtered Mendelian
randomisation estimates the causal effect of activity on each gene across five
molecular layers. Gene-level MR effects are aggregated into a continuous
multi-omic trait importance score. A supervised graph attention network
propagates that signal over a protein-protein interaction network to produce a
denoised gene ranking. The ranked genes are then tested for statistical overlap
with independently defined exercise-responsive and ageing-causal gene sets.

## Exposure and instruments

Instruments were derived from UK Biobank whole-exome sequencing, restricted to
the subset of roughly 75,000 participants with seven-day wrist-worn
accelerometry. Vigorous activity was defined as the fraction of time above 400
milligravities, corresponding to at least six METs. Association testing used
REGENIE, adjusted for sedentary activity, age, sex, assessment centre, season of
wear, body mass index and Townsend Deprivation Index. The accelerometry subset
was retained in full rather than restricted to disease-free participants, because
conditioning on health status is conditioning on a collider and would bias the
instrument set. Fine mapping yielded 86 SNPs carried into the downstream MR.

## Mendelian randomisation

Two-sample MR with explicit modelling of linkage disequilibrium rather than
clumping it away. Causal estimates used an LD-aware inverse-variance-weighted
estimator with a covariance matrix built from the LD matrix; overlap-aware
correction was applied to the UK Biobank proteomic layer. Horizontal pleiotropy
was tested with the MR-Egger intercept, and gene-level estimates showing
directional pleiotropy were excluded, uniformly across layers. Only the relevance
assumption is fully verifiable, so estimates are described throughout as
causally-anchored rather than definitively causal.

## Multi-omic trait importance

Per-layer MR effect sizes were z-scored across genes, then aggregated into a
single continuous score as the root of the summed squared normalised effects
across layers with available evidence. All layers were weighted equally. Because
normalised effects are squared before summation, the score is non-negative and
reflects magnitude of multi-layer causal association irrespective of direction.
The score served as the supervised regression target, not as an independent
result.

## Graph and model

Nodes are genes, with features encoding standardised per-layer MR effect sizes.
The trait importance score and every quantity derived from it were excluded from
the feature matrix by an explicit leakage guard, whose absence was asserted
before training, so the model saw the per-layer effects but never the composite
it was asked to predict. Structural and functional annotations were removed as
node features, so the model could not recover biological enrichments by being
told in advance which genes carry which annotations. The chromosome 17q21.31
inversion region was excluded before graph construction.

Edges are STRING v12.0 protein-protein interactions retained at combined
confidence 700 or above, treated as unweighted. The connected component
comprised 2,473 genes joined by 17,193 high-confidence edges, and this connected
graph is the tested universe for all rank-based enrichment.

The encoder is a two-layer graph attention network: a graph attention
convolution, ELU activation, dropout, and a second graph attention convolution,
followed by two heads sharing the node embeddings — a regression head predicting
the trait importance score, and a classification head predicting a logit for
multi-layer support. Loss combines mean squared error with binary cross-entropy,
with a class-imbalance positive weight capped at 50. Trained with Adam for 300
epochs. Genes are ranked by a hybrid score combining z-scored predictions from
both heads.

## Enrichment results

Two background universes are kept separate: the connected STRING graph of 2,473
genes for all rank-based enrichment, and the multi-omic MR universe of 2,959
trait-importance-scored genes for the model-free convergence test.

Genes causally anchored to vigorous activity at FDR below 0.05 numbered 906, and
were significantly enriched for ageing-causal genes: 16 observed against 10.1
expected, a 1.6-fold enrichment at p = 0.023. This convergence between two
independent MR analyses was present before any deep learning step. It was not,
however, recoverable by ranking on MR statistics alone — neither best MR p-value
nor effect-size ranking produced significant enrichment at any depth.

The supervised graph model concentrated the ageing-causal signal among its
top-ranked genes and held significance across the top 100 to 200, reproduced in
five of five random initialisations. A supervised non-graph baseline, a
multilayer perceptron trained on the same node features, recovered enrichment
only at the top 100 and lost it as the ranking widened, localising the
depth-robust signal specifically to the graph architecture.

The exercise-responsive reference behaved differently. No method based on the MR
signal alone showed any enrichment: the FDR-significant set, p-value ranking,
effect-size ranking and the non-graph baseline were all non-significant, with the
raw MR set marginally depleted. Only the graph model recovered significance, at
44 observed against 32 expected among the top 100 genes, again in five of five
initialisations, weakening as the ranking widened.

## Convergence and validated target

Within the MR universe, three sets were compared: 906 activity-anchored genes,
948 acutely exercise-responsive genes, and 33 ageing-causal genes. The two
exercise sets shared 269 genes. Eight genes lay at the intersection of all three,
spanning lipid metabolism, lysosomal proteostasis, growth factor and matrix
signalling, transcriptional regulation and cytoskeletal signalling. Two of the
eight ranked in the model's top 20.

The eight were then tested for causal association with four ageing outcomes using
two-sample cis Mendelian randomisation with colocalisation, across protein and
expression instrument layers. A pair counted as validated only with
FDR-significant MR, a Steiger-consistent direction, and conditional posterior
probability of a shared causal variant above 0.7. One gene met all three
criteria: cathepsin F, causally associated with exceptional longevity, with
concordant positive estimates in both arms and colocalisation support at the
protein level. Several other FDR-significant MR signals failed colocalisation,
illustrating why colocalisation rather than MR p-value magnitude is the
discriminating criterion.

## What the model does and does not do

Because the supervision target is derived from the MR effect sizes, the network
performs no independent causal discovery: it cannot generate causal information
the upstream MR did not contain, and its attention weights indicate relative
importance within the learned representation rather than biological mechanism.
What it does is propagate sparse, distributed causal signal across the
interaction graph, so genes embedded among many causally-implicated neighbours
surface even when their own per-gene signal is weak. The central claim of the
study does not depend on the model at all: the convergence between
exercise-anchored and ageing-causal genes was established model-free.

## Limitations

Every instrument and outcome dataset derives from European-ancestry samples. All
outcome layers and the exercise-responsive reference are blood-accessible, so
extension to muscle, adipose, vascular or neural tissue requires data not
available here, and plasma proteomics is biased toward secreted factors. Of the
three MR assumptions only relevance is fully verifiable. The ageing-causal
reference contains only 33 testable genes, so the convergence rests on modest
numbers and was characterised by per-gene annotation rather than pathway
enrichment. The exercise-responsive reference reflects a single acute bout
whereas the exposure is a germline proxy for habitual activity, so the overlap
indicates broad exercise-responsiveness rather than validation of chronic
adaptation. Finally, a convergence is not a demonstration of mediation.
