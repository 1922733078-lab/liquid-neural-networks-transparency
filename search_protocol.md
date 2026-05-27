# Search Protocol and Screening Boundary

## Review Type

This archive supports a narrative technical survey with PRISMA-style transparency. It is not a registered systematic review and does not claim independent dual screening, conflict arbitration, formal risk-of-bias scoring, or effect-size meta-analysis.

## Review Questions

1. What technical property makes a neural model liquid, and where is the boundary between liquid networks and adjacent continuous-time or gated sequence models?
2. Which architectural families currently constitute the liquid neural network landscape?
3. What is known about expressivity, stability, memory, solver cost, sparsity, interpretability, and robustness?
4. Which application claims are supported by replicated evidence, and which remain promising but preliminary?
5. What benchmark and reporting protocol would make future liquid-network results comparable?

## Databases and Sources

Searches were last updated on 11 May 2026. The search covered:

- Google Scholar
- DBLP
- Crossref
- OpenAlex
- IEEE Xplore
- ACM Digital Library
- OpenReview
- arXiv
- Publisher pages

## Query Families

Five query families were used:

1. Exact liquid-network terms:
   - "liquid time-constant network"
   - "liquid neural network"
   - "closed-form continuous-time neural network"
   - "neural circuit policy"
   - "liquid structural state-space model"
2. Continuous-time and neural differential-equation terms:
   - "continuous-time recurrent neural network"
   - "neural ordinary differential equation"
   - "neural controlled differential equation"
   - "neural stochastic differential equation"
   - "adaptive time constant neural network"
3. Irregular and missing time-series terms:
   - "irregular time series"
   - "missing time series"
   - "time-aware recurrent neural network"
   - "medical irregular time series"
4. State-space, long-sequence, and time-series foundation-model terms:
   - "structured state space"
   - "selective state spaces"
   - "Mamba"
   - "long sequence forecasting"
   - "time series foundation model"
5. Application, deployment, robustness, and reporting-standard terms:
   - "liquid neural network robotics"
   - "liquid time-constant biomedical"
   - "liquid neural network wireless"
   - "liquid neural network manufacturing"
   - "liquid neural network neuromorphic"
   - "robustness"
   - "calibration"
   - "reproducibility checklist"

## Screening Counts

The search log contained 472 candidate records:

- Google Scholar: 128
- DBLP: 52
- Crossref: 78
- OpenAlex: 71
- IEEE Xplore / ACM Digital Library: 58
- OpenReview / arXiv: 85

After title, author, venue, DOI, and URL deduplication, 396 records remained. Of these, 290 entered technical screening, 228 were assessed for mechanism or benchmark relevance, and 204 were retained in the literature matrix.

## Inclusion Criteria

A source was eligible when it satisfied at least one of these roles:

- It introduced or extended a liquid-network mechanism.
- It provided a strong adjacent baseline, such as a neural ODE/CDE, gated RNN, transformer, reservoir model, or structured SSM.
- It defined a benchmark, reporting standard, robustness criterion, interpretability criterion, or reproducibility practice.
- It provided representative application evidence needed for evidence-maturity assessment.

## Exclusion Criteria

Sources were excluded from the retained matrix when they were:

- Duplicate records for the same source.
- Nontechnical or marketing-only material.
- Uses of the term "liquid" without an interpretable adaptive dynamical mechanism.
- Application reports whose model or evaluation protocol could not be interpreted from public information.
- Records outside the survey boundary unless they supplied essential background or benchmark context.

## Evidence-Maturity Scoring

Application evidence was graded using four binary criteria:

1. Public or externally accessible data.
2. Strong non-liquid baselines.
3. Independent replication or multi-dataset evaluation.
4. Deployment evidence or external validation.

Scores of 0-1 were treated as early evidence, 2 as early-to-medium, 3 as medium, and 4 as high. These labels are qualitative screening labels, not effect-size estimates.

## Recent Source Audit

Recent 2024-2026 records cited in the main text were checked through DOI, publisher, OpenReview, arXiv, or public URL metadata where available. The audit is provided in `data/recent_source_verification_20260527.tsv`.

## Interpretation Boundary

The matrix is an audit trail for transparency and reproducibility. It should not be interpreted as a preregistered systematic-review dataset.
