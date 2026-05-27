# Liquid Neural Networks Review: Transparency Archive

This repository provides transparency materials for a review manuscript on liquid neural networks and adaptive continuous-time neural computation.

It is a transparency archive, not a prospective registration of a systematic review. The archive is intended to make the literature screening record, recent-source verification, and synthetic benchmark demonstration auditable.

## Contents

- `data/literature_matrix.csv`: literature-screening matrix used for the survey audit trail.
- `data/literature_matrix.xlsx`: spreadsheet version of the literature-screening matrix.
- `data/recent_source_verification_20260527.tsv`: DOI/URL verification record for recent cited sources and target-journal positioning sources.
- `search_protocol.md`: search scope, query families, screening criteria, and interpretation boundaries.
- `benchmark/run_reproducible_benchmark.py`: synthetic irregular event-gap benchmark script.
- `benchmark/results/`: generated benchmark summary, per-seed results, and metadata.

## What Is Not Included

The repository does not include the submitted manuscript, final English manuscript, revision logs, cover letter, author-only submission documents, downloaded third-party article PDFs, or publisher-formatted copyrighted material.

## Interpretation Boundary

The literature matrix is an author audit trail for a narrative technical survey with PRISMA-style transparency. It is not a registered systematic-review database with independent dual screening, conflict arbitration, formal risk-of-bias scoring, or effect-size meta-analysis.

The synthetic benchmark is a didactic mechanism-isolation example. It should not be interpreted as evidence that any deployed LTC, CfC, NCP, or liquid SSM implementation dominates real-world time-series benchmarks.

## Reproducing the Benchmark

```bash
cd benchmark
python run_reproducible_benchmark.py
```

The script uses Python, NumPy, and the standard library. It writes outputs to `benchmark_results/` when run from the benchmark folder.

## Citation

If citing this archive, cite the GitHub release or the Zenodo DOI associated with the release, when available.
