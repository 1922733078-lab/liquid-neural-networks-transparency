# Reproducible Synthetic Case Study

This supplementary case study supports the review manuscript section
"Reproducible Mechanism-Isolation Case Study." It is a lightweight controlled
example, not a full LTC/CfC benchmark and not a state-of-the-art claim.

## Purpose

The script tests whether an adaptive-decay recurrent encoder can be isolated
from matched fixed-decay baselines on a synthetic irregular event-gap
classification task. The label depends on the time gap between a positive and a
negative marker event under distractor spikes and irregular observation times.

## How To Reproduce

Run from the manuscript folder:

```bash
python run_reproducible_benchmark.py
```

The script uses Python, NumPy, and the standard library. It writes all outputs
to `benchmark_results/`.

## Outputs

- `benchmark_results/synthetic_irregular_benchmark_summary.csv`: mean and
  standard deviation over eight seeds; this file corresponds to Table 11 in the
  manuscript.
- `benchmark_results/synthetic_irregular_benchmark_runs.csv`: per-seed results
  for each model and test setting.
- `benchmark_results/synthetic_irregular_benchmark_metadata.json`: task,
  configuration, model descriptions, and evaluation-setting metadata.

## Models Compared

- `summary_features`: twelve aggregate statistics with a ridge readout.
- `fixed_single_scale`: random recurrent encoder with one fixed time scale.
- `fixed_multi_scale`: random recurrent encoder with fixed heterogeneous time
  scales.
- `adaptive_decay`: random recurrent encoder with input-dependent event-gated
  decay and a matched ridge readout.

## Interpretation Boundary

The result supports only a methodological point: adaptive time-scale mechanisms
can be tested through controlled ablation under matched baselines. It should not
be interpreted as evidence that liquid neural networks dominate real-world
time-series benchmarks.
