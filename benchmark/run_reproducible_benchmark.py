"""Reproducible synthetic benchmark for the liquid-network survey.

This script intentionally avoids heavyweight ML dependencies. It tests a
mechanism-level claim rather than a full LTC/CfC implementation: adaptive
time-scale updates can help when the label depends on irregular event gaps.

Outputs:
  benchmark_results/synthetic_irregular_benchmark_summary.csv
  benchmark_results/synthetic_irregular_benchmark_runs.csv
  benchmark_results/synthetic_irregular_benchmark_metadata.json
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmark_results"
OUT.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Config:
    train_n: int = 700
    test_n: int = 450
    seeds: int = 8
    sequence_horizon: float = 8.0
    background_events: int = 48
    reservoir_dim: int = 32
    ridge_l2: float = 1.0


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n: int, seed: int, cfg: Config, shifted_test: bool = False):
    """Generate irregular one-dimensional event streams.

    Class 1 contains a short positive-to-negative marker gap; class 0 contains
    a longer gap. Background events and distractor spikes make aggregate
    summary statistics intentionally weak.
    """
    rng = np.random.default_rng(seed)
    data: list[tuple[np.ndarray, np.ndarray]] = []
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        y = int(rng.integers(0, 2))
        labels[i] = y

        marker_1 = rng.uniform(1.5, 3.8)
        marker_gap = rng.uniform(0.2, 0.65) if y == 1 else rng.uniform(1.6, 3.0)
        marker_2 = min(marker_1 + marker_gap, cfg.sequence_horizon - 0.7)

        if shifted_test:
            raw = np.cumsum(rng.gamma(0.7, 0.20, size=cfg.background_events))
            background_t = raw / raw[-1] * cfg.sequence_horizon
            background_t += rng.normal(0.0, 0.015, size=cfg.background_events)
            background_t = np.clip(background_t, 0.0, cfg.sequence_horizon)
        else:
            background_t = np.sort(rng.uniform(0.0, cfg.sequence_horizon, size=cfg.background_events))

        times = np.concatenate([background_t, [marker_1, marker_2]])
        values = rng.normal(0.0, 0.22, size=times.shape[0])
        distractors = rng.choice(cfg.background_events, size=8, replace=False)
        values[distractors] += rng.choice([-1.0, 1.0], size=8) * rng.uniform(0.8, 1.5, size=8)
        values[-2] += 2.0 + rng.normal(0.0, 0.05)
        values[-1] += -2.0 + rng.normal(0.0, 0.05)

        order = np.argsort(times)
        data.append((times[order].astype(float), values[order].astype(float)))

    return data, labels


def summary_features(data: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    rows = []
    for times, values in data:
        delta = np.diff(times, prepend=times[0])
        rows.append(
            [
                values.mean(),
                values.std(),
                values.min(),
                values.max(),
                values[0],
                values[-1],
                np.mean(np.abs(values)),
                np.sum(np.abs(values) > 1.0),
                delta.mean(),
                delta.std(),
                delta.max(),
                np.sum(values[:-1] * values[1:] < -1.0),
            ]
        )
    return np.asarray(rows, dtype=float)


class ReservoirEncoder:
    """Fixed random encoder with either fixed or adaptive decay.

    This is not presented as a trained liquid neural network. It is a compact
    mechanism-isolation proxy: the readout is trained, while the recurrent
    dynamics are held fixed so the effect of adaptive decay is easy to ablate.
    """

    def __init__(self, dim: int, seed: int, mode: str):
        self.dim = dim
        self.mode = mode
        rng = np.random.default_rng(seed)
        self.input_weight = rng.normal(0.0, 0.9, size=dim)
        self.bias = rng.normal(0.0, 0.25, size=dim)
        if mode == "fixed_single_scale":
            self.base_tau = np.ones(dim) * 0.9
        else:
            self.base_tau = np.exp(np.linspace(math.log(0.15), math.log(3.5), dim))

    def transform(self, data: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        rows = []
        for times, values in data:
            state = np.zeros(self.dim, dtype=float)
            prev_time = times[0]
            states = []
            alphas = []

            for t, u in zip(times, values):
                dt = max(float(t - prev_time), 0.0)
                prev_time = float(t)
                event_gate = sigmoid(4.0 * (abs(float(u)) - 0.85))

                if self.mode == "adaptive_decay":
                    tau = self.base_tau * (0.55 + 3.0 * event_gate)
                    alpha = np.exp(-dt / (tau + 1e-6))
                    write = np.clip((1.0 - alpha) + 0.35 * event_gate, 0.03, 1.0)
                else:
                    alpha = np.exp(-dt / (self.base_tau + 1e-6))
                    write = 1.0 - alpha

                candidate = np.tanh(self.input_weight * float(u) + self.bias)
                state = np.tanh(alpha * state + write * candidate)
                states.append(state.copy())
                alphas.append(alpha.copy())

            stacked = np.asarray(states)
            alpha_stack = np.asarray(alphas)
            rows.append(
                np.concatenate(
                    [
                        stacked[-1],
                        stacked.mean(axis=0),
                        stacked.max(axis=0),
                        stacked.min(axis=0),
                        alpha_stack.mean(axis=0),
                    ]
                )
            )

        return np.asarray(rows)


def fit_ridge_classifier(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, l2: float):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
    x_train = np.clip(x_train, -50.0, 50.0)
    x_test = np.clip(x_test, -50.0, 50.0)

    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_test = np.c_[np.ones(len(x_test)), x_test]
    y_one_hot = np.zeros((len(y_train), 2), dtype=float)
    y_one_hot[np.arange(len(y_train)), y_train] = 1.0

    penalty = np.eye(x_train.shape[1])
    penalty[0, 0] = 0.0
    aug_x = np.vstack([x_train, math.sqrt(l2) * penalty])
    aug_y = np.vstack([y_one_hot, np.zeros((x_train.shape[1], y_one_hot.shape[1]))])
    weights, *_ = np.linalg.lstsq(aug_x, aug_y, rcond=None)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, -1e6, 1e6)
    scores = np.einsum("ij,jk->ik", x_test, weights, optimize=True)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores.argmax(axis=1), int(weights.size)


def score(y_true: np.ndarray, y_pred: np.ndarray):
    accuracy = float(np.mean(y_true == y_pred))
    f1_values = []
    for cls in (0, 1):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1_values.append(2.0 * precision * recall / (precision + recall + 1e-12))
    return accuracy, float(np.mean(f1_values))


def evaluate_model(model: str, train, y_train, test, y_test, seed: int, cfg: Config):
    if model == "summary_features":
        x_train = summary_features(train)
        x_test = summary_features(test)
    else:
        encoder = ReservoirEncoder(cfg.reservoir_dim, seed=3000 + seed, mode=model)
        x_train = encoder.transform(train)
        x_test = encoder.transform(test)

    start = time.perf_counter()
    prediction, parameter_count = fit_ridge_classifier(x_train, y_train, x_test, cfg.ridge_l2)
    fit_predict_seconds = time.perf_counter() - start
    accuracy, macro_f1 = score(y_test, prediction)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "readout_parameters": parameter_count,
        "fit_predict_ms_per_sequence": 1000.0 * fit_predict_seconds / len(y_test),
    }


def summarize(rows: list[dict]):
    grouped = {}
    for row in rows:
        key = (row["setting"], row["model"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (setting, model), items in sorted(grouped.items()):
        def values(name):
            return np.asarray([float(item[name]) for item in items])

        summary_rows.append(
            {
                "setting": setting,
                "model": model,
                "runs": len(items),
                "accuracy_mean": values("accuracy").mean(),
                "accuracy_std": values("accuracy").std(ddof=0),
                "macro_f1_mean": values("macro_f1").mean(),
                "macro_f1_std": values("macro_f1").std(ddof=0),
                "fit_predict_ms_per_sequence_mean": values("fit_predict_ms_per_sequence").mean(),
                "readout_parameters": int(round(values("readout_parameters").mean())),
            }
        )
    return summary_rows


def paired_permutation_pvalue(a: np.ndarray, b: np.ndarray, permutations: int = 4096):
    """Two-sided paired sign-flip permutation test for paired seed results."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(float(diff.mean()))
    rng = np.random.default_rng(12345)
    count = 0
    for _ in range(permutations):
        signs = rng.choice([-1.0, 1.0], size=diff.shape[0])
        if abs(float((diff * signs).mean())) >= observed - 1e-12:
            count += 1
    return (count + 1.0) / (permutations + 1.0)


def significance_rows(rows: list[dict]):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["setting"], row["model"]), []).append(row)
    out = []
    for setting in sorted({row["setting"] for row in rows}):
        adaptive = sorted(grouped[(setting, "adaptive_decay")], key=lambda x: int(x["seed"]))
        for baseline in ("fixed_single_scale", "fixed_multi_scale"):
            base = sorted(grouped[(setting, baseline)], key=lambda x: int(x["seed"]))
            for metric in ("accuracy", "macro_f1"):
                a = np.asarray([float(x[metric]) for x in adaptive])
                b = np.asarray([float(x[metric]) for x in base])
                out.append(
                    {
                        "setting": setting,
                        "comparison": f"adaptive_decay_vs_{baseline}",
                        "metric": metric,
                        "mean_difference": float((a - b).mean()),
                        "paired_sign_flip_p": float(paired_permutation_pvalue(a, b)),
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = Config()
    models = ["summary_features", "fixed_single_scale", "fixed_multi_scale", "adaptive_decay"]
    settings = [("in_distribution", False), ("time_gap_shift", True)]
    run_rows = []

    for seed in range(cfg.seeds):
        train, y_train = generate_dataset(cfg.train_n, 1000 + seed, cfg, shifted_test=False)
        for setting, shifted in settings:
            test, y_test = generate_dataset(cfg.test_n, 2000 + seed, cfg, shifted_test=shifted)
            for model in models:
                result = evaluate_model(model, train, y_train, test, y_test, seed, cfg)
                run_rows.append(
                    {
                        "setting": setting,
                        "seed": seed,
                        "model": model,
                        **result,
                    }
                )

    summary_rows = summarize(run_rows)
    sig_rows = significance_rows(run_rows)
    write_csv(OUT / "synthetic_irregular_benchmark_runs.csv", run_rows)
    write_csv(OUT / "synthetic_irregular_benchmark_summary.csv", summary_rows)
    write_csv(OUT / "synthetic_irregular_benchmark_significance.csv", sig_rows)

    metadata = {
        "purpose": "Mechanism-isolation case study for irregular event-gap classification.",
        "not_a_full_ltc_or_cfc_claim": True,
        "configuration": cfg.__dict__,
        "encoder_configuration": {
            "reservoir_dim": cfg.reservoir_dim,
            "activation": "tanh recurrent state update; sigmoid event gate for adaptive proxy",
            "input_weight_initialization": "Normal(0, 0.9)",
            "bias_initialization": "Normal(0, 0.25)",
            "fixed_single_scale_tau": 0.9,
            "fixed_multi_scale_tau": "log-spaced from 0.15 to 3.5",
            "adaptive_proxy_tau": "base_tau * (0.55 + 3.0 * sigmoid(4 * (abs(u) - 0.85)))",
            "readout": "ridge classifier with l2 = 1.0",
            "trainable_parameters": "readout only; recurrent encoders are fixed random features",
        },
        "models": {
            "summary_features": "Twelve aggregate statistics with ridge readout.",
            "fixed_single_scale": "Random recurrent encoder with one fixed time scale and ridge readout.",
            "fixed_multi_scale": "Random recurrent encoder with fixed heterogeneous time scales and ridge readout.",
            "adaptive_decay": "Random recurrent adaptive-decay proxy with input-dependent event-gated decay and matched ridge readout.",
        },
        "significance_test": "Two-sided paired sign-flip permutation test across the eight matched seeds; provided as a descriptive check for the didactic proxy, not as evidence for deployed liquid architectures.",
        "task": "Binary classification where the label depends on the time gap between a positive and a negative marker event under distractor spikes.",
        "settings": {
            "in_distribution": "Training and test background event times are sampled from the same uniform process.",
            "time_gap_shift": "Test background event times are sampled from a clustered gamma process to stress irregular time gaps.",
        },
    }
    (OUT / "synthetic_irregular_benchmark_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Wrote:")
    print(OUT / "synthetic_irregular_benchmark_runs.csv")
    print(OUT / "synthetic_irregular_benchmark_summary.csv")
    print(OUT / "synthetic_irregular_benchmark_significance.csv")
    print(OUT / "synthetic_irregular_benchmark_metadata.json")
    for row in summary_rows:
        print(
            f"{row['setting']:16s} {row['model']:19s} "
            f"acc={row['accuracy_mean']:.3f}+/-{row['accuracy_std']:.3f} "
            f"f1={row['macro_f1_mean']:.3f}+/-{row['macro_f1_std']:.3f}"
        )


if __name__ == "__main__":
    main()
