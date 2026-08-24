from __future__ import annotations

import math

import numpy as np


def brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and probability must have the same shape")
    return float(np.mean((p - y) ** 2))


def binary_log_loss(y_true: np.ndarray, probability: np.ndarray, epsilon: float = 1e-12) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), epsilon, 1 - epsilon)
    if y.shape != p.shape:
        raise ValueError("y_true and probability must have the same shape")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))))


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (a.shape == lo.shape == hi.shape):
        raise ValueError("actual, lower and upper must have the same shape")
    if np.any(lo > hi):
        raise ValueError("lower interval cannot exceed upper interval")
    return float(np.mean((a >= lo) & (a <= hi)))


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    if bins < 2:
        raise ValueError("bins must be at least 2")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(p), 1)
    error = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (p >= left) & ((p < right) if idx < bins - 1 else (p <= right))
        if not np.any(mask):
            continue
        confidence = float(np.mean(p[mask]))
        accuracy = float(np.mean(y[mask]))
        error += (np.sum(mask) / total) * math.fabs(confidence - accuracy)
    return float(error)
