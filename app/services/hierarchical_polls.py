from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Iterable

import numpy as np
import pandas as pd

from app.ml.pollster import PollsterCalibration


DEFAULTS = {
    "collection_mode": "unknown",
    "target_population": "registered_voters",
    "undecided_share": 0.0,
    "scope": "national",
    "uf": "BR",
}
REQUIRED_COLUMNS = {
    "poll_id", "field_date", "institute", "candidate_id", "candidate_name",
    "share", "sample_size", "margin_error",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _safe_cholesky(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2
    jitter = 1e-8
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + np.eye(matrix.shape[0]) * jitter)
        except np.linalg.LinAlgError:
            jitter *= 10
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-8, None)
    return vectors @ np.diag(np.sqrt(values))


def _softmax_alr(values: np.ndarray) -> np.ndarray:
    zeros = np.zeros((*values.shape[:-1], 1), dtype=float)
    logits = np.concatenate([values, zeros], axis=-1)
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True) * 100.0


def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    std = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    correlation = covariance / np.outer(std, std)
    correlation = np.clip((correlation + correlation.T) / 2, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


@dataclass
class PollPosterior:
    candidate_ids: list[str]
    candidate_names: list[str]
    state_ids: list[str]
    national_draws: np.ndarray
    state_draws: np.ndarray
    undecided_state_draws: np.ndarray
    national_summary: pd.DataFrame
    state_summary: pd.DataFrame
    institute_reliability: pd.DataFrame
    residual_covariance: np.ndarray
    residual_correlation: np.ndarray
    diagnostics: dict


def _prepare_poll_matrix(
    polls: pd.DataFrame,
    as_of_date: date,
) -> tuple[pd.DataFrame, list[str], list[str], np.ndarray, np.ndarray]:
    missing = REQUIRED_COLUMNS - set(polls.columns)
    if missing:
        raise ValueError(f"Colunas ausentes nas pesquisas: {sorted(missing)}")
    frame = polls.copy()
    for column, default in DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
    frame["uf"] = frame["uf"].fillna("BR").astype(str)
    frame.loc[frame["scope"] == "national", "uf"] = "BR"
    frame["field_date"] = pd.to_datetime(frame["field_date"]).dt.date
    frame = frame[frame["field_date"] <= as_of_date].copy()
    if frame.empty:
        raise ValueError("Não há pesquisas válidas até a data de referência.")

    candidates = frame[["candidate_id", "candidate_name"]].drop_duplicates().sort_values("candidate_id")
    candidate_ids = candidates["candidate_id"].astype(str).tolist()
    candidate_names = candidates["candidate_name"].astype(str).tolist()
    poll_meta = (
        frame.groupby("poll_id", as_index=False)
        .agg(
            field_date=("field_date", "max"),
            institute=("institute", "first"),
            collection_mode=("collection_mode", "first"),
            target_population=("target_population", "first"),
            sample_size=("sample_size", "max"),
            margin_error=("margin_error", "max"),
            scope=("scope", "first"),
            uf=("uf", "first"),
            undecided_share=("undecided_share", "max"),
        )
        .sort_values(["field_date", "poll_id"])
        .reset_index(drop=True)
    )
    pivot = frame.pivot_table(index="poll_id", columns="candidate_id", values="share", aggfunc="mean")
    pivot = pivot.reindex(index=poll_meta["poll_id"], columns=candidate_ids)
    if pivot.isna().any().any():
        incomplete = pivot.index[pivot.isna().any(axis=1)].tolist()
        raise ValueError(f"Pesquisas sem todos os candidatos: {incomplete[:5]}")

    shares = np.clip(pivot.to_numpy(dtype=float), 0.01, None)
    shares = shares / shares.sum(axis=1, keepdims=True)
    reference = shares[:, [-1]]
    y = np.log(shares[:, :-1] / reference)
    undecided = np.clip(poll_meta["undecided_share"].to_numpy(dtype=float) / 100.0, 0.001, 0.95)
    undecided_logit = np.log(undecided / (1 - undecided))[:, None]
    return poll_meta, candidate_ids, candidate_names, y, undecided_logit


def _build_design(
    meta: pd.DataFrame,
    as_of_date: date,
    state_ids: Iterable[str],
) -> tuple[np.ndarray, list[tuple[str, str]], np.ndarray]:
    columns: list[tuple[str, str]] = [("intercept", "intercept"), ("time", "trend")]
    levels = {
        "institute": sorted(meta["institute"].astype(str).unique()),
        "collection_mode": sorted(meta["collection_mode"].astype(str).unique()),
        "target_population": sorted(meta["target_population"].astype(str).unique()),
        "uf": sorted(
            set(str(x) for x in state_ids if str(x) != "BR")
            | set(meta.loc[meta["uf"] != "BR", "uf"].astype(str))
        ),
    }
    for group, values in levels.items():
        columns.extend((group, value) for value in values)

    x = np.zeros((len(meta), len(columns)), dtype=float)
    x[:, 0] = 1.0
    x[:, 1] = np.array([(row - as_of_date).days / 14.0 for row in meta["field_date"]], dtype=float)
    index = {item: i for i, item in enumerate(columns)}
    for row_index, row in meta.iterrows():
        for group in ("institute", "collection_mode", "target_population"):
            key = (group, str(row[group]))
            if key in index:
                x[row_index, index[key]] = 1.0
        uf_key = ("uf", str(row["uf"]))
        if str(row["uf"]) != "BR" and uf_key in index:
            x[row_index, index[uf_key]] = 1.0

    prior_sd = np.empty(len(columns), dtype=float)
    for i, (group, _) in enumerate(columns):
        prior_sd[i] = {
            "intercept": 4.0,
            "time": 0.35,
            "institute": 0.40,
            "collection_mode": 0.25,
            "target_population": 0.25,
            "uf": 0.70,
        }[group]
    return x, columns, 1.0 / np.square(prior_sd)


def _target_design(columns: list[tuple[str, str]], state_ids: list[str]) -> np.ndarray:
    targets = ["BR", *state_ids]
    x = np.zeros((len(targets), len(columns)), dtype=float)
    x[:, 0] = 1.0
    index = {item: i for i, item in enumerate(columns)}
    for row, uf in enumerate(targets):
        key = ("uf", uf)
        if uf != "BR" and key in index:
            x[row, index[key]] = 1.0
    return x


def _observation_covariance(
    meta: pd.DataFrame,
    base_weights: np.ndarray,
    reliability: dict[str, float],
    calibration: PollsterCalibration | None,
) -> np.ndarray:
    multipliers = meta["institute"].astype(str).map(reliability).to_numpy(dtype=float)
    variances = 1.0 / np.clip(base_weights * multipliers, 1e-5, 1e6)
    covariance = np.diag(variances)
    institutes = meta["institute"].astype(str).tolist()
    dates = meta["field_date"].tolist()
    ufs = meta["uf"].astype(str).tolist()
    for i in range(len(meta)):
        for j in range(i + 1, len(meta)):
            days = abs((dates[i] - dates[j]).days)
            if days > 42:
                continue
            if calibration is not None:
                rho = calibration.pair_correlation(institutes[i], institutes[j])
            else:
                rho = 0.35 if institutes[i] == institutes[j] else 0.08
            temporal_decay = math.exp(-days / 18.0)
            geography_factor = 1.0 if ufs[i] == ufs[j] else (0.65 if "BR" in {ufs[i], ufs[j]} else 0.30)
            effective_rho = float(np.clip(rho * temporal_decay * geography_factor, -0.20, 0.75))
            value = effective_rho * math.sqrt(variances[i] * variances[j])
            covariance[i, j] = covariance[j, i] = value
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    values = np.clip(values, 1e-8, None)
    return vectors @ np.diag(values) @ vectors.T


def _fit_matrix_normal(
    x: np.ndarray,
    y: np.ndarray,
    base_weights: np.ndarray,
    prior_precision: np.ndarray,
    meta: pd.DataFrame,
    calibration: PollsterCalibration | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    reliability = {str(k): 1.0 for k in meta["institute"].astype(str).unique()}
    beta = np.zeros((x.shape[1], y.shape[1]), dtype=float)
    a_inv = np.eye(x.shape[1])
    residual = y.copy()
    for _ in range(4):
        observation_covariance = _observation_covariance(meta, base_weights, reliability, calibration)
        observation_precision = np.linalg.pinv(observation_covariance)
        coefficient_precision = x.T @ observation_precision @ x + np.diag(prior_precision)
        a_inv = np.linalg.pinv(coefficient_precision)
        beta = a_inv @ x.T @ observation_precision @ y
        residual = y - x @ beta
        median_var = float(np.median(np.square(residual).mean(axis=1)) + 1e-6)
        updated: dict[str, float] = {}
        for institute, idx in meta.groupby("institute").groups.items():
            values = residual[np.array(list(idx), dtype=int)]
            n = max(values.size, 1)
            if calibration is not None:
                prior_var = calibration.institute_variance.get(str(institute), calibration.global_variance)
            else:
                prior_var = median_var
            posterior_var = (8.0 * prior_var + float(np.square(values).sum())) / (8.0 + n)
            updated[str(institute)] = float(np.clip(median_var / max(posterior_var, 1e-8), 0.20, 4.0))
        reliability = updated

    weights = np.clip(
        base_weights * meta["institute"].astype(str).map(reliability).to_numpy(dtype=float),
        1e-5,
        1e6,
    )
    centered = residual - np.average(residual, axis=0, weights=weights)
    covariance = (centered * weights[:, None]).T @ centered / max(weights.sum(), 1.0)
    covariance = np.atleast_2d(covariance)
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.70 * covariance + 0.30 * diagonal + np.eye(covariance.shape[0]) * 1e-5
    if calibration is not None and calibration.residual_correlation.shape == covariance.shape:
        std = np.sqrt(np.clip(np.diag(covariance), 1e-10, None))
        historical_covariance = np.outer(std, std) * calibration.residual_correlation
        covariance = 0.75 * covariance + 0.25 * historical_covariance
        covariance = (covariance + covariance.T) / 2 + np.eye(covariance.shape[0]) * 1e-6
    return beta, a_inv, covariance, reliability


def _apply_state_priors(
    alr_draws: np.ndarray,
    state_ids: list[str],
    candidate_ids: list[str],
    priors: pd.DataFrame | None,
    state_poll_counts: dict[str, int],
) -> np.ndarray:
    if priors is None or priors.empty:
        return alr_draws
    frame = priors.copy()
    for uf_index, uf in enumerate(state_ids, start=1):
        group = frame[frame["uf"].astype(str) == uf]
        if group.empty:
            continue
        shares = group.set_index("candidate_id").reindex(candidate_ids)["prior_share"].to_numpy(dtype=float)
        if np.isnan(shares).any() or shares.sum() <= 0:
            continue
        shares = np.clip(shares / shares.sum(), 1e-5, None)
        prior_alr = np.log(shares[:-1] / shares[-1])
        strength = float(group["prior_strength"].mean()) if "prior_strength" in group else 3.0
        observed = float(state_poll_counts.get(uf, 0))
        data_weight = observed / (observed + strength)
        alr_draws[:, uf_index, :] = data_weight * alr_draws[:, uf_index, :] + (1.0 - data_weight) * prior_alr
    return alr_draws


def fit_hierarchical_poll_model(
    polls: pd.DataFrame,
    as_of_date: date,
    *,
    state_priors: pd.DataFrame | None = None,
    calibration: PollsterCalibration | None = None,
    n_draws: int = 8_000,
    seed: int = 42,
    half_life_days: float = 24.0,
) -> PollPosterior:
    meta, candidate_ids, candidate_names, y, undecided_y = _prepare_poll_matrix(polls, as_of_date)
    if len(candidate_ids) < 2:
        raise ValueError("O modelo requer pelo menos dois candidatos.")

    state_ids = sorted(
        set(meta.loc[meta["uf"] != "BR", "uf"].astype(str))
        | (set(state_priors["uf"].astype(str)) if state_priors is not None and not state_priors.empty else set())
    )
    x, columns, prior_precision = _build_design(meta, as_of_date, state_ids)
    age = np.array([(as_of_date - value).days for value in meta["field_date"]], dtype=float)
    recency = np.exp(-math.log(2) * np.maximum(age, 0) / half_life_days)
    sampling_sigma = np.maximum(meta["margin_error"].to_numpy(dtype=float) / 1.96 / 10.0, 0.08)
    # The reported sampling margin already scales approximately as 1/sqrt(n).
    # Adding an explicit sample-size multiplier would count the same information twice.
    base_weights = recency / np.square(sampling_sigma)
    if calibration:
        calibration_penalty = np.array([
            1.0 / max(calibration.prior_variance(r.institute, r.collection_mode, r.target_population), 0.05)
            for r in meta.itertuples(index=False)
        ])
        calibration_penalty /= np.median(calibration_penalty)
        base_weights *= np.clip(calibration_penalty, 0.25, 4.0)

    beta, a_inv, covariance, reliability = _fit_matrix_normal(
        x, y, base_weights, prior_precision, meta, calibration,
    )
    undecided_beta, undecided_a_inv, undecided_cov, _ = _fit_matrix_normal(
        x, undecided_y, base_weights, prior_precision, meta, calibration,
    )

    rng = np.random.default_rng(seed)
    target_x = _target_design(columns, state_ids)
    p, d = beta.shape
    l_a = _safe_cholesky(a_inv)
    l_sigma = _safe_cholesky(covariance)
    z = rng.normal(size=(n_draws, p, d))
    beta_draws = beta[None, :, :] + np.einsum("pq,nqd,dr->npr", l_a, z, l_sigma.T)
    alr_targets = np.einsum("sp,npd->nsd", target_x, beta_draws)
    state_counts = meta.loc[meta["uf"] != "BR"].groupby("uf")["poll_id"].nunique().to_dict()
    alr_targets = _apply_state_priors(alr_targets, state_ids, candidate_ids, state_priors, state_counts)
    support_targets = _softmax_alr(alr_targets)

    l_ua = _safe_cholesky(undecided_a_inv)
    l_us = _safe_cholesky(undecided_cov)
    uz = rng.normal(size=(n_draws, undecided_beta.shape[0], 1))
    undecided_beta_draws = undecided_beta[None, :, :] + np.einsum("pq,nqd,dr->npr", l_ua, uz, l_us.T)
    undecided_targets = _sigmoid(np.einsum("sp,npd->nsd", target_x, undecided_beta_draws)[..., 0]) * 100.0

    national_draws = support_targets[:, 0, :]
    state_draws = support_targets[:, 1:, :] if state_ids else national_draws[:, None, :]
    undecided_state_draws = undecided_targets[:, 1:] if state_ids else undecided_targets[:, [0]]
    effective_states = state_ids or ["BR"]

    current_x = target_x[0].copy()
    previous_x = current_x.copy()
    previous_x[1] = -1.0
    current_mean_support = _softmax_alr((current_x @ beta)[None, :])[0]
    previous_mean_support = _softmax_alr((previous_x @ beta)[None, :])[0]
    trend_14d = current_mean_support - previous_mean_support

    national_records: list[dict[str, object]] = []
    winners = np.argmax(national_draws, axis=1)
    for idx, (candidate_id, candidate_name) in enumerate(zip(candidate_ids, candidate_names)):
        values = national_draws[:, idx]
        lower = float(np.quantile(values, 0.05))
        upper = float(np.quantile(values, 0.95))
        mean = float(values.mean())
        median = float(np.median(values))
        momentum = float(trend_14d[idx])
        national_records.append({
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            # Backward-compatible operational feature names.
            "poll_mean": mean,
            "poll_lower": lower,
            "poll_upper": upper,
            "poll_uncertainty": float(values.std(ddof=1)),
            "poll_trend_14d": momentum,
            "poll_count": int(meta["poll_id"].nunique()),
            # Research-oriented aliases used by the historical evaluation layer.
            "mean_share": mean,
            "median_share": median,
            "lower": lower,
            "upper": upper,
            "win_probability": float(np.mean(winners == idx)),
            "momentum": momentum,
        })
    national_summary = pd.DataFrame(national_records).sort_values("poll_mean", ascending=False).reset_index(drop=True)

    state_rows: list[dict[str, object]] = []
    for state_index, uf in enumerate(effective_states):
        draws = state_draws[:, state_index, :]
        state_winners = np.argmax(draws, axis=1)
        for candidate_index, (candidate_id, candidate_name) in enumerate(zip(candidate_ids, candidate_names)):
            values = draws[:, candidate_index]
            lower = float(np.quantile(values, 0.05))
            upper = float(np.quantile(values, 0.95))
            mean = float(values.mean())
            state_rows.append({
                "uf": uf,
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "poll_mean": mean,
                "poll_lower": lower,
                "poll_upper": upper,
                "undecided_mean": float(undecided_state_draws[:, state_index].mean()),
                "mean_share": mean,
                "median_share": float(np.median(values)),
                "lower": lower,
                "upper": upper,
                "win_probability": float(np.mean(state_winners == candidate_index)),
            })

    institute_rows: list[dict[str, object]] = []
    for institute, multiplier in sorted(reliability.items()):
        count = int(meta.loc[meta["institute"].astype(str) == institute, "poll_id"].nunique())
        prior_score = calibration.quality_score(institute) if calibration else 1.0
        institute_rows.append({
            "institute": institute,
            "posterior_precision_multiplier": float(multiplier),
            "historical_quality_score": float(prior_score),
            "poll_count": count,
            "reliability_weight": float(multiplier),
        })

    correlation = _covariance_to_correlation(covariance)
    diagnostics = {
        "model": "closed_form_matrix_normal_hierarchical_bayes",
        "candidate_dimensions": len(candidate_ids) - 1,
        "design_parameters": len(columns),
        "posterior_draws": n_draws,
        "state_count": len(effective_states),
        "n_polls": int(meta["poll_id"].nunique()),
        "n_rows": int(len(meta)),
        "uses_external_institute_quality": False,
        "hierarchical_effects": ["institute", "collection_mode", "target_population", "uf"],
        "undecided_modeled": True,
        "correlated_candidate_error": True,
        "correlated_institute_error": True,
        "weighting": "recency_times_inverse_reported_sampling_variance",
    }
    return PollPosterior(
        candidate_ids=candidate_ids,
        candidate_names=candidate_names,
        state_ids=effective_states,
        national_draws=national_draws,
        state_draws=state_draws,
        undecided_state_draws=undecided_state_draws,
        national_summary=national_summary,
        state_summary=pd.DataFrame(state_rows),
        institute_reliability=pd.DataFrame(institute_rows),
        residual_covariance=covariance,
        residual_correlation=correlation,
        diagnostics=diagnostics,
    )
