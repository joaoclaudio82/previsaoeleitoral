from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from app.ml.transfer import TransferModel
from app.services.hierarchical_polls import PollPosterior


@dataclass
class SimulationOutput:
    candidates: pd.DataFrame
    states: pd.DataFrame
    diagnostics: dict


def _fallback_pair_probabilities(
    fundamentals: pd.DataFrame,
    n_simulations: int,
    seed: int,
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    indexed = fundamentals.set_index("candidate_id", drop=False)
    ids = indexed.index.astype(str).tolist()
    rng = np.random.default_rng(seed)
    result: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for source_id in ids:
        source = indexed.loc[source_id]
        for a_id in ids:
            for b_id in ids:
                if len({source_id, a_id, b_id}) < 3:
                    continue
                a = indexed.loc[a_id]
                b = indexed.loc[b_id]
                distance_a = abs(float(source.get("ideology_score", 0)) - float(a.get("ideology_score", 0)))
                distance_b = abs(float(source.get("ideology_score", 0)) - float(b.get("ideology_score", 0)))
                score = 1.8 * (distance_b - distance_a) + 0.025 * (
                    float(b.get("rejection", 50)) - float(a.get("rejection", 50))
                )
                mean_pa = 1 / (1 + np.exp(-score))
                p_a = np.clip(rng.normal(mean_pa, 0.08, n_simulations), 0.02, 0.98)
                mean_abs = np.clip(0.06 + 0.0015 * float(source.get("rejection", 50)) + 0.04 * min(distance_a, distance_b), 0.03, 0.35)
                p_abs = np.clip(rng.normal(mean_abs, 0.025, n_simulations), 0.0, 0.60)
                result[(source_id, a_id, b_id)] = (p_a, p_abs)
    return result


def simulate_election(
    posterior: PollPosterior,
    fundamentals: pd.DataFrame,
    turnout_draws: np.ndarray,
    registered_voters: np.ndarray,
    n_simulations: int,
    seed: int,
    transfer_model: TransferModel | None = None,
) -> SimulationOutput:
    ids = posterior.candidate_ids
    names = posterior.candidate_names
    k = len(ids)
    s = len(posterior.state_ids)
    if turnout_draws.shape != (n_simulations, s):
        raise ValueError(f"turnout_draws deve ter formato {(n_simulations, s)}, recebeu {turnout_draws.shape}")
    if len(registered_voters) != s:
        raise ValueError("registered_voters não corresponde aos estados do posterior.")

    fundamentals_indexed = fundamentals.set_index("candidate_id").reindex(ids)
    if fundamentals_indexed.isna().any().any():
        missing = fundamentals_indexed.index[fundamentals_indexed.isna().any(axis=1)].tolist()
        raise ValueError(f"Fundamentos incompletos para: {missing}")
    ml_probs = fundamentals_indexed["ml_probability"].to_numpy(dtype=float)
    rejection = fundamentals_indexed["rejection"].to_numpy(dtype=float)
    late_score = fundamentals_indexed.get("late_decider_score", pd.Series(0.0, index=ids)).to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, posterior.state_draws.shape[0], size=n_simulations)
    state_support = posterior.state_draws[draw_indices].copy()
    undecided = np.clip(posterior.undecided_state_draws[draw_indices] / 100.0, 0.0, 0.60)

    prior_factor = np.exp((ml_probs - 1.0 / k) * 0.20)
    state_support *= prior_factor[None, None, :]
    state_support = state_support / state_support.sum(axis=2, keepdims=True) * 100.0

    propensity = np.clip(state_support / 100.0, 1e-5, None)
    propensity *= np.exp(late_score[None, None, :] + (50.0 - rejection)[None, None, :] / 140.0)
    propensity /= propensity.sum(axis=2, keepdims=True)
    concentration = propensity * 70.0 + 0.5
    gamma = rng.gamma(shape=concentration, scale=1.0)
    undecided_allocation = gamma / gamma.sum(axis=2, keepdims=True)
    final_state_share = state_support / 100.0 * (1.0 - undecided[:, :, None]) + undecided_allocation * undecided[:, :, None]
    final_state_share /= final_state_share.sum(axis=2, keepdims=True)

    electorate = np.asarray(registered_voters, dtype=float)[None, :, None]
    turnout = np.clip(turnout_draws, 0.35, 0.95)[:, :, None]
    state_votes = final_state_share * electorate * turnout
    national_votes = state_votes.sum(axis=1)
    first_round = national_votes / national_votes.sum(axis=1, keepdims=True) * 100.0

    leaders = np.argmax(first_round, axis=1)
    lead_counts = np.bincount(leaders, minlength=k)
    pair_draws = (
        transfer_model.precompute_pair_draws(fundamentals, n_simulations, seed + 17)
        if transfer_model is not None
        else _fallback_pair_probabilities(fundamentals, n_simulations, seed + 17)
    )
    winner_counts = np.zeros(k, dtype=int)
    runoff_abstained_votes = np.zeros(n_simulations, dtype=float)

    for simulation_index, row_votes in enumerate(national_votes):
        top2 = np.argsort(row_votes)[-2:]
        finalist_a, finalist_b = int(top2[1]), int(top2[0])
        runoff_a = float(row_votes[finalist_a])
        runoff_b = float(row_votes[finalist_b])
        for eliminated_idx in np.argsort(row_votes)[:-2]:
            source_id = ids[int(eliminated_idx)]
            key = (source_id, ids[finalist_a], ids[finalist_b])
            p_a_values, p_abs_values = pair_draws[key]
            p_a = float(p_a_values[simulation_index])
            p_abstain = float(p_abs_values[simulation_index])
            transferable = float(row_votes[int(eliminated_idx)]) * (1.0 - p_abstain)
            runoff_abstained_votes[simulation_index] += float(row_votes[int(eliminated_idx)]) * p_abstain
            runoff_a += transferable * p_a
            runoff_b += transferable * (1.0 - p_a)
        winner_counts[finalist_a if runoff_a >= runoff_b else finalist_b] += 1

    candidate_rows = []
    national_summary = posterior.national_summary.set_index("candidate_id")
    for index, (candidate_id, candidate_name) in enumerate(zip(ids, names)):
        shares = first_round[:, index]
        poll_row = national_summary.loc[candidate_id]
        candidate_rows.append({
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "poll_mean": float(poll_row["poll_mean"]),
            "poll_lower": float(poll_row["poll_lower"]),
            "poll_upper": float(poll_row["poll_upper"]),
            "poll_uncertainty": float(poll_row["poll_uncertainty"]),
            "ml_probability": float(ml_probs[index]),
            "first_round_lead_probability": float(lead_counts[index] / n_simulations),
            "win_probability": float(winner_counts[index] / n_simulations),
            "expected_first_round_share": float(shares.mean()),
            "expected_first_round_share_low": float(np.quantile(shares, 0.05)),
            "expected_first_round_share_high": float(np.quantile(shares, 0.95)),
        })
    candidate_frame = pd.DataFrame(candidate_rows).sort_values("win_probability", ascending=False).reset_index(drop=True)

    state_rows: list[dict] = []
    for state_index, uf in enumerate(posterior.state_ids):
        shares = final_state_share[:, state_index, :] * 100.0
        state_leaders = np.argmax(shares, axis=1)
        counts = np.bincount(state_leaders, minlength=k)
        leader_index = int(np.argmax(counts))
        expected = shares.mean(axis=0)
        turnout_values = turnout_draws[:, state_index]
        state_rows.append({
            "uf": uf,
            "expected_turnout": float(turnout_values.mean()),
            "turnout_low": float(np.quantile(turnout_values, 0.05)),
            "turnout_high": float(np.quantile(turnout_values, 0.95)),
            "leading_candidate": names[leader_index],
            "leader_probability": float(counts[leader_index] / n_simulations),
            "expected_shares": {candidate_id: float(expected[i]) for i, candidate_id in enumerate(ids)},
        })
    state_frame = pd.DataFrame(state_rows)
    total_first_votes = national_votes.sum(axis=1)
    diagnostics = {
        "turnout_national_mean": float(total_first_votes.mean() / np.sum(registered_voters)),
        "turnout_national_low": float(np.quantile(total_first_votes / np.sum(registered_voters), 0.05)),
        "turnout_national_high": float(np.quantile(total_first_votes / np.sum(registered_voters), 0.95)),
        "expected_runoff_additional_abstention_share": float(np.mean(runoff_abstained_votes / np.maximum(total_first_votes, 1.0))),
        "transfer_matrix_supplied": False,
        "transfer_model": "learned_bayesian_binomial" if transfer_model is not None else "transparent_ideology_rejection_fallback",
        "undecided_allocated_probabilistically": True,
    }
    return SimulationOutput(candidate_frame, state_frame, diagnostics)
