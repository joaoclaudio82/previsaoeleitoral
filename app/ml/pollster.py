from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


@dataclass
class PollsterCalibration:
    institute_variance: dict[str, float]
    mode_variance: dict[str, float]
    population_variance: dict[str, float]
    global_variance: float
    residual_correlation: np.ndarray
    institute_error_correlation: dict[str, dict[str, float]]
    version: str = "0.2.0"

    @classmethod
    def fit(cls, frame: pd.DataFrame, prior_df: float = 8.0) -> "PollsterCalibration":
        required = {
            "election_id", "poll_id", "institute", "collection_mode", "target_population",
            "candidate_id", "poll_share", "result_share",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Colunas ausentes na calibração de institutos: {sorted(missing)}")
        work = frame.copy()
        work["error"] = work["poll_share"].astype(float) - work["result_share"].astype(float)
        global_var = float(max(work["error"].var(ddof=1), 0.25))

        def posterior_variance(column: str) -> dict[str, float]:
            result: dict[str, float] = {}
            for key, group in work.groupby(column):
                squared = float(np.square(group["error"].to_numpy(dtype=float)).sum())
                n = max(len(group), 1)
                result[str(key)] = float((prior_df * global_var + squared) / (prior_df + n))
            return result

        poll_wide = work.pivot_table(index="poll_id", columns="candidate_id", values="poll_share", aggfunc="mean")
        result_wide = work.pivot_table(index="poll_id", columns="candidate_id", values="result_share", aggfunc="mean")
        common_index = poll_wide.index.intersection(result_wide.index)
        common_columns = sorted(set(poll_wide.columns).intersection(result_wide.columns))
        poll_wide = poll_wide.reindex(index=common_index, columns=common_columns).dropna(axis=0, how="any")
        result_wide = result_wide.reindex(index=poll_wide.index, columns=common_columns).dropna(axis=0, how="any")
        if len(poll_wide) >= 3 and len(common_columns) >= 2:
            poll_values = np.clip(poll_wide.to_numpy(dtype=float), 0.05, None)
            result_values = np.clip(result_wide.to_numpy(dtype=float), 0.05, None)
            poll_alr = np.log(poll_values[:, :-1] / poll_values[:, [-1]])
            result_alr = np.log(result_values[:, :-1] / result_values[:, [-1]])
            alr_error = poll_alr - result_alr
            corr = np.atleast_2d(np.corrcoef(alr_error, rowvar=False))
            corr = np.nan_to_num(corr, nan=0.0)
            corr = 0.75 * corr + 0.25 * np.eye(corr.shape[0])
        else:
            size = max(int(work["candidate_id"].nunique()) - 1, 1)
            corr = np.eye(size)

        # Correlation of polling errors across institutes, estimated only from
        # elections/candidates observed by both institutes and shrunk to zero.
        institute_mean = (
            work.groupby(["election_id", "institute", "candidate_id"], as_index=False)["error"]
            .mean()
        )
        institutes = sorted(institute_mean["institute"].astype(str).unique())
        institute_corr: dict[str, dict[str, float]] = {name: {} for name in institutes}
        for left in institutes:
            for right in institutes:
                if left == right:
                    institute_corr[left][right] = 0.55
                    continue
                left_frame = institute_mean[institute_mean["institute"].astype(str) == left][
                    ["election_id", "candidate_id", "error"]
                ].rename(columns={"error": "left_error"})
                right_frame = institute_mean[institute_mean["institute"].astype(str) == right][
                    ["election_id", "candidate_id", "error"]
                ].rename(columns={"error": "right_error"})
                paired = left_frame.merge(right_frame, on=["election_id", "candidate_id"], how="inner")
                if len(paired) < 8 or paired["left_error"].std() == 0 or paired["right_error"].std() == 0:
                    estimate = 0.08
                else:
                    raw = float(np.corrcoef(paired["left_error"], paired["right_error"])[0, 1])
                    shrink = len(paired) / (len(paired) + 30.0)
                    estimate = float(np.clip(raw * shrink, -0.15, 0.65))
                institute_corr[left][right] = estimate

        return cls(
            institute_variance=posterior_variance("institute"),
            mode_variance=posterior_variance("collection_mode"),
            population_variance=posterior_variance("target_population"),
            global_variance=global_var,
            residual_correlation=corr,
            institute_error_correlation=institute_corr,
        )

    def prior_variance(self, institute: str, mode: str, population: str) -> float:
        values = [
            self.institute_variance.get(str(institute), self.global_variance),
            self.mode_variance.get(str(mode), self.global_variance),
            self.population_variance.get(str(population), self.global_variance),
        ]
        return float(np.mean(values))

    def quality_score(self, institute: str) -> float:
        variance = self.institute_variance.get(str(institute), self.global_variance)
        return float(np.clip(np.sqrt(self.global_variance / max(variance, 1e-9)), 0.25, 2.5))

    def pair_correlation(self, left: str, right: str) -> float:
        if left == right:
            return 0.55
        return float(self.institute_error_correlation.get(str(left), {}).get(str(right), 0.08))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PollsterCalibration":
        return joblib.load(path)
