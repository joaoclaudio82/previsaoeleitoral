# Beyond Predicting the Winner: Temporal Backtesting and Probabilistic Calibration of a Hierarchical Bayesian Framework for Brazilian Presidential Elections

## Abstract

Election forecasting is often evaluated by whether a model identifies the eventual winner, even though a scientifically useful forecast must also quantify uncertainty, remain temporally valid, and provide calibrated probabilities. We present ElectionAI, a reproducible hierarchical Bayesian framework for Brazilian presidential election forecasting. The framework combines pre-election polling with partial pooling across polling organizations, collection modes, target populations, and federative units; explicitly represents undecided voters; propagates correlated polling error; and supports state-level priors derived exclusively from previous elections. We evaluate the framework retrospectively on the 2014, 2018, and 2022 Brazilian presidential elections using rolling historical snapshots at multiple horizons before Election Day. The evaluation protocol prevents look-ahead bias by excluding information released after each forecast cutoff and separates exploratory snapshots from scoreable snapshots when the candidate slate changed during the campaign. ElectionAI is compared with four polling baselines and with a state-prior ablation. Performance is assessed using vote-share mean absolute error, Brier score, log loss, expected calibration error, and posterior interval coverage. {{ABSTRACT_RESULTS}} The study emphasizes probability calibration and temporal validity rather than winner classification alone and provides an open, auditable workflow for retrospective electoral forecasting in a multiparty federal setting.

**Keywords:** election forecasting; Bayesian hierarchical models; opinion polls; probabilistic calibration; backtesting; Brazil; uncertainty quantification.

## 1. Introduction

Pre-election polls provide noisy and heterogeneous measurements of latent electoral support. Polling organizations differ in sampling frames, collection modes, likely-voter assumptions, fieldwork timing, and systematic house effects. Moreover, polling errors are not independent: surveys conducted close in time can share common shocks, polling organizations can exhibit correlated errors, and state-level deviations can move together during national electoral swings. These features make simple polling averages attractive as transparent baselines but insufficient as complete probabilistic forecasts.

A second challenge concerns evaluation. A model evaluated with information that became available after the historical forecast date can appear substantially more accurate than it would have been in real time. Election forecasting therefore requires a strict temporal protocol in which each retrospective prediction is reconstructed using only information available at that cutoff. This issue is especially important in Brazil, where presidential campaigns may experience candidate substitutions, changes in the effective candidate slate, and substantial late-campaign movement.

ElectionAI was developed to address these issues in a unified research framework. The system represents candidate support in log-ratio space, estimates partially pooled effects for pollsters and survey design characteristics, models correlated error, produces posterior distributions at national and federative-unit levels, and supports turnout and second-round simulation layers. The present study focuses on the first-round historical forecasting component because it can be evaluated against official election outcomes with a transparent temporal design.

The contribution of this paper is fourfold. First, we define a reproducible temporal backtesting protocol for Brazilian presidential elections using snapshots from D-180 to D-1. Second, we evaluate forecasts probabilistically rather than only by winner accuracy. Third, we compare the hierarchical model with simple and recency-weighted polling baselines. Fourth, we quantify the contribution of historical state priors through an explicit ablation experiment.

The main research questions are:

1. Does a hierarchical Bayesian polling model improve vote-share accuracy over simple polling averages across historical Brazilian presidential elections?
2. How does predictive performance evolve as Election Day approaches?
3. Are the model's winner probabilities and posterior intervals empirically calibrated?
4. Do priors derived from previous state-level election results improve federative-unit forecasts relative to neutral state priors?

## 2. Related Work

Election forecasting has a long tradition of combining polls, historical election results, and political or economic fundamentals. Bayesian approaches are particularly suitable because they provide a coherent mechanism for pooling heterogeneous information and propagating uncertainty. Lock and Gelman proposed Bayesian integration of state polls and election forecasts, emphasizing the distinction between national swings and relative state positions. Stoetzer et al. developed a dynamic Bayesian model for multiparty elections that combines polls and fundamentals in log-ratio space and evaluates predictive uncertainty. Chen, Garnett, and Montgomery later presented a dynamic hierarchical Bayesian approach with Gaussian-process priors for U.S. Senate elections and emphasized both accuracy and interval coverage.

Operational polling averages also illustrate the importance of recency, sample-size adjustments, population definitions, house effects, and correlated error. These considerations motivate the baseline hierarchy used in this study: latest poll, unweighted mean, recency-weighted mean, and recency-plus-sample-size weighting. The ElectionAI model extends these baselines by representing multiple sources of survey heterogeneity in a common probabilistic model.

A central distinction in this work is between prediction accuracy and probability calibration. A forecast that repeatedly assigns 90% probability to events occurring only 60% of the time is overconfident even if it often identifies the correct winner. We therefore report proper scoring rules and empirical coverage alongside vote-share errors.

## 3. Data

### 3.1 Official election results

Official presidential election results are obtained from the Tribunal Superior Eleitoral (TSE) Open Data Portal. The historical pipeline normalizes schema differences across election vintages and aggregates valid candidate votes by election, round, and federative unit. The 2010 election is used as historical information for subsequent state priors, while 2014, 2018, and 2022 are used as scoreable retrospective forecast elections.

### 3.2 Historical polling data

The TSE PesqEle datasets provide authoritative records of registered surveys, methodology, contracting entities, and related documentation. However, the registry is not treated as a complete machine-readable time series of all published candidate percentages. Historical candidate vote-intention percentages are therefore ingested through a separate adapter for public polling tables. Source provenance is retained, and the research pipeline is designed so that these secondary tables can be replaced progressively with primary pollster releases without changing the forecasting interface.

### 3.3 Temporal snapshots

For each election, forecasts are reconstructed at nominal horizons D-180, D-120, D-90, D-60, D-30, D-15, D-7, D-3, and D-1. At each cutoff, polls with field dates after the cutoff are excluded. Polls older than the configured historical window are also excluded. Candidate-slate consistency is enforced within each snapshot.

The 2014 and 2018 campaigns require special treatment because the effective candidate slate changed during the campaign. Snapshots before the stabilized slate are retained for exploratory analysis but excluded from proper scoring metrics. This prevents retrospective replacement of candidates from introducing look-ahead information.

## 4. Methods

### 4.1 Compositional representation

Let y_i denote the vector of reported support shares in poll i for K candidates. Because vote shares are compositional, they are normalized and transformed to additive log-ratio coordinates relative to a reference candidate. The transformed observations are modeled in an unconstrained Euclidean space and converted back to the simplex through the inverse log-ratio transformation.

### 4.2 Hierarchical polling model

The linear predictor includes an intercept, a temporal trend, and partially pooled effects associated with polling organization, collection mode, target population, and federative unit. Gaussian priors regularize these effects toward zero. Polls receive recency and effective-sample-size information through their observation precision.

The observation covariance matrix allows errors from nearby surveys to be correlated. Correlation is stronger for repeated observations from the same polling organization and decays with temporal distance. The framework also permits historical pollster calibration to inform prior variance and cross-pollster error correlation when sufficient historical information is available.

### 4.3 State priors

For federative units without direct contemporaneous polling, the model uses priors constructed strictly from elections completed before the forecast year. Matching follows a documented hierarchy: the same canonical candidate, the same party, historical national information, and finally a weak neutral prior. This ordering avoids using the outcome of the election being forecast.

The ablation experiment replaces these historical state priors with equal-share neutral priors while preserving the remaining hierarchical forecasting machinery. The difference in state-level performance isolates the empirical contribution of historical geographic information.

### 4.4 Baselines

We compare ElectionAI with four transparent polling baselines:

- **Latest poll:** only surveys with the latest field date at the cutoff receive weight.
- **Simple mean:** all eligible polls receive equal weight.
- **Recency weighted:** poll weights decay exponentially with age.
- **Sample + recency weighted:** recency weights are multiplied by the square root of relative sample size.

Uncertainty for each baseline is approximated from the weighted dispersion of poll vectors plus a sampling-variance floor, allowing probabilistic winner scores to be compared with the Bayesian model.

### 4.5 Evaluation metrics

Vote-share accuracy is evaluated using mean absolute error (MAE). Winner probabilities are evaluated with the Brier score and binary log loss. Calibration is summarized by expected calibration error (ECE), reliability bins, and an approximate calibration intercept and slope. Posterior uncertainty is assessed through empirical coverage of 90% predictive intervals.

Metrics are reported by election, forecast horizon, model, and geographic level. Because only three presidential elections are scoreable, probability-calibration results are interpreted as diagnostic rather than definitive; candidate-by-state observations provide additional repeated forecast instances but are not statistically independent.

## 5. Results

{{RESULTS_OVERVIEW}}

### 5.1 Comparison with polling baselines

{{BASELINE_RESULTS}}

### 5.2 Forecast horizon

{{HORIZON_RESULTS}}

### 5.3 Probabilistic calibration

{{CALIBRATION_RESULTS}}

### 5.4 State-prior ablation

{{ABLATION_RESULTS}}

## 6. Discussion

The retrospective design is intended to answer a stricter question than whether a model can fit past election outcomes. At every historical cutoff, the forecasting system is restricted to information that could have been known at that time. This distinction is essential because even subtle use of future polls, final candidate slates, or election-day turnout can materially understate real forecasting error.

The baseline comparison also provides an important guard against unnecessary model complexity. A hierarchical Bayesian system is scientifically useful only if its additional structure improves accuracy, calibration, uncertainty representation, or geographic generalization relative to transparent polling averages. If a simpler baseline performs similarly at a given horizon, that result should be reported rather than hidden.

State-level inference is particularly challenging. National polling contains limited information about heterogeneous regional shifts, while historical geographic voting patterns can be persistent. The state-prior ablation directly tests whether borrowing information from the previous election improves forecasts or merely adds inertia. Future versions should compare the current partial-pooling approach with spatial conditional autoregressive priors and demographic similarity structures.

Probability calibration should also be interpreted conservatively. Three presidential cycles provide few independent national election outcomes. State-by-candidate observations increase the number of scored instances but share common national shocks and therefore cannot be treated as independent Bernoulli trials. Future work should expand the historical archive and use hierarchical calibration models that explicitly represent dependence among states, candidates, and snapshots.

## 7. Limitations

The historical polling percentages currently rely partly on secondary public tables rather than a fully reconstructed archive of primary pollster releases. The TSE registry is used for authoritative survey registration and methodological metadata, while candidate percentages remain a distinct provenance layer. Candidate substitutions in 2014 and 2018 require curated scoring start dates. The present paper evaluates the first-round polling component and does not claim validated performance for the second-round transfer model or the operational 2026 forecast. Digital sentiment and search signals are intentionally excluded from the core historical comparison until their platform drift and manipulation sensitivity can be validated retrospectively.

The number of independent Brazilian presidential elections is small. Consequently, model ranking should be interpreted jointly across proper scoring rules, vote-share errors, coverage, forecast horizons, and state-level diagnostics rather than through a single aggregate statistic.

## 8. Conclusion

ElectionAI provides a reproducible framework for temporally valid probabilistic forecasting of Brazilian presidential elections. Its main methodological emphasis is not the production of a single predicted winner but the disciplined reconstruction of what could have been forecast at a given historical date, together with explicit uncertainty and calibration diagnostics. {{CONCLUSION_RESULTS}} The framework is designed to support continued replacement of secondary historical sources with primary polling archives, richer spatial models, turnout nowcasting, and independently reviewed release criteria before any operational forecast is presented as a prediction of a future election.

## Data and Reproducibility Statement

The software, data-ingestion code, temporal snapshot logic, evaluation scripts, and model-comparison workflow are maintained in the public ElectionAI repository. Large raw election files are not committed to version control; they are downloaded from their registered sources and verified through recorded provenance and hashes. Synthetic demonstration data remain explicitly separated from historical operational data and are blocked from publication as real forecasts.

## References

Chen, Y., Garnett, R., & Montgomery, J. M. (2023). Polls, Context, and Time: A Dynamic Hierarchical Bayesian Forecasting Model for US Senate Elections. *Political Analysis, 31*(1), 113–133. https://doi.org/10.1017/pan.2021.42

Lock, K., & Gelman, A. (2010). Bayesian Combination of State Polls and Election Forecasts. *Political Analysis, 18*(3), 337–348. https://doi.org/10.1093/pan/mpq002

Stoetzer, L. F., Neunhoeffer, M., Gschwend, T., Munzert, S., & Sternberg, S. (2019). Forecasting Elections in Multiparty Systems: A Bayesian Approach Combining Polls and Fundamentals. *Political Analysis, 27*(2), 255–262. https://doi.org/10.1017/pan.2018.49

Tribunal Superior Eleitoral. Portal de Dados Abertos do TSE. Historical election results and PesqEle survey registry datasets.
