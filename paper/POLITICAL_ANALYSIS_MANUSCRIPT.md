# Leakage-Safe Election Forecasting in Multiparty Federal Systems: A Compositional Hierarchical Bayesian Framework with Historical Geographic Leans

**João Cláudio Nunes Carvalho**  
Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE), Brazil  
Corresponding author: [insert institutional e-mail before submission]

## Abstract

Retrospective election forecasts are vulnerable to two forms of hidden optimism: information released after the forecast date and uncertainty estimates that describe only a fitted latent mean rather than a future election outcome. We develop a leakage-safe forecasting framework for multiparty federal elections and evaluate it on the 2014, 2018, and 2022 Brazilian presidential elections at four common pre-election cutoffs (D-15, D-7, D-3, and D-1). Candidate support is modeled compositionally in additive log-ratio space with regularized pollster and survey-design effects, temporally correlated polling error, election-day projection, and posterior-predictive innovations. Subnational forecasts combine the current national distribution with probabilistic historical state-versus-national leans estimated only from previous elections. National vote-share MAE is 4.64 percentage points, close to 4.70 for the best transparent polling baseline; national 90% interval coverage remains underdispersed at 39.6%. In a matched state-prior ablation, historical geographic leans reduce MAE from 9.12 to 6.10 points, improve 90% coverage from 57.1% to 89.4%, and reduce winner Brier score from 0.147 to 0.079. The results show where hierarchical structure adds value and where calibration remains unresolved.

**Keywords:** election forecasting; Bayesian hierarchical models; opinion polls; compositional data; probabilistic calibration; backtesting; Brazil.

## 1. Introduction

Election forecasting is unusually easy to evaluate badly. A model can identify an eventual winner while being overconfident, can fit historical elections using information that was unavailable in real time, or can report uncertainty around a fitted latent support trajectory without accounting for the error of forecasting the realized election. These distinctions matter whenever a forecast is interpreted probabilistically rather than as a descriptive polling average.

The problem is particularly visible in multiparty federal systems. Polling organizations differ in field dates, sample sizes, collection modes, target populations, and persistent house effects. Candidate support is compositional: increasing one candidate's share necessarily changes the remaining shares. The relevant state-level quantity is also not simply the previous election result. A state can remain systematically more favorable to a party than the country as a whole while the national electoral environment changes substantially. Finally, candidate sets can change during a campaign, creating a subtle retrospective leakage channel if an analyst evaluates an early forecast using a candidate slate known only later.

This article develops a forecasting and evaluation design around those problems. The empirical setting is the Brazilian presidential election, a two-round national contest conducted across 26 states and the Federal District. We reconstruct first-round forecasts for 2014, 2018, and 2022 using only information available at each historical cutoff. The scoreable design is deliberately compact: D-15, D-7, D-3, and D-1 are used in all three elections, yielding a common post-slate comparison rather than maximizing the number of retrospective snapshots.

The framework, implemented in the ElectionAI research pipeline, contributes four methodological elements. First, historical forecasts are generated from immutable, hashed inputs and date-restricted snapshots, making look-ahead violations auditable. Second, candidate shares are modeled jointly in additive log-ratio (ALR) coordinates rather than as independent univariate percentages. Third, subnational forecasts use a national-swing-plus-state-lean construction: previous-election geography is represented relative to the previous national composition and then applied to the current national forecast, with uncertainty propagated through a Dirichlet prior. Fourth, the retrospective evaluation distinguishes posterior uncertainty in latent support from posterior-predictive election uncertainty by adding a correlated residual innovation estimated only from polls available at the cutoff.

The study is designed as a methodological stress test rather than a demonstration that a complex model necessarily dominates simple polling averages. That distinction is important. At the national level, the final hierarchical specification only slightly improves vote-share mean absolute error (MAE) relative to the latest-poll baseline and produces worse winner-probability scores in this small historical sample. At the state level, however, the matched geographic-prior ablation shows a much larger gain. The model also remains underdispersed nationally. These mixed results are scientifically useful because they identify both the contribution of geographic structure and an unresolved uncertainty-calibration problem.

The analysis therefore addresses four questions. Does a leakage-safe hierarchical forecast improve national vote-share accuracy over transparent polling baselines? Does accuracy improve as Election Day approaches? Does posterior-predictive uncertainty achieve adequate empirical coverage? And, holding the rest of the forecasting machinery fixed, how much does a historical state-versus-national lean contribute to subnational accuracy and calibration?

## 2. Related work

Bayesian election forecasting provides a natural language for combining heterogeneous information and propagating uncertainty. Lock and Gelman (2010) emphasized a decomposition that is central to the present study: national vote share and the relative positions of states should be forecast separately. Their argument is especially relevant when national swings are large but geographic deviations from the national mean are more persistent. The state-lean construction below extends that logic to a multiparty composition rather than a two-party share.

Multiparty forecasting creates additional statistical constraints. Stoetzer et al. (2019) use a dynamic Bayesian framework to combine polls and fundamentals while respecting the joint nature of party support. More generally, the statistical analysis of compositions motivates log-ratio transformations because raw shares live on a simplex rather than in unconstrained Euclidean space (Aitchison 1982). The ALR representation used here makes it possible to fit correlated candidate movements while ensuring that transformed posterior draws map back to shares summing to 100%.

Dynamic hierarchical models also provide a precedent for integrating time, polls, and subnational structure. Chen, Garnett, and Montgomery (2023) show that a hierarchical model can achieve both accuracy and useful interval coverage when uncertainty is learned from historical elections and polls. Their results motivate an important diagnostic distinction in this article: narrow intervals are not evidence of model quality if they omit forecast error. A forecast distribution should be evaluated with proper scoring rules and empirical interval coverage, not merely by whether its posterior mean is close to the realized result (Gneiting and Raftery 2007).

The Brazilian setting adds two practical complications. First, historical polling data are distributed across election authorities, pollster releases, media archives, and secondary compilations. Second, candidate substitutions can occur late enough to make early polling scenarios incomparable with the final contest. A credible retrospective evaluation must therefore treat data lineage and candidate-slate timing as part of the statistical design rather than as preprocessing details.

## 3. Data and retrospective design

### 3.1 Elections and geographic outcomes

The scoreable sample contains the first rounds of the 2014, 2018, and 2022 Brazilian presidential elections. State-level first-round results for those elections are used as outcomes. The 2010 election is never scored; it enters only as information available before 2014 when constructing historical geographic priors. For each subsequent election, only an election completed before the forecast year can contribute to the state prior.

Minor first-round candidacies are represented as an `Others` category in the frozen research inputs. This preserves the vote simplex while keeping the candidate dimension comparable within a historical snapshot. Candidate-name canonicalization and party labels are versioned so that a prior can match, in order, the same candidate, the same party, a national fallback, or a neutral fallback.

### 3.2 Polling inputs and provenance

The research pipeline separates survey-registration metadata from published candidate percentages. The Tribunal Superior Eleitoral (TSE) is the authoritative source for Brazilian election results and registered-survey metadata. Candidate vote-intention percentages are stored in frozen research tables with row-level provenance. The 2022 table is based on the public Nexo Dados presidential polling archive. The current 2014 and 2018 frozen tables are derived from secondary historical public tabulations and retain their source URLs in every row.

This provenance choice is deliberately visible rather than abstracted away. It is also a limitation: before journal acceptance, the 2014 and 2018 percentages should be reconciled against the open Poder360 archive or primary pollster releases. Poder360 maintains a historical polling database extending back to 2000, which provides an independent route for source verification. Replacing a secondary table with a primary or independently curated archive does not require changing the model interface because ingestion and forecasting are separated.

Every frozen input used by the publication workflow is included in a manifest with a SHA-256 hash. The historical workflow materializes processed data from those files rather than downloading mutable web pages during the experiment. This makes the numerical results tied to a specific repository state and prevents later source edits from silently changing a retrospective forecast.

### 3.3 Leakage-safe snapshots

Let T be Election Day and let h denote the forecast horizon in days. The scoreable cutoffs are T-15, T-7, T-3, and T-1. For a cutoff t, only polls whose field date is no later than t are eligible. Polls older than the configured 90-day lookback are excluded. Within a snapshot, only polls matching the most recent candidate-set signature are retained, avoiding the combination of incompatible polling scenarios.

Earlier research code generated exploratory cutoffs as far as D-180. They are not used in the results reported here. Restricting the main design to the four common horizons prevents the 2018 candidate substitution and other campaign-specific slate changes from creating an asymmetric comparison. The resulting scored data contain 12 national election-horizon snapshots and 324 state-horizon units (3 elections x 4 horizons x 27 federative units).

## 4. Model

### 4.1 Compositional polling likelihood

Suppose poll i reports support vector s_i = (s_i1,...,s_iK) for K candidates or candidate groups. After clipping zero cells to a small positive value and renormalizing, we use candidate K as the reference and transform the first K-1 shares to ALR coordinates,

\[
y_{ik}=\log\left(\frac{s_{ik}}{s_{iK}}\right), \qquad k=1,\ldots,K-1. \tag{1}
\]

For poll i, the transformed vector is modeled as

\[
\mathbf y_i = \mathbf x_i^\top \mathbf B + \boldsymbol\varepsilon_i, \tag{2}
\]

where the design contains an intercept, linear campaign time, polling-organization indicators, collection-mode indicators, target-population indicators, and federative-unit indicators when state polls are present. Coefficients receive zero-centered Gaussian regularization. In the current implementation the prior standard deviations are fixed before retrospective scoring: 4.0 for the intercept, 0.35 for the temporal coefficient, 0.40 for institute effects, 0.25 for collection mode and target population, and 0.70 for state effects.

The implementation is a closed-form matrix-normal approximation rather than MCMC. This choice makes thousands of historical posterior draws inexpensive and reproducible while retaining a multivariate covariance across ALR candidate dimensions. Posterior draws are mapped back to the simplex with the inverse ALR transform.

### 4.2 Poll weighting and correlated observation error

A poll's sampling margin already scales approximately with 1/sqrt(n). To avoid counting sample size twice, the final historical model does not multiply inverse sampling variance by an additional sample-size factor. Its baseline precision is

\[
w_i = r_i / \sigma_i^2, \tag{3}
\]

where r_i is an exponential recency weight with a 24-day half-life and \sigma_i is derived from the reported margin of error (or an implied margin when the historical table lacks one).

Survey errors are allowed to correlate across polls. When external historical pollster calibration is unavailable, the prespecified correlation is 0.35 for two observations from the same institute and 0.08 otherwise. It decays exponentially with the number of days between surveys, with an 18-day scale, and is attenuated when geography differs. These constants are model hyperparameters rather than estimates from the three scored elections, which avoids tuning them directly on the reported outcomes but makes sensitivity analysis an important next step.

Institute-specific precision multipliers are updated iteratively from residual dispersion with shrinkage toward a common variance. Candidate residual covariance is estimated jointly in ALR space. The covariance matrix is then converted to a correlation matrix by standardization rather than by applying a correlation operator to the covariance entries themselves.

### 4.3 Projection from the cutoff to Election Day

A retrospective forecast at T-h is not evaluated as an estimate of public opinion at T-h. The fitted temporal coefficient is projected h days forward to T. In design-matrix units, the election target has time coordinate h/14 because campaign time is scaled in two-week units. Consequently the posterior mean can move between the final poll cutoff and the election instead of implicitly assuming that the latent state at the cutoff is already the election result.

The undecided share is modeled on the logit scale using the same design structure. In the present first-round historical evaluation, winner probabilities and vote-share metrics are computed from candidate support after normalization; the operational turnout and second-round transfer modules are not part of the claims evaluated here.

### 4.4 Historical state-versus-national leans

The key geographic construction separates national movement from state position. For the previous election, let q_s be the composition for state s and q_N the corresponding national composition after matching the current candidate set by canonical candidate, party, or fallback. Define the previous geographic lean in ALR space as

\[
\boldsymbol\ell_s = \operatorname{alr}(\mathbf q_s)-\operatorname{alr}(\mathbf q_N). \tag{4}
\]

At the current forecast date, let \boldsymbol\theta_N denote a draw from the national election-day ALR posterior. The state prior target is

\[
\boldsymbol\theta_s^{prior} = \boldsymbol\theta_N + \boldsymbol\ell_s. \tag{5}
\]

Thus the previous election contributes a *relative geographic displacement*, not an absolute vote share. A state that was more favorable than the country to a candidate can remain relatively favorable while the entire national distribution moves.

Geographic priors are probabilistic. Previous state compositions are drawn from a Dirichlet distribution centered on their historical shares, with concentration determined by the prespecified prior strength. This draw is converted to ALR coordinates before computing the lean. If n_s contemporaneous state polls exist, the model blends the direct state component and historical target with weight n_s/(n_s+kappa), where kappa is the prior strength. In the frozen national-poll archive used for the main retrospective experiment, the state predictions are driven primarily by the national forecast plus historical geographic lean; the paper therefore describes the state result as a geographic-prior evaluation rather than claiming extensive contemporaneous state-poll pooling.

### 4.5 Posterior-predictive election error

Parameter uncertainty alone produced severely underdispersed intervals in preliminary validation. The final evaluation therefore distinguishes latent-support uncertainty from prediction of a future realized election. For each posterior draw we sample a common national innovation

\[
\boldsymbol\eta^{(m)} \sim \mathcal N(\mathbf 0,\widehat{\boldsymbol\Sigma}_{res}), \tag{6}
\]

where \widehat{\boldsymbol\Sigma}_{res} is the ALR residual covariance estimated only from polls available at that snapshot. The innovation is added to the national ALR draw and shared across state draws before converting back to shares. Sharing the innovation represents a correlated national polling miss while state-prior uncertainty supplies additional geographic dispersion.

No election outcome is used to estimate the scale of this posterior-predictive innovation. This restriction is essential: inflating intervals until they cover known historical outcomes would convert calibration assessment into post-hoc fitting.

## 5. Evaluation

Four transparent national polling baselines are evaluated at the same cutoffs: the latest field-date poll, a simple mean, an exponentially recency-weighted mean, and a sample-plus-recency weighted mean. Baseline distributions use weighted poll dispersion with a sampling-variance floor. The primary national outcomes are candidate vote-share MAE, winner Brier score, binary log loss, expected calibration error (ECE), and empirical coverage of nominal 90% intervals.

For the geographic analysis, a symmetric ablation changes only the state prior. The historical-state-lean model and equal-share neutral-prior model use the same national polling data, Election-Day projection, posterior draw count, residual predictive-error mechanism, cutoffs, candidate set, and scoring code. This matched design is preferable to comparing the historical model with a separately parameterized state model because it isolates the contribution of geographic history.

The calibration intercept and slope are also reported descriptively. However, repeated candidates, states, and horizons share common election shocks; they are not independent Bernoulli observations. With only three independent presidential cycles, no single calibration statistic should be interpreted as a precise population estimate.

## 6. Results

### 6.1 National forecasting

Table 1 compares the final ElectionAI national forecast with the four transparent polling baselines. ElectionAI produces the lowest vote-share MAE, 4.64 percentage points, but the difference from the latest-poll baseline (4.70) is only 0.06 points. The evidence therefore does not support a claim of a large national accuracy advantage.

**Table 1. National forecast performance across 2014, 2018, and 2022, D-15/D-7/D-3/D-1**

| Model | MAE (p.p.) | Brier | Log loss | ECE | 90% coverage |
|---|---:|---:|---:|---:|---:|
| ElectionAI | 4.64 | 0.0071 | 0.0470 | 0.0429 | 39.6% |
| Latest poll | 4.70 | 0.0012 | 0.0079 | 0.0072 | 29.2% |
| Sample + recency weighted | 6.03 | 0.0061 | 0.0454 | 0.0420 | 54.2% |
| Recency weighted | 6.04 | 0.0059 | 0.0454 | 0.0421 | 54.2% |
| Simple mean | 6.27 | 0.0070 | 0.0521 | 0.0481 | 56.3% |

The latest-poll baseline has substantially lower Brier score and log loss than ElectionAI in this sample. This is a useful counterweight to the MAE result: propagating posterior-predictive uncertainty makes ElectionAI's winner probabilities less extreme, and the three observed winners happened to align strongly with the final polls. With so few independent elections, it would be inappropriate to infer that latest-poll probabilities are generally better calibrated from this comparison alone, but the observed proper scores must be reported as they stand.

National accuracy improves as Election Day approaches. ElectionAI's average MAE falls from 5.88 points at D-15 to 5.19 at D-7, 4.04 at D-3, and 3.45 at D-1. Election-specific national MAE averages 5.97 points in 2014, 4.55 in 2018, and 3.42 in 2022. This trajectory is consistent with the basic value of late campaign information, although three elections are too few to distinguish a structural horizon effect from election-specific difficulty.

### 6.2 National uncertainty remains underdispersed

Posterior-predictive error improves national interval coverage relative to latent-parameter uncertainty alone, but it does not solve the problem. Nominal 90% coverage is 39.6% overall: 43.8% in 2014, 31.3% in 2018, and 43.8% in 2022. By horizon, coverage rises from 25.0% at D-15 to 33.3% at D-7 and 50.0% at both D-3 and D-1. The direction is sensible, but the level is far below nominal.

The pooled calibration regression yields an intercept of -0.083 and slope of 0.765. A slope below one is consistent with probabilities that remain too extreme in some parts of the pooled forecast distribution, but the regression combines dependent state, candidate, and horizon observations and is therefore descriptive only.

This national undercoverage is the most important negative result in the paper. The residual covariance estimated within a snapshot does not fully represent election-day forecast error. Plausible missing sources include correlated methodological error shared across pollsters, uncertainty in the linear time extrapolation, late movement not represented by recent residuals, and errors introduced by changing turnout or vote validity between polling questions and the official valid-vote outcome. These components should be estimated from a larger archive rather than calibrated against the three outcomes used here.

### 6.3 Historical geography adds substantial state-level information

The strongest result is the matched state-prior ablation. Table 2 reports 324 state-horizon units. The historical state-lean specification achieves a 6.10-point vote-share MAE, compared with 9.12 under an equal-share neutral geographic prior. Historical geography therefore reduces MAE by 3.02 percentage points while holding the rest of the forecasting pipeline fixed.

**Table 2. Symmetric state-prior ablation, 324 state-horizon units**

| Geographic prior | MAE (p.p.) | Winner Brier | 90% coverage |
|---|---:|---:|---:|
| Historical state-versus-national lean | 6.10 | 0.079 | 89.4% |
| Equal-share neutral prior | 9.12 | 0.147 | 57.1% |

The improvement is not limited to point accuracy. Nominal 90% interval coverage is 89.4% with historical leans, close to the target, compared with 57.1% under the neutral prior. Winner Brier score falls from 0.147 to 0.079. Because both specifications receive the same national forecast and posterior-predictive national shock, the difference reflects information in persistent state-versus-national structure and its uncertainty representation.

The result also clarifies what the model is learning. It is not predicting state outcomes from a large panel of contemporaneous state polls. Instead, it transports current national support into the federal geography through historically informed relative positions. In settings where subnational polling is sparse, this decomposition can be more useful than treating the previous state result as an absolute prior.

## 7. Discussion

The empirical results support a narrower and more defensible claim than “hierarchical Bayes beats polling averages.” At the national level, ElectionAI and the latest poll are nearly tied in MAE, and the latest poll has the better winner-probability scores in the three-election sample. At the state level, however, a theoretically motivated geographic decomposition yields a large and consistent improvement over a matched neutral prior. The contribution is therefore primarily about leakage-safe evaluation, compositional state transport, and uncertainty diagnostics rather than a universal national forecasting advantage.

This distinction matters for political methodology. Complex election models are often judged after the election using whichever final specification appears most successful. The frozen-snapshot design makes retrospective choices inspectable: a forecast at D-7 cannot access D-3 polling, the prior for 2018 cannot use 2018 returns, and a candidate set cannot be silently rewritten using final-election knowledge. The same principle applies beyond election forecasting to any time-indexed political prediction problem in which features, labels, or entity definitions change over time.

The state-lean result also generalizes conceptually. Equation (5) separates a national composition that can move sharply from a geographic displacement that may be more persistent. Lock and Gelman's national-versus-state decomposition has this intuition in a two-party setting. The ALR formulation makes the displacement additive in an unconstrained coordinate system and therefore naturally extends it to more than two candidates. The construction is simple enough to audit, and a matched ablation shows that it contributes useful signal in Brazil.

The uncertainty result is equally important. State coverage near 90% might suggest that the framework is well calibrated, but national coverage shows that this conclusion would be premature. The state Dirichlet layer contributes substantial dispersion around geographic leans, while the national layer relies on residual covariance from available polls. The discrepancy reveals that the current model has a better representation of uncertainty in *where* support differs geographically than of uncertainty in *how far* the national election can deviate from the polling trajectory.

A next methodological step is therefore to estimate national forecast-error variance out of sample across a longer collection of elections rather than increase dispersion by hand. One possibility is a hierarchical error model that separates sampling variance, pollster house effects, common polling error, and late-campaign innovation. Another is to model the latent campaign trajectory as a stochastic process rather than a linear trend, allowing forecast variance to increase naturally with the distance from the final observed poll. The present pipeline is designed so that such alternatives can be compared under the same frozen-snapshot protocol.

## 8. Limitations

The first limitation is the number of independent national elections. Twelve national snapshots do not equal twelve independent electoral events; they are repeated forecasts of only three outcomes. State-horizon observations similarly share common national shocks. Standard errors based on treating all candidate-state rows as independent would therefore be misleading, and this article deliberately avoids claims of precise population-level superiority.

Second, the historical polling archive is not yet ideal for a final archival publication. The 2022 inputs have a dedicated public data source, but 2014 and 2018 currently rely on secondary historical tabulations with preserved URLs. The code and numerical lineage are reproducible, but source quality can be improved by reconciling each historical row against the Poder360 open archive or primary pollster releases. This should be completed before final journal submission and is more important than adding another modeling layer.

Third, several covariance hyperparameters are prespecified rather than estimated from a broad external training archive. They are not tuned against the three scored election outcomes, which limits direct overfitting, but robustness should be evaluated through sensitivity grids or prior predictive analysis.

Fourth, the paper validates only the first-round polling/geographic component. ElectionAI contains operational modules for turnout, runoff transfer, digital signals, and other features, but those components are excluded from the present empirical claims. In particular, this paper should not be read as a validated forecast of the 2026 Brazilian election.

Finally, the model uses a linear campaign trend between observed polling and Election Day. The monotonic reduction in MAE as the horizon shortens suggests that long-horizon trajectory uncertainty is important. A Gaussian process, local-level state-space model, or dynamic random walk is a natural comparison for future work.

## 9. Conclusion

A credible probabilistic election forecast requires more than selecting the correct winner. It requires a historical information set that can be reconstructed without leakage, a joint representation of competing vote shares, a principled mechanism for carrying national information into sparse subnational geographies, and predictive uncertainty that is evaluated rather than assumed.

Across the 2014, 2018, and 2022 Brazilian presidential elections, ElectionAI produces national vote-share MAE of 4.64 percentage points, essentially tied with the 4.70-point latest-poll baseline. National predictive intervals remain substantially underdispersed, with 39.6% coverage for nominal 90% intervals. The geographic result is stronger: under a matched ablation, historical state-versus-national leans reduce state-level MAE from 9.12 to 6.10 points, improve interval coverage from 57.1% to 89.4%, and reduce winner Brier score from 0.147 to 0.079.

The value of the framework is therefore not a claim of universal forecasting dominance. It is an auditable way to identify which modeling assumptions add predictive information and which uncertainty components remain missing. That combination of temporal discipline, compositional modeling, and explicit negative calibration results provides a basis for stronger comparative election forecasting as the historical archive expands.

## Funding

No external funding was used specifically for the analysis reported in this manuscript. [Confirm before submission.]

## Acknowledgements

The author thanks the maintainers of the public electoral and polling archives used to reconstruct the historical inputs. [Add project collaborators or institutional acknowledgements if appropriate before submission.]

## Data Availability Statement

Code, frozen research inputs, source provenance, SHA-256 manifests, and scripts that reproduce the historical backtests are available in the public ElectionAI repository at https://github.com/joaoclaudio82/previsaoeleitoral. The publication workflow runs the 2014, 2018, and 2022 backtests from frozen inputs and generates the reported tables and figures. Upon conditional acceptance, the exact replication snapshot should be archived in the Political Analysis Dataverse (and optionally Zenodo/Code Ocean) and this statement should be replaced with the resulting persistent citation.

## Conflicts of Interest

The author declares no conflicts of interest. [Confirm before submission.]

## References

Aitchison, John. 1982. “The Statistical Analysis of Compositional Data.” *Journal of the Royal Statistical Society: Series B (Methodological)* 44(2): 139–177.

Chen, Yehu, Roman Garnett, and Jacob M. Montgomery. 2023. “Polls, Context, and Time: A Dynamic Hierarchical Bayesian Forecasting Model for US Senate Elections.” *Political Analysis* 31(1): 113–133. https://doi.org/10.1017/pan.2021.42.

Gneiting, Tilmann, and Adrian E. Raftery. 2007. “Strictly Proper Scoring Rules, Prediction, and Estimation.” *Journal of the American Statistical Association* 102(477): 359–378. https://doi.org/10.1198/016214506000001437.

Lock, Kari, and Andrew Gelman. 2010. “Bayesian Combination of State Polls and Election Forecasts.” *Political Analysis* 18(3): 337–348. https://doi.org/10.1093/pan/mpq002.

Nexo Dados. 2022. “Pesquisas Presidenciais 2022.” Public data repository used for the frozen 2022 polling input. https://github.com/Nexo-Dados/pesquisas-presidenciais-2022.

Poder360. “Agregador de Pesquisas Eleitorais.” Historical archive of Brazilian electoral polling since 2000. https://www.poder360.com.br/wp-content/themes/poder/agregador/.

Stoetzer, Lukas F., Marcel Neunhoeffer, Thomas Gschwend, Simon Munzert, and Sebastian Sternberg. 2019. “Forecasting Elections in Multiparty Systems: A Bayesian Approach Combining Polls and Fundamentals.” *Political Analysis* 27(2): 255–262. https://doi.org/10.1017/pan.2018.49.

Tribunal Superior Eleitoral. “Portal de Dados Abertos do TSE.” Historical Brazilian election results and electoral survey registry data. https://dadosabertos.tse.jus.br/.
