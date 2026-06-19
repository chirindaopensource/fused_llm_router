# Beyond Benchmark Rankings: Information-Aware and Risk-Aware Optimization of Large Language Model Ensembles

**A Concept and Methodology Paper**



**CS Chirinda**



**18 June 2026**



## Abstract

Large language models (LLMs) are predominantly evaluated and deployed as standalone systems, under an implicit assumption that the model with the highest benchmark score is also the best component of a larger intelligent system. This paper argues that the assumption is unjustified. The value of an LLM-based system depends not only on the standalone capability of its constituent models, but also on the *informational novelty* of their outputs, the *structure of their disagreements*, and the degree to which their *failures are correlated*. Two models with near-identical benchmark scores can differ sharply in ensemble value: one may be redundant, while the other contributes complementary information by failing on different inputs.

To formalize this claim we propose a framework for **information-aware and risk-aware ensemble optimization**. The framework (i) separates *benchmark qualification* (an eligibility filter) from *ensemble selection* (the optimization objective); (ii) replaces scalar error with a *multidimensional error taxonomy* spanning mathematical, coding, factual, long-context, planning, and tool-use failures; and (iii) introduces the **Final Marginal Ensemble Contribution (FMEC)**, a conditional, deployment-oriented contribution score that integrates expected marginal utility, conditional information gain, and correlated-failure risk into a single optimization primitive. FMEC is estimated by Monte Carlo coalition sampling, outcome-based conditional-information estimation, and covariance-aware risk adjustment, and is then used inside a *constrained* optimization problem that selects ensemble members under cost, latency, and risk budgets.

We provide the full operational machinery required for implementation and replication: a precise model of inputs, processes, and outputs; four explicit algorithms (FMEC estimation, greedy constrained selection, nested-bootstrap validation, oracle-regret evaluation); a complete set of named equations; the raw and derived data structures with example schemas; a fully specified study-configuration dictionary including prompt templates and inference settings; and a falsifiable experimental and statistical protocol with pre-registered hypotheses, numeric decision thresholds, ablations, multiple-comparison control, contamination auditing, and drift instrumentation. The framework is explicitly designed to be either supported or rejected by data. The central, testable hypothesis is that benchmark rank is an incomplete proxy for deployment value, and that ensembles selected by FMEC outperform benchmark-selected ensembles on both absolute performance and quality-adjusted efficiency.

We are precise about reproducibility, separating two tiers that have different attainable ceilings. **Tier 1 (computational reproducibility)** — that the published pipeline returns identical numbers from identical inputs — is made bit-exact by releasing the frozen per-(model, task, replicate) *outcome tensor* as the primary artifact, together with a content hash, all random seeds, and every intermediate checkpoint, so that the entire chain downstream of generation is a deterministic, independently re-runnable function of released data. **Tier 2 (empirical replicability)** — that *re-collecting* model outputs reproduces the scientific conclusion — is bounded above by the fact that the constituent models are mutating, black-box, non-deterministic commercial endpoints; we do not pretend to remove this irreducible variability but instead *instrument and bound* it through a variance-components design that explicitly decomposes task-sampling, generation, and adjudication variance. The reproduction protocol is exercised by a fully specified **reference instantiation** (Section 12.1) on a small, frozen, hash-pinned pool, whose released artifacts make Tier-1 reproduction verifiable rather than merely promised.

**Keywords:** large language models, ensemble optimization, model orchestration, conditional information gain, cooperative game theory, covariance-aware risk, submodular optimization, decision theory, reproducibility.



## 1\. Introduction

### 1.1 The model-centric paradigm and its hidden assumption

The history of machine learning has largely been a search for stronger *individual* models. Progress is measured by benchmark improvements, reductions in inference cost, longer context, and stronger reasoning; public leaderboards and industrial evaluations reinforce a single narrative — *stronger models are progress*. The emergence of LLMs has amplified this view: models are ranked by reasoning, coding, scientific, and long-context benchmark suites, and deployment decisions are frequently driven by those rankings.

Implicit in the paradigm is a foundational assumption:

> \\\*\\\*The strongest model produces the strongest intelligent system.\\\*\\\*

Intuitive as it is, this assumption has received little scientific scrutiny. Many complex systems derive their effectiveness not from the superiority of individual components but from *interactions* among components — ensemble learning, distributed systems, biological ecosystems, financial portfolios, committee decision-making, and error-correcting codes all exhibit this property. In each, collective performance depends on component *complementarity*, not merely component quality. The same may hold for LLMs.

### 1.2 The model-orchestration problem

Consider two models with nearly identical benchmark performance. A benchmark-centric evaluation declares them interchangeable. From the standpoint of ensemble construction, however, they may be radically different. If they make the *same* mistakes, combining them yields little; if they fail on *different* inputs, combining them can materially improve the system. Benchmarks measure expected *individual* capability; ensemble optimization requires understanding *interaction* effects. Consequently, benchmark rankings alone are insufficient for constructing optimal systems. This motivates the central research question:

> \\\*\\\*How should LLM ensembles be constructed when model complementarity, informational novelty, and correlated failures are explicitly considered?\\\*\\\*

A widely circulated industry observation sharpens the stakes: a fused panel of mid-tier, lower-cost models has been reported to match a single frontier model's quality on certain task classes at roughly half the inference cost. Such reports are, until independently replicated, *task-specific* claims rather than general laws. But they articulate a market-structure hypothesis worth testing rigorously: that the strategic center of gravity in applied AI may be shifting from *owning the single best model* to *orchestrating many good-enough models efficiently*. This paper provides the statistical apparatus needed to evaluate that hypothesis on solid methodological ground rather than anecdote.

### 1.3 Limitations of existing ensemble approaches

Existing LLM ensemble methodologies fall into several families: (1) majority-voting and self-consistency systems; (2) routing architectures in which a controller directs queries to specific models; (3) debate-based and multi-agent reasoning systems; and (4) best-of-$N$ approaches. These methods often improve performance, but they share recurring limitations:

1. Model selection is frequently driven by benchmark rankings.
2. Many systems rely on manually selected, hand-tuned model combinations.
3. Diversity is approximated by simplistic disagreement metrics.
4. Little attention is paid to the statistical structure of model *failures*.
5. Most methods lack a formal measure of *conditional* model utility.

As a result, ensemble construction remains largely a heuristic engineering practice rather than a statistically grounded optimization problem.

### 1.4 Contributions

This paper makes the following contributions, each designed to be directly implementable and independently testable.

1. **A formal distinction between benchmark qualification and ensemble selection.** Benchmark scores are treated as a minimum-quality *filter*, not as the optimization objective.
2. **A multidimensional error taxonomy** for characterizing model failures, enabling covariance estimation at the granularity of failure *type*.
3. **An information-theoretic account of model complementarity** via conditional information gain, with an outcome-based operationalization that is estimable from correctness events rather than raw text.
4. **A Shapley-inspired estimator of conditional model utility** based on Monte Carlo coalition sampling, with explicit consistency, variance, and sample-complexity analysis.
5. **The Final Marginal Ensemble Contribution (FMEC)** — a risk-adjusted, information-aware contribution metric that rewards utility gain and informational novelty while penalizing correlated-failure risk.
6. **A covariance-aware risk framework** for quantifying correlated failures, with robust estimator alternatives (empirical, Ledoit–Wolf shrinkage, Bayesian, bootstrap).
7. **A constrained optimization formulation** of ensemble construction under cost, latency, and risk budgets, with a greedy algorithm and approximation analysis under (approximate) submodularity.
8. **A complete, reproducible experimental and statistical protocol**, with pre-registered hypotheses carrying *numeric* decision thresholds (non-inferiority margin, rank-discordance margin, recommendation-stability and inter-rater floors, negligible-effect floors), baselines, ablations, multiple-comparison control, robustness analysis over utility weights, contamination auditing, and oracle-regret evaluation.
9. **A two-tier reproducibility design** that separates computational reproducibility (made bit-exact by releasing the frozen outcome tensor, seeds, content hash, and intermediate checkpoints) from empirical replicability (bounded by black-box model non-determinism and *quantified* rather than assumed away), via a variance-components analysis that decomposes task-sampling, generation, and adjudication variance.
10. **A fully specified, hash-pinned reference instantiation** (Section 12.1) that exercises the entire protocol on a small frozen pool, converting the reproduction claim from a promise into a verifiable procedure.

Collectively, these contributions transform ensemble construction from a heuristic practice into a statistical decision problem. We emphasize at the outset that this is a *concept and methodology* paper: the experiments of Sections 11–13 are specified in full but are presented as a protocol to be executed, and **no empirical results are claimed here**. What the paper *does* commit to operationally is the reproducibility apparatus: the protocol is written against a hash-pinned reference instantiation (Section 12.1) whose released artifacts make Tier-1 (computational) reproduction bit-exact and verifiable, while Tier-2 (empirical) replicability is explicitly instrumented and bounded rather than assumed. Wherever a quantity can only be produced by executing the study (the tensor hash, contamination rates, realized variance components, realized inter-rater agreement), it is referenced as an artifact-to-be-populated, never as a fabricated value; wherever a quantity is a pre-registration *decision* (the decision thresholds of Section 11.13), a concrete value is fixed in advance, as pre-registration requires.



## 2\. Related Work

The framework sits at the intersection of ensemble learning, cooperative game theory, information theory, and risk-aware optimization.

**Ensemble learning.** Bagging, boosting, random forests, stacking, and committee machines establish that collections of weak or moderately strong predictors can outperform individual predictors when their errors are sufficiently diverse (Breiman, 1996, 2001; Freund and Schapire, 1997; Wolpert, 1992; Dietterich, 2000). The central lesson is that predictive quality is not determined by average component strength alone — diversity matters, and the relationship between ensemble diversity and ensemble accuracy has itself been studied extensively (Kuncheva and Whitaker, 2003). Classical settings, however, assume relatively simple, cheap predictors and do not address LLM-specific issues such as variable inference cost, latency, open-ended outputs, and heterogeneous task types. We adopt the diversity principle and extend it to heterogeneous, expensive, multi-task LLM ecosystems.

**LLM ensembles and orchestration.** Recent work explores majority voting, self-consistency (Wang et al., 2023), debate and multi-agent collaboration (Du et al., 2023), mixture-of-agents (Wang et al., 2024), output fusion (Jiang et al., 2023), and cost-aware routing (Chen et al., 2023; Ong et al., 2024). These demonstrate that multi-model systems can improve reliability, but they typically select constituents heuristically or by leaderboard rank, asking *which model is strongest* rather than *which model adds the most incremental value to a given system*. The present work formalizes that selection problem.

**Cooperative game theory.** Shapley values estimate the average marginal contribution of an agent across coalitions and provide principled fairness properties (Shapley, 1953); Shapley-based attribution has been applied to feature importance and model valuation (Lundberg and Lee, 2017). Most directly related to the present setting, Shapley values have been used to value and prune the members of a classifier ensemble (Rozemberczki and Sarkar, 2021) and to measure the contribution of component models to ensemble forecast accuracy (Kim, Ray and Reich, 2024); the marginal-ensemble-contribution term defined in Section 4 builds on this line of work. Our objective differs from contribution attribution alone: not attribution fairness but *deployment optimization*. We therefore integrate coalition-based contribution with utility, information gain, cost, latency, and correlated risk; Shapley-style estimation becomes a *component* of FMEC rather than its replacement.

**Information theory.** Mutual information, conditional entropy, and conditional information gain provide a principled vocabulary for novelty (Cover and Thomas, 2006). They distinguish a model that *repeats* existing information from one that *contributes new* information — the crucial distinction for ensemble selection. We use conditional mutual information to define the information-gain component of FMEC and give it an estimable, outcome-based form.

**Risk-aware optimization.** In finance and operations research, system quality depends not only on component strength but on the *correlation structure* of component risks (Markowitz, 1952). The mathematical insight — that correlated uncertainty must be penalized — transfers to ensemble design. We are explicit that LLMs are *not* financial assets; we borrow only the statistical structure (covariance-aware risk), and we use robust covariance estimation (Ledoit and Wolf, 2004) because naive estimation is unstable.

**Difference from prior work.** The framework differs from previous ensemble literature in three ways. It explicitly separates benchmark qualification from ensemble selection; it defines conditional contribution as a function of utility, information, *and* risk; and it formulates selection as a constrained optimization problem rather than a heuristic rule.



## 3\. Problem Setting and Notation

### 3.1 Model universe and ensembles

Let
$$
N = {M\_1, M\_2, \\ldots, M\_n}
$$
denote a finite candidate pool of language models that satisfy a minimum eligibility criterion (Section 3.4). Let $S \\subseteq N$ denote an *ensemble*. The ensemble may be realized through majority voting, weighted voting, judge-based synthesis, routing, or a hybrid mechanism. **The framework is agnostic to the exact fusion operator**, provided that ensemble utility can be measured consistently on held-out tasks (Section 8.3).

### 3.2 Model representation

Each model is represented by an attribute tuple
$$
M\_i = (\\mu\_i,, c\_i,, l\_i,, r\_i,, E\_i), \\tag{1}
$$
where $\\mu\_i$ is expected capability, $c\_i$ is inference cost, $l\_i$ is latency, $r\_i$ is robustness (e.g., historical non-error rate), and $E\_i$ is a multidimensional error representation. This representation enables simultaneous consideration of capability, efficiency, reliability, and failure behavior.

### 3.3 Multidimensional error taxonomy

A central weakness of prior methodologies is the treatment of error as a *scalar*. We instead define a multidimensional error vector. For model $M\_i$,
$$
E\_i = (E\_{\\mathrm{math}},, E\_{\\mathrm{code}},, E\_{\\mathrm{fact}},, E\_{\\mathrm{context}},, E\_{\\mathrm{plan}},, E\_{\\mathrm{tool}}) \\in \\mathbb{R}^m, \\tag{2}
$$
with coordinates denoting mathematical-reasoning errors, coding errors, factual hallucinations, long-context failures, planning failures, and tool-use failures, respectively. The implemented taxonomy is extensible; the experimental protocol (Section 11.5) uses an eight-category refinement that additionally separates *safety failures* and *refusal failures*. The taxonomy allows covariance structures to be estimated **separately across failure categories**, which is necessary because models exhibit different interaction patterns across domains.

### 3.4 Qualification versus selection

Benchmark performance serves a limited but important role: it determines *eligibility*, not *membership*. A model qualifies for the candidate pool if it clears a benchmark-percentile threshold (Section 11.2); thereafter, final selection depends on conditional contribution (FMEC), not benchmark rank. The evaluation suite used to *measure* ensemble utility must be **disjoint** from the qualification benchmarks; otherwise the procedure risks benchmark leakage and overfitting (Assumption A1, Section 6.1).

A consolidated notation table is provided in Appendix A; a complete index of named equations is provided in Appendix B.



## 4\. The Final Marginal Ensemble Contribution (FMEC)

### 4.1 Motivation

The central limitation of existing ensemble construction is the absence of a principled measure of *conditional* model utility. Benchmark scores measure individual capability; correlation measures behavioral similarity; mutual information measures informational dependence; Shapley values measure average coalition contribution. None of these, *individually*, answers the question central to ensemble optimization:

> \\\*\\\*Given an existing ensemble, how much useful value does a candidate model contribute?\\\*\\\*

FMEC is introduced as an *operational* measure of conditional ensemble value. It does not replace mutual information or Shapley values; it integrates multiple dimensions of ensemble utility into a single optimization primitive suitable for deployment-oriented construction. Specifically, FMEC captures (i) expected utility gain, (ii) informational novelty, (iii) correlated-failure risk, and (iv) cost- and latency-sensitive deployment considerations.

### 4.2 Ensemble utility function

The framework begins with a utility function $v(S)$ mapping an ensemble to a real-valued score. Unlike benchmark scores, utility explicitly incorporates deployment objectives:
$$
v(S) = \\alpha, Q(S) - \\beta, R(S) - \\gamma, C(S) - \\delta, L(S), \\tag{3}
$$
where $Q(S)$ is expected task performance, $R(S)$ is correlated-failure risk, $C(S)$ is inference cost, $L(S)$ is latency, and $\\alpha,\\beta,\\gamma,\\delta > 0$ are application-dependent preference weights. The formulation is intentionally modular and permits deployment-specific optimization: safety-critical environments emphasize risk reduction (large $\\beta$); consumer applications emphasize latency (large $\\delta$); research systems emphasize performance (large $\\alpha$). The framework thus **separates utility specification from optimization**.

#### 4.2.1 A closed form for $Q(S)$ and a validated fusion-simulation gap

The quality term $Q(S)$ must be evaluable for *arbitrary* coalitions $S$, because the coalition-sampling estimator of Section 4.4 evaluates $v(\\cdot)$ on thousands of subsets. Yet the primitive measurement collected in Stage 2 is *per-model* correctness $A\_{it}\\in{0,1}$, not coalition-level quality. The map from per-model correctness to coalition quality is therefore part of the definition of the objective and must be fixed, not left implicit. We make it explicit and distinguish two routes.

**(i) Simulated-fusion quality (the computed default).** For a deterministic fusion rule that depends only on members' correctness indicators, $Q(S)$ is a closed-form function of the released outcome tensor. For unweighted majority vote over an odd coalition,
$$
\\widehat{Q}*{\\mathrm{maj}}(S) ;=; \\frac{1}{|T|}\\sum*{t\\in T}\\mathbb{1}!\\left\[\\sum\_{i\\in S} A\_{it} ;>; \\tfrac{|S|}{2}\\right], \\tag{3a}
$$
with ties (even $|S|$) broken by the highest-reliability member $r\_i$ (Eq. 1); for the verifier-augmented operator (Section 8.3) the indicator is replaced by the deterministic verifier's pass/fail, which is likewise a function of stored outputs. Equation (3a) is exactly reproducible from the tensor and seeds and injects **no** LLM non-determinism into the objective, which is what permits the entire estimation chain to be Tier-1 reproducible (Section 13.3).

**(ii) Realized-fusion quality (the ground truth).** The deployed system does not vote on correctness flags; it runs the fusion operator $f$ — often a judge LLM — on the members' *outputs* and scores the synthesized answer. Realized quality $Q\_f(S)$ is the honest target, but evaluating it for every sampled coalition reintroduces both combinatorial cost ($K$ coalitions $\\times$ judge calls) and judge non-determinism into $v(\\cdot)$.

**The simulation gap, measured not assumed.** We compute $\\widehat{Q}\_{\\mathrm{maj}}$ (or the verifier variant) for all coalitions and *validate* it against $Q\_f$ on a random audit sample of coalitions (the reference instantiation uses $200$ coalitions stratified across $|S|$). We regress realized on simulated quality, $Q\_f(S)=\\beta\_0+\\beta\_1\\widehat{Q}(S)+u(S)$, and report the slope, $R^2$, and the residual standard deviation $\\widehat{\\sigma}\_u$ as a **bounded fusion-simulation error**; the protocol treats $\\widehat{\\sigma}\_u$ as a component of total uncertainty (Section 11.8) and flags any coalition-size stratum where $|Q\_f-\\widehat{Q}|$ exceeds a pre-registered tolerance. This keeps $v(\\cdot)$ deterministic and cheap while making explicit — and quantitatively bounding — the fidelity cost of the simulation. (The realized-fusion path remains the served object of Section 8.3; (3a) is the *evaluation-time* surrogate.)

### 4.3 Marginal Ensemble Contribution (MEC)

Given an ensemble $S$ and a candidate $M\_j \\notin S$, the instantaneous marginal contribution is
$$
\\mathrm{MEC}(M\_j \\mid S) = v(S \\cup {M\_j}) - v(S). \\tag{4}
$$
This is the immediate utility gained by adding $M\_j$ to $S$, analogous to marginal utility in economics and marginal contribution in cooperative game theory. Instantaneous MEC is necessary but insufficient: a model's contribution depends heavily on the coalition to which it is added, so a single coalition cannot provide a stable estimate.

### 4.4 Expected MEC and its Monte Carlo estimator

Let $\\mathcal{S}\_j$ denote a distribution over ensembles excluding $M\_j$. The expected contribution is
$$
\\overline{\\mathrm{MEC}}*j = \\mathbb{E}*{S \\sim \\mathcal{S}\_j}!\\left\[,v(S \\cup {M\_j}) - v(S),\\right]. \\tag{5}
$$
Because the exact expectation is computationally intractable for large pools, it is estimated by Monte Carlo sampling. Given $K$ sampled coalitions $S\_1, \\ldots, S\_K$ (each excluding $M\_j$), the estimator is
$$
\\widehat{\\mathrm{MEC}}*j = \\frac{1}{K}\\sum*{k=1}^{K}\\Big\[,v(S\_k \\cup {M\_j}) - v(S\_k),\\Big]. \\tag{6}
$$
This represents the average incremental utility attributable to $M\_j$ across many ensemble contexts. Its statistical properties (consistency, variance, sample complexity) are established in Section 6.

### 4.5 Relationship to Shapley values

The Shapley value of $M\_j$ is
$$
\\phi\_j = \\sum\_{S \\subseteq N \\setminus {M\_j}} \\frac{|S|!,(n-|S|-1)!}{n!},\\Big\[,v(S \\cup {M\_j}) - v(S),\\Big]. \\tag{7}
$$
Equation (6) is *Shapley-inspired* in that it averages marginal contribution over coalitions, but the objectives differ fundamentally. Shapley estimation answers *who deserves credit?* (attribution); FMEC answers *which model should be deployed?* (optimization). Because deployment utility includes cost, latency, and risk — considerations outside the classical Shapley formulation — Shapley estimation functions as a *component* of FMEC, not its replacement. The use of approximate Shapley values over an ensemble "game" to rank and select constituent models follows Rozemberczki and Sarkar (2021); the departure here is the deployment-oriented value function together with the information- and risk-aware terms introduced below. (Equation (6) corresponds to a particular coalition-sampling distribution $\\mathcal{S}\_j$; the uniform-over-permutations choice recovers an unbiased Shapley estimator, while task- or budget-conditioned choices target the deployment regime of interest.)

**Exact specification of the sampler and the estimand it targets.** The phrase "sample a coalition $S\_k\\subseteq N\\setminus{M\_j}$" is ambiguous between (a) drawing a coalition *size* uniformly on ${1,\\dots,n-1}$ then a subset of that size uniformly, and (b) drawing a subset uniformly over all $2^{,n-1}$ subsets; these are different distributions and yield different $\\widehat{\\mathrm{MEC}}*j$, and — as noted above — the choice fixes which estimand is being estimated. We remove the ambiguity by fixing the **permutation sampler** (Castro, Gómez and Tejada, 2009) as the default, which is the unbiased Shapley estimand and is the configuration named in `monte\\\_carlo.coalition\\\_sampler = "permutation"` (Section 10.1):
$$
\\widehat{\\phi}j ;=; \\frac{1}{K}\\sum{k=1}^{K}\\Big\[,v\\big(P^{(k)}*{\\prec j}\\cup{M\_j}\\big) - v\\big(P^{(k)}\_{\\prec j}\\big)\\Big],\\qquad P^{(k)}\\sim\\mathrm{Unif}(\\mathfrak{S}*n), \\tag{6a}
$$
where $P^{(k)}$ is a uniformly random permutation of the $n$ candidates and $P^{(k)}*{\\prec j}$ is the set of candidates preceding $M\_j$ in that permutation. Equation (6a) is an unbiased estimator of the Shapley value (Eq. 7) and is the recommended default precisely because its estimand is unambiguous and order-symmetric. When the deployment-conditioned estimand is wanted instead, the sampler is replaced by an explicit, documented mixing distribution over coalition sizes (`monte\\\_carlo.coalition\\\_sampler = "size\\\_mixture"` with the size-weight vector recorded in the config), and the paper reports *which* estimand a given table targets. The reference instantiation (Section 12.1) uses the permutation sampler so that the released $\\widehat{\\mathrm{MEC}}$ column is reproducible without knowledge of an undocumented size distribution.

### 4.6 Conditional information gain

**Why correlation is insufficient.** Suppose model $A$ achieves $95%$ accuracy and model $B$ achieves $60%$, with uncorrelated errors. A correlation-based criterion may reward $B$ as "diverse," yet $B$ contributes little *useful* information. Diversity alone does not imply value; *informative* diversity does. We therefore introduce conditional information gain.

Let $Y\_j$ be the output of $M\_j$, let $Y\_S$ be the outputs already available from ensemble $S$, and let $Z$ be the ground-truth outcome. The conditional information gain of $M\_j$ given $S$ is
$$
\\mathrm{IG}(M\_j \\mid S) = I(Y\_j; Z \\mid Y\_S), \\tag{8}
$$
where $I(,\\cdot,;,\\cdot \\mid ,\\cdot,)$ is conditional mutual information. This measures how much additional information about the correct answer $M\_j$ provides *after* conditioning on the current ensemble. Models providing novel information score higher; models repeating existing information score lower.

**Outcome-based operationalization.** Exact mutual-information estimation over free-form text is difficult and ill-posed. We resolve the ambiguity by defining information gain over *correctness events* rather than raw text. Let $Z$ be the task outcome and let $A\_j$ be the correctness indicator of $M\_j$ (and $A\_S$ the correctness indicators of the ensemble). Define the operational information gain
$$
\\mathrm{IG}(M\_j \\mid S) = I(A\_j; Z \\mid A\_S). \\tag{9}
$$
All variables in (9) are discrete, which makes estimation tractable and removes a major source of ambiguity. The protocol evaluates four estimators of (8)–(9) — agreement-based, entropy-reduction, judge-model, and distributional-output estimators — to reduce dependence on any single modeling assumption (Section 11.7, ablation A6).

**Estimator, conditioning reduction, and bias correction (made explicit).** Equation (9) conditions on the full correctness pattern $A\_S\\in{0,1}^{|S|}$, which has up to $2^{|S|}$ strata and is therefore *sparse* relative to the number of evaluation tasks; the naive plug-in conditional-MI estimator is upward-biased in exactly this regime, so leaving the estimator unspecified would make the IG term — the framework's distinctive content — irreproducible and optimistically inflated. We fix three things.

*Conditioning reduction.* We condition on the **correct-count sufficient statistic** $c\_S=\\sum\_{i\\in S}A\_{S,i}\\in{0,\\dots,|S|}$ rather than the full pattern, reducing the conditioning support from $2^{|S|}$ to $|S|+1$ cells (`information\\\_gain.conditioning = "correct\\\_count\\\_sufficient\\\_statistic"`). This is the natural reduction for a vote-style fusion in which what matters is *how many* members are correct, not their identities; the full-pattern conditioning is retained only as a sensitivity check on small coalitions.

*Plug-in estimator with explicit smoothing.* With $\\widehat{p}(\\cdot)$ the Laplace-smoothed empirical frequencies (additive constant $\\alpha=0.5$ per joint cell, `information\\\_gain.smoothing\\\_alpha`), the operational estimator is
$$
\\widehat{\\mathrm{IG}}(M\_j\\mid S);=;\\widehat{H}(Z\\mid c\_S);-;\\widehat{H}(Z\\mid A\_j,c\_S), \\tag{9a}
$$
each conditional entropy being the count-weighted average of per-stratum plug-in entropies on the smoothed counts. Equation (9a) is a deterministic function of the released outcome tensor.

*Bias correction.* Each entropy term carries a **Miller–Madow** correction (Miller, 1955), $\\widehat{H}*{\\mathrm{MM}}=\\widehat{H}*{\\mathrm{plug}}+\\frac{\\widehat{m}-1}{2N\_{\\text{stratum}}}$ with $\\widehat{m}$ the number of observed-support outcome categories, applied per stratum before averaging; this removes the leading $O(1/N)$ entropy bias that otherwise propagates into a spurious positive IG. The response-based vs. outcome-based comparison (Ablation A6) is run under the **same** corrected estimator on both signals, so the comparison isolates the signal rather than confounding it with estimator bias.

### 4.7 Correlated-failure risk

We define risk as expected correlated failure. Let $E\_i$ be the multidimensional error vector of $M\_i$ (Eq. 2) and let
$$
\\Sigma = \\mathrm{Cov}(E) \\tag{10}
$$
be the covariance matrix over model errors. For ensemble weights $w$, ensemble risk is the quadratic form
$$
R(w) = w^{\\top} \\Sigma, w. \\tag{11}
$$
High values indicate that models tend to fail *together*; low values indicate complementary failure structures. The marginal risk contribution of $M\_j$ under weights $w$ is
$$
\\mathrm{MRC}\_j = (\\Sigma w)\_j. \\tag{12}
$$
Naive sample covariance is notoriously unstable, and small estimation errors can produce dramatically different optimization outcomes. Covariance estimation is therefore treated as a *first-class* problem (Section 11.7, ablation A5): the protocol compares the empirical estimator, Ledoit–Wolf shrinkage (which regularizes toward a structured target), a Bayesian estimator (which incorporates prior uncertainty), and a bootstrap estimator (which quantifies estimation uncertainty).

### 4.8 The final FMEC score

Combining marginal utility, information gain, and risk yields the central ranking primitive:
$$
\\mathrm{FMEC}\_j = \\widehat{\\mathrm{MEC}}\_j + \\eta,\\mathrm{IG}\_j - \\lambda,\\mathrm{MRC}\_j, \\tag{13}
$$
where $\\eta,\\lambda > 0$ control the relative importance of informational novelty and correlated-risk penalty. FMEC **rewards** utility contribution and informational novelty while **penalizing** correlated-failure risk, directly addressing the principal weaknesses identified in Section 1.3.



## 5\. Ensemble Construction as Constrained Optimization

### 5.1 The constrained program

Let $w = (w\_1, \\ldots, w\_n)$ denote ensemble allocation weights. The objective is not to maximize benchmark performance but to maximize expected deployment utility while accounting for informational contribution, risk, cost, and latency. The optimization problem is
$$
\\max\_{w}; U(w) \\quad \\text{subject to} \\quad \\sum\_{i=1}^{n} w\_i = 1,;; w\_i \\ge 0,;; C(w) \\le C\_{\\max},;; L(w) \\le L\_{\\max},;; R(w) \\le R\_{\\max}, \\tag{14}
$$
where the utility aggregates per-model FMEC scores,
$$
U(w) = \\sum\_{i=1}^{n} w\_i, \\mathrm{FMEC}*i. \\tag{15}
$$
Expanding FMEC yields the explicit objective
$$
U(w) = \\sum*{i=1}^{n} w\_i \\left(\\widehat{\\mathrm{MEC}}\_i + \\eta,\\mathrm{IG}\_i - \\lambda,\\mathrm{MRC}\_i\\right). \\tag{16}
$$
The weights $w$ may represent voting weights, routing probabilities, or mixture coefficients, depending on the fusion mechanism. The combinatorial *set-selection* variant of (14) — choosing $S \\subseteq N$ rather than continuous $w$ — is treated in Section 7 (Algorithm 2) with an approximation guarantee under submodularity.

### 5.2 Efficient-frontier interpretation

The program (14) induces an *efficient frontier*. Each feasible ensemble is a point in a multi-objective space of (expected performance, information gain, cost, latency, correlated-failure risk). An ensemble is **Pareto efficient** if no alternative improves one objective without degrading another. Unlike benchmark rankings — which impose a *total* ordering on models — the efficient-frontier perspective recognizes that the optimal ensemble depends on deployment constraints, and that different applications may legitimately select different points on the frontier.

### 5.3 Quality-adjusted cost

A key derived efficiency metric is quality-adjusted cost,
$$
\\mathrm{QAC} = \\frac{C}{Q}, \\tag{17}
$$
with lower values indicating more efficient deployment. QAC separates *absolute performance* from *efficiency*: an ensemble that delivers $95%$ of a frontier model's quality at half its cost has a substantially lower (better) QAC, even though it is marginally weaker in absolute terms. This is the quantity through which the cost-structure hypothesis of Section 1.2 becomes measurable (see also H5, Section 11.1). The relative utility gain of an FMEC ensemble over a baseline is reported as
$$
\\mathrm{RUG} = \\frac{U(S\_{\\mathrm{FMEC}}) - U(S\_{\\mathrm{baseline}})}{\\lvert U(S\_{\\mathrm{baseline}})\\rvert}. \\tag{18}
$$



## 6\. Theoretical Analysis

We state assumptions explicitly, then establish identifiability, the estimator's statistical properties, two structural theorems, the central dominance proposition, submodularity-based tractability, a generalization bound, ranking consistency, and oracle-regret/recoverability results. Proofs are elementary and are included to make the framework auditable.

### 6.1 Assumptions

**A1 (Stable task distribution).** Evaluation pairs $(X, Y) \\sim \\mathcal{D}$ are drawn i.i.d., and the qualification and evaluation distributions agree in expectation, $\\mathcal{D}*{\\text{qual}} \\stackrel{d}{=} \\mathcal{D}*{\\text{eval}}$. Without distributional stability no statistical estimator can generalize.

**A2 (Finite utility variance).** $\\mathrm{Var}(v(S)) < \\infty$ for all $S$. This guarantees existence of Monte Carlo moments; without it, consistency results collapse.

**A3 (Observable error structure).** Each model has a measurable error vector $E\_i = (E\_{i1}, \\ldots, E\_{im})$, so that $\\Sigma$ in (10) can be estimated. Without observable errors, $\\Sigma$ is undefined.

**A4 (Non-degenerate diversity).** There exists at least one pair $M\_i, M\_j$ with $I(Y\_i; Z \\mid Y\_j) > 0$ — i.e., at least one model contributes information unavailable from another. This defines the regime in which ensemble optimization matters; absent it, ensembles collapse to benchmark rankings.

### 6.2 Identifiability

FMEC is *identifiable* if $v(S)$ and $\\mathrm{IG}(M\_j \\mid S)$ can be estimated from observable quantities. Under the framework: utility is observed through evaluation outcomes; information gain is estimated from correctness events and ground truth (Eq. 9); risk is estimated from observed error vectors (Eq. 10). Therefore FMEC is identifiable given sufficient evaluation data — a necessary condition for empirical estimation.

### 6.3 Consistency of the Monte Carlo estimator

**Proposition (Consistency).** Under A1–A2, with independent coalition samples and a stationary evaluation distribution, the estimator (6) satisfies
$$
\\widehat{\\mathrm{MEC}}\_j \\xrightarrow{;a.s.;} \\overline{\\mathrm{MEC}}\_j \\quad \\text{as } K \\to \\infty. \\tag{19}
$$
*Proof.* Each summand $v(S\_k \\cup {M\_j}) - v(S\_k)$ is an i.i.d. draw with mean $\\overline{\\mathrm{MEC}}\_j$ and finite variance (A2). The Strong Law of Large Numbers yields almost-sure convergence of the sample mean. $\\qquad\\blacksquare$

### 6.4 Variance and the cost–precision trade-off

Let $\\sigma\_j^2 = \\mathrm{Var}\\big(v(S \\cup {M\_j}) - v(S)\\big)$. Then
$$
\\mathrm{Var}\\big(\\widehat{\\mathrm{MEC}}\_j\\big) = \\frac{\\sigma\_j^2}{K}, \\tag{20}
$$
so estimation uncertainty decreases at the standard Monte Carlo rate $O(K^{-1/2})$. This is a direct, tunable trade-off between computational cost (number of coalition evaluations) and estimator precision.

### 6.5 Sample complexity

**Proposition (Sample complexity).** To estimate $\\overline{\\mathrm{MEC}}\_j$ within tolerance $\\epsilon$ with probability at least $1-\\delta$, it suffices (by a Hoeffding-style bound on bounded marginal contributions) to take
$$
K = O!\\left(\\frac{\\log(1/\\delta)}{\\epsilon^2}\\right) \\tag{21}
$$
coalition samples. Crucially, the requirement grows with desired *precision*, not *exponentially* with model count — which is what makes the approach feasible for moderate-to-large pools.

### 6.6 Theorem 1 (Information complementarity)

**Theorem 1.** For an ensemble $S$, if $\\mathrm{IG}(M\_j \\mid S) > 0$ then
$$
H(Z \\mid Y\_S, Y\_j) < H(Z \\mid Y\_S). \\tag{22}
$$
*Proof.* By definition $\\mathrm{IG}(M\_j \\mid S) = I(Y\_j; Z \\mid Y\_S)$, and the chain rule for conditional mutual information gives $I(Y\_j; Z \\mid Y\_S) = H(Z \\mid Y\_S) - H(Z \\mid Y\_S, Y\_j)$. If the left side is strictly positive, then $H(Z \\mid Y\_S, Y\_j) < H(Z \\mid Y\_S)$. $\\qquad\\blacksquare$

**Interpretation.** Positive conditional information gain *strictly* reduces residual uncertainty about the correct answer. FMEC therefore rewards models that reduce uncertainty, providing a theoretical justification for the information-gain term in (13).

### 6.7 Theorem 2 (Correlated failure)

**Theorem 2.** Suppose ensemble risk is $R(w) = w^{\\top}\\Sigma w$, and let $M\_i$ and $M\_j$ have identical expected utility. If
$$
\\mathrm{Cov}(E\_i, E\_S) < \\mathrm{Cov}(E\_j, E\_S), \\tag{23}
$$
then replacing $M\_j$ with $M\_i$ reduces ensemble risk and (holding performance fixed) increases utility.
*Proof.* The marginal risk contribution is $\\mathrm{MRC}\_k = (\\Sigma w)\_k$. Lower covariance with the incumbent ensemble implies $\\mathrm{MRC}\_i < \\mathrm{MRC}\_j$. Since expected utility $Q$ is held fixed and $U = (\\text{performance}) - \\lambda(\\text{risk})$ is decreasing in risk, the substitution increases $U$. $\\qquad\\blacksquare$

**Interpretation.** Covariance matters *independently* of benchmark performance: the risk penalty is theoretically motivated, not heuristic.

### 6.8 Proposition 1 (FMEC dominance)

This is the central scientific claim of the paper.

**Proposition 1.** Consider candidates $M\_a, M\_b$ with $\\mathrm{Benchmark}(M\_a) > \\mathrm{Benchmark}(M\_b)$ but $\\mathrm{FMEC}(M\_b) > \\mathrm{FMEC}(M\_a)$. Then there exist ensembles $S$ for which
$$
v(S \\cup {M\_b}) > v(S \\cup {M\_a}). \\tag{24}
$$
*Proof sketch.* By (13), a higher FMEC reflects a higher expected conditional utility contribution over the coalition distribution $\\mathcal{S}$; hence $\\mathbb{E}*{S}\\big\[v(S \\cup {M\_b})\\big] > \\mathbb{E}*{S}\\big\[v(S \\cup {M\_a})\\big]$, and a witnessing $S$ attaining the strict inequality exists. $\\qquad\\blacksquare$

**Interpretation.** The strongest model need not be the most valuable ensemble member — a direct attack on the benchmark-ranking paradigm.

### 6.9 Submodularity and tractable optimization

The set-selection form of (14) is combinatorial: exhaustive search over $2^n$ subsets is infeasible for even moderate $n$. We seek structure.

**Definition.** A set function $v$ is *submodular* if the marginal gain $v(S \\cup {M}) - v(S)$ is non-increasing in $S$ (diminishing returns), and *monotone* if $v(S \\cup {M}) \\ge v(S)$.

**Plausibility.** In ensemble systems, the first reasoning model adds substantial value while the tenth similar reasoning model adds less; information gain diminishes as redundancy grows; and the risk term is supermodular in the relevant regime. Hence the *performance-plus-information* component is approximately submodular.

**Proposition 2 (Approximate submodularity).** If (i) utility gains diminish, (ii) information gain diminishes, and (iii) risk penalties increase with ensemble size, then $U(S)$ is approximately submodular.

**Consequence and a precise statement of the guarantee.** For *monotone submodular* maximization under a **cardinality** constraint, the greedy algorithm achieves a $\\left(1 - \\tfrac{1}{e}\\right) \\approx 0.632$ approximation of the optimum (Nemhauser, Wolsey and Fisher, 1978). We emphasize, for rigor, that the program (14) contains **knapsack-type budget** constraints (cost, latency, risk), for which the guarantees are more delicate:

* Plain greedy under a single knapsack constraint guarantees $\\tfrac{1}{2}!\\left(1 - \\tfrac{1}{e}\\right)$ (Khuller, Moss and Naor, 1999; Leskovec et al., 2007).
* A *modified* greedy with partial enumeration recovers $\\left(1 - \\tfrac{1}{e}\\right)$ under a knapsack constraint (Sviridenko, 2004).
* When $U$ is only $\\gamma$-*weakly* submodular, greedy retains a $\\left(1 - e^{-\\gamma}\\right)$ guarantee (Das and Kempe, 2011).

Algorithm 2 (Section 7) implements the budget-feasible greedy; the protocol additionally reports oracle regret (Section 6.12 and Algorithm 4) for small pools to *measure* the realized gap rather than relying solely on worst-case bounds. This corrects a common informal overstatement that plain greedy attains $1 - 1/e$ under budget constraints.

### 6.10 Generalization bound

FMEC is estimated from finite tasks; will it generalize? Assume bounded utility $|v(S)| \\le B$. By Hoeffding's inequality (Hoeffding, 1963), for the task-averaged utility estimate $\\hat{v}(S)$ over $N$ evaluation tasks,
$$
\\Pr!\\big(,|\\hat{v}(S) - v(S)| > \\epsilon,\\big) \\le 2\\exp!\\left(-\\frac{2N\\epsilon^2}{B^2}\\right), \\tag{25}
$$
equivalently, with probability at least $1-\\delta$,
$$
|\\hat{v}(S) - v(S)| = O!\\left(\\sqrt{\\frac{\\log(1/\\delta)}{N}}\\right). \\tag{26}
$$
As evaluation tasks increase, estimated utility converges to true utility and FMEC rankings become increasingly reliable. (A uniform-convergence statement over a finite family of $\\binom{n}{k}$ candidate ensembles follows by a union bound, replacing $\\log(1/\\delta)$ with $\\log(,|,\\text{family},|/\\delta)$.)

### 6.11 Ranking consistency

Practitioners care which model is selected, not whether a score converges numerically. Let $\\mathrm{FMEC}\_i^{*}$ be the true contribution and $\\widehat{\\mathrm{FMEC}}\_i$ the estimate. The estimator is ranking-consistent if
$$
\\Pr!\\big(\\widehat{\\mathrm{FMEC}}\_i > \\widehat{\\mathrm{FMEC}}\_j\\big) \\to 1 \\quad \\text{whenever } \\mathrm{FMEC}\_i^{*} > \\mathrm{FMEC}\_j^{\*}. \\tag{27}
$$
**Proposition (Ranking consistency).** Under consistent MEC estimation (6.3), consistent information-gain estimation, and consistent covariance estimation, $\\widehat{\\mathrm{FMEC}}$ is ranking-consistent. *Sketch.* Each component converges; FMEC is a continuous combination of the components; the continuous-mapping theorem implies convergence of rankings whenever true scores are separated by a non-zero margin. This result is arguably more important than estimator consistency, because it governs *selection stability*.

### 6.12 Oracle ensemble and oracle regret

Define the oracle ensemble
$$
S^{*} = \\arg\\max\_{S} v(S). \\tag{28}
$$
$S^{*}$ is generally unobservable (it requires evaluating every subset) but provides a theoretical reference. The performance measure is *regret*,
$$
\\mathrm{Regret}(\\hat{S}) = v(S^{*}) - v(\\hat{S}), \\tag{29}
$$
with relative regret $\\mathrm{Regret}(\\hat S)/v(S^{*})$. An ideal procedure satisfies $\\mathrm{Regret}(\\hat{S}) \\to 0$. This decision-theoretic objective is substantially stronger than merely demonstrating improvement over a baseline: it measures *distance from the best feasible ensemble*.

### 6.13 Recoverability of the optimal ensemble

**Assumption (Approximate additivity).** Suppose utility is approximately additive,
$$
v(S) = \\sum\_{i \\in S} u\_i + \\varepsilon(S), \\tag{30}
$$
where $\\varepsilon(S)$ captures interaction effects.

**Theorem (Informal recoverability).** If $|\\varepsilon(S)|$ is uniformly bounded and sufficiently small, then selecting models in descending order of FMEC converges toward the oracle ensemble. **Interpretation.** FMEC works best when interaction effects *exist* (otherwise A4 fails and ensembles collapse to rankings) but *do not dominate* utility — a realistic regime for many ensemble systems.

### 6.14 Decision-theoretic reformulation

Ensemble construction is naturally a *statistical decision problem*. Let $\\mathcal{M} = {M\_1, \\ldots, M\_n}$ be the model universe and $\\mathcal{A}$ the set of feasible ensembles; each action $a \\in \\mathcal{A}$ selects an ensemble. The objective is
$$
a^{\*} = \\arg\\max\_{a \\in \\mathcal{A}} \\mathbb{E}\[U(a)]. \\tag{31}
$$
Under this framing, *benchmark ranking*, *random selection*, and *FMEC optimization* are all decision rules and become directly comparable on a common axis (expected utility, and regret against (28)).



## 7\. Algorithms

The methodology is operationalized by four algorithms. They are stated in `algorithmic`-style pseudocode (LaTeX) so that another researcher can implement them directly. Together they cover estimation (Algorithm 1), constrained selection (Algorithm 2), uncertainty quantification (Algorithm 3), and proximity-to-optimal evaluation (Algorithm 4).

### 7.1 Algorithm 1 — FMEC Estimation

**Purpose.** Estimate $\\mathrm{FMEC}\_i$ for every candidate $M\_i$ via coalition sampling (Eq. 6), outcome-based information gain (Eq. 9), and covariance-aware risk (Eqs. 10–12).

```latex
\\\\begin{algorithm}\\\[t]
\\\\caption{FMEC Estimation}
\\\\label{alg:fmec-estimation}
\\\\begin{algorithmic}\\\[1]
\\\\Require Candidate pool $N=\\\\{M\\\_1,\\\\dots,M\\\_n\\\\}$; evaluation task set $T=\\\\{t\\\_1,\\\\dots,t\\\_m\\\\}$;
         coalition count $K$; utility weights $\\\\theta=(\\\\alpha,\\\\beta,\\\\gamma,\\\\delta)$;
         FMEC weights $(\\\\eta,\\\\lambda)$; reference weights $w$; covariance estimator $\\\\widehat{\\\\Sigma}(\\\\cdot)$.
\\\\Ensure  FMEC scores $\\\\{\\\\mathrm{FMEC}\\\_i\\\\}\\\_{i=1}^{n}$.
\\\\State Evaluate every model on $T$; record correctness indicators $A\\\_{it}$ and error vectors $E\\\_{it}$.
\\\\Statex \\\\textbf{// Stage A: Monte Carlo marginal utility}
\\\\For{each model $M\\\_i \\\\in N$}
    \\\\State $\\\\mathrm{MEC\\\\\\\_sum} \\\\gets 0$
    \\\\For{$k = 1$ to $K$}
        \\\\State sample coalition $S\\\_k \\\\subseteq N \\\\setminus \\\\{M\\\_i\\\\}$ from $\\\\mathcal{S}\\\_i$
        \\\\State $U\\\_{\\\\text{with}} \\\\gets v(S\\\_k \\\\cup \\\\{M\\\_i\\\\})$ \\\\Comment{Eq.\\\~(3)}
        \\\\State $U\\\_{\\\\text{without}} \\\\gets v(S\\\_k)$
        \\\\State $\\\\mathrm{MEC\\\\\\\_sum} \\\\gets \\\\mathrm{MEC\\\\\\\_sum} + (U\\\_{\\\\text{with}} - U\\\_{\\\\text{without}})$
    \\\\EndFor
    \\\\State $\\\\widehat{\\\\mathrm{MEC}}\\\_i \\\\gets \\\\mathrm{MEC\\\\\\\_sum} / K$ \\\\Comment{Eq.\\\~(6)}
\\\\EndFor
\\\\Statex \\\\textbf{// Stage B: risk structure}
\\\\State $\\\\widehat{\\\\Sigma} \\\\gets \\\\widehat{\\\\Sigma}\\\\big(\\\\{E\\\_{it}\\\\}\\\\big)$ \\\\Comment{empirical / Ledoit--Wolf / Bayesian / bootstrap}
\\\\Statex \\\\textbf{// Stage C: information gain and final score}
\\\\For{each model $M\\\_i \\\\in N$}
    \\\\State $\\\\mathrm{IG}\\\_i \\\\gets \\\\widehat{I}(A\\\_i; Z \\\\mid A\\\_S)$ \\\\Comment{Eq.\\\~(9)}
    \\\\State $\\\\mathrm{MRC}\\\_i \\\\gets (\\\\widehat{\\\\Sigma} w)\\\_i$ \\\\Comment{Eq.\\\~(12)}
    \\\\State $\\\\mathrm{FMEC}\\\_i \\\\gets \\\\widehat{\\\\mathrm{MEC}}\\\_i + \\\\eta\\\\,\\\\mathrm{IG}\\\_i - \\\\lambda\\\\,\\\\mathrm{MRC}\\\_i$ \\\\Comment{Eq.\\\~(13)}
\\\\EndFor
\\\\State \\\\Return $\\\\{\\\\mathrm{FMEC}\\\_i\\\\}\\\_{i=1}^{n}$
\\\\end{algorithmic}
\\\\end{algorithm}
```

**Computational complexity.** Monte Carlo estimation requires $O(nK)$ utility evaluations; information estimation is $O(nm)$; covariance estimation is $O(n^2 m)$. The overall cost is
$$
O\\big(nK + n^2 m\\big), \\tag{32}
$$
i.e., *linear* in the coalition budget $K$ and at most *quadratic* in the model count $n$ — never exponential.

### 7.2 Algorithm 2 — Greedy Constrained FMEC Ensemble Selection

**Purpose.** Construct an ensemble $S$ that maximizes utility under the budget constraints of (14). The algorithm adds, at each step, the feasible candidate with the largest marginal utility gain $\\Delta U(M\_i \\mid S)$, halting when no feasible candidate remains.

```latex
\\\\begin{algorithm}\\\[t]
\\\\caption{Greedy Constrained FMEC Ensemble Selection}
\\\\label{alg:greedy-fmec}
\\\\begin{algorithmic}\\\[1]
\\\\Require Candidate pool $N$; FMEC scores $\\\\{\\\\mathrm{FMEC}\\\_i\\\\}$; budgets $C\\\_{\\\\max}, L\\\_{\\\\max}, R\\\_{\\\\max}$;
         utility function $U(\\\\cdot)$.
\\\\Ensure  Selected ensemble $S^{\\\*}$.
\\\\State $S \\\\gets \\\\varnothing$
\\\\While{feasible candidates remain}
    \\\\For{each candidate $M\\\_i \\\\notin S$}
        \\\\State compute marginal gain $\\\\Delta U(M\\\_i \\\\mid S) = U(S \\\\cup \\\\{M\\\_i\\\\}) - U(S)$
        \\\\If{$C(S \\\\cup \\\\{M\\\_i\\\\}) \\\\le C\\\_{\\\\max}$ \\\\textbf{and} $L(S \\\\cup \\\\{M\\\_i\\\\}) \\\\le L\\\_{\\\\max}$
            \\\\textbf{and} $R(S \\\\cup \\\\{M\\\_i\\\\}) \\\\le R\\\_{\\\\max}$}
            \\\\State $\\\\mathrm{Score}(M\\\_i) \\\\gets \\\\Delta U(M\\\_i \\\\mid S)$
        \\\\Else
            \\\\State $\\\\mathrm{Score}(M\\\_i) \\\\gets -\\\\infty$ \\\\Comment{infeasible}
        \\\\EndIf
    \\\\EndFor
    \\\\State $M^{\\\*} \\\\gets \\\\arg\\\\max\\\_{M\\\_i \\\\notin S} \\\\mathrm{Score}(M\\\_i)$
    \\\\If{$\\\\mathrm{Score}(M^{\\\*}) = -\\\\infty$ \\\\textbf{or} $\\\\Delta U(M^{\\\*}\\\\mid S) \\\\le 0$}
        \\\\State \\\\textbf{break}
    \\\\EndIf
    \\\\State $S \\\\gets S \\\\cup \\\\{M^{\\\*}\\\\}$
\\\\EndWhile
\\\\State \\\\Return $S^{\\\*} \\\\gets S$
\\\\end{algorithmic}
\\\\end{algorithm}
```

**Approximation guarantee.** If $U(\\cdot)$ is monotone submodular under a cardinality constraint, Algorithm 2 attains $\\left(1 - \\tfrac{1}{e}\\right)$ of the optimum (Nemhauser, Wolsey and Fisher, 1978). Under the knapsack-type budgets of (14), plain greedy attains $\\tfrac{1}{2}!\\left(1 - \\tfrac{1}{e}\\right)$, and the partial-enumeration variant attains $\\left(1 - \\tfrac{1}{e}\\right)$ (Sviridenko, 2004) (Section 6.9). Complexity is $O(n^2)$ marginal-utility evaluations in the worst case (at most $n$ rounds, each scanning at most $n$ candidates); with cached per-model FMEC and incremental risk updates, each $\\Delta U$ evaluation is $O(n)$ for the $(\\Sigma w)$ term.

### 7.3 Algorithm 3 — Nested (Cluster) Bootstrap FMEC Validation

**Purpose.** Quantify uncertainty *honestly*. The naive bootstrap that resamples only tasks captures task-sampling variance alone and is **silent on the two largest run-to-run sources** in a black-box-LLM study — generation variance (the same model, same task, same decoding, different draw) and adjudication variance (the judge's own non-determinism). Algorithm 3 therefore resamples at two levels — tasks (clusters) *and* replicates within a task — and resamples the judge on the adjudication subsample, so that the BCa intervals and the recommendation-stability statistics reflect total variance (Eq. 20a), not a fraction of it. It returns $95%$ BCa intervals for each $\\mathrm{FMEC}\_i$, the variance decomposition, per-model selection frequency, and the Recommendation-Stability Probability $\\mathrm{RSP}$ (Eq. 51).

```latex
\\\\begin{algorithm}\\\[t]
\\\\caption{Nested (Cluster) Bootstrap FMEC Validation}
\\\\label{alg:bootstrap-fmec}
\\\\begin{algorithmic}\\\[1]
\\\\Require Outcome tensor $\\\\{A\\\_{itr}, E\\\_{itr}\\\\}$ over tasks $T$ and replicates $r=1..R$;
         adjudication subsample $T\\\_{\\\\mathrm{adj}}\\\\subseteq T$ with judge replicates $r\\\_J=1..R\\\_J$;
         bootstrap count $B$; pipeline of Algorithms 1--2.
\\\\Ensure  $\\\\mathrm{CI}\\\_{95\\\\%}(\\\\mathrm{FMEC}\\\_i)$; variance components $(\\\\widehat\\\\sigma^2\\\_{\\\\mathrm{task}},\\\\widehat\\\\sigma^2\\\_{\\\\mathrm{gen}},\\\\widehat\\\\sigma^2\\\_{\\\\mathrm{adj}})$;
         per-model selection frequency; $\\\\mathrm{RSP}$.
\\\\For{$b = 1$ to $B$}
    \\\\State $T\\\_b \\\\gets$ resample tasks from $T$ with replacement \\\\Comment{outer: cluster level}
    \\\\For{each retained task $t \\\\in T\\\_b$}
        \\\\State $r\\\_t \\\\gets$ resample replicate indices from $\\\\{1,\\\\dots,R\\\\}$ with replacement \\\\Comment{inner: generation level}
        \\\\State assemble bootstrap rows $\\\\{A\\\_{i t r\\\_t}, E\\\_{i t r\\\_t}\\\\}$ for all models $i$
    \\\\EndFor
    \\\\For{each task $t \\\\in T\\\_b \\\\cap T\\\_{\\\\mathrm{adj}}$}
        \\\\State resample a judge replicate $r\\\_J$ \\\\Comment{adjudication level}
        \\\\State overwrite the correctness / error labels for $t$ with judge draw $r\\\_J$
    \\\\EndFor
    \\\\State recompute $\\\\widehat{\\\\mathrm{MEC}}, \\\\mathrm{IG}, \\\\widehat{\\\\Sigma}, \\\\mathrm{MRC}, \\\\{\\\\mathrm{FMEC}\\\_i\\\\}$ \\\\Comment{Algorithm 1, on the bootstrap tensor}
    \\\\State $S\\\_b \\\\gets$ \\\\Call{GreedyConstrainedFMEC}{$\\\\cdot$} \\\\Comment{Algorithm 2}
    \\\\State store $\\\\{\\\\mathrm{FMEC}\\\_i^{(b)}\\\\}$ and $S\\\_b$
\\\\EndFor
\\\\State compute $95\\\\%$ BCa intervals from $\\\\{\\\\mathrm{FMEC}\\\_i^{(b)}\\\\}\\\_{b=1}^{B}$
\\\\State estimate $(\\\\widehat\\\\sigma^2\\\_{\\\\mathrm{task}},\\\\widehat\\\\sigma^2\\\_{\\\\mathrm{gen}},\\\\widehat\\\\sigma^2\\\_{\\\\mathrm{adj}})$ by the variance-components decomposition (Eq. 20a)
\\\\State $\\\\mathrm{SelFreq}\\\_i \\\\gets \\\\dfrac{\\\\#\\\\{\\\\,b : M\\\_i \\\\in S\\\_b\\\\,\\\\}}{B}$;\\\\quad
       $\\\\mathrm{RSP} \\\\gets \\\\dfrac{\\\\#\\\\{\\\\,b : S\\\_b = S^\\\\star\\\\,\\\\}}{B}$ \\\\Comment{Eq. 51}
\\\\State \\\\Return BCa intervals, variance components, selection frequencies, $\\\\mathrm{RSP}$
\\\\end{algorithmic}
\\\\end{algorithm}
```

**Interpretation.** $\\mathrm{SelFreq}*i \\approx 1$ indicates an indispensable model; $\\approx 0.5$ an unstable contribution; $\\approx 0$ a rarely useful model. **Crucially, $\\mathrm{RSP}$ and $\\mathrm{SelFreq}$ are now computed over resamples that perturb all three variance components**, so they no longer overstate stability by ignoring generation and adjudication noise. Complexity is $O\\big(B \\cdot (nK + n^2 m + n^2)\\big)$; the inner replicate resample adds only a constant factor because it draws from the already-collected $R$ replicates rather than re-querying the providers — the cost of capturing $\\sigma^2*{\\mathrm{gen}}$ is paid once, at collection, in the $R$ replicates of the tensor. BCa is preferred over the percentile interval for its robustness to the skew and bias of these nonlinear functionals.

### 7.4 Algorithm 4 — Oracle-Regret Evaluation (small pools)

**Purpose.** For small candidate pools ($n \\le 20$), enumerate all ensembles to obtain the oracle $S^{\*}$ (Eq. 28) and measure regret (Eq. 29) directly, validating the greedy selection against the true optimum rather than against a worst-case bound.

```latex
\\\\begin{algorithm}\\\[t]
\\\\caption{Oracle-Regret Evaluation}
\\\\label{alg:oracle-regret}
\\\\begin{algorithmic}\\\[1]
\\\\Require Candidate pool $N$ with $n \\\\le 20$; utility $U(\\\\cdot)$; FMEC ensemble $S\\\_{\\\\mathrm{FMEC}}$.
\\\\Ensure  Absolute and relative regret.
\\\\State $U^{\\\*} \\\\gets -\\\\infty$;\\\\quad $S^{\\\*} \\\\gets \\\\varnothing$
\\\\For{each subset $S \\\\subseteq N$} \\\\Comment{$2^n$ enumeration}
    \\\\If{$S$ satisfies budgets $C\\\_{\\\\max}, L\\\_{\\\\max}, R\\\_{\\\\max}$ \\\\textbf{and} $U(S) > U^{\\\*}$}
        \\\\State $U^{\\\*} \\\\gets U(S)$;\\\\quad $S^{\\\*} \\\\gets S$
    \\\\EndIf
\\\\EndFor
\\\\State $\\\\mathrm{Regret} \\\\gets U(S^{\\\*}) - U(S\\\_{\\\\mathrm{FMEC}})$ \\\\Comment{Eq.\\\~(29)}
\\\\State $\\\\mathrm{RelativeRegret} \\\\gets \\\\dfrac{U(S^{\\\*}) - U(S\\\_{\\\\mathrm{FMEC}})}{U(S^{\\\*})}$
\\\\State \\\\Return $\\\\mathrm{Regret},\\\\ \\\\mathrm{RelativeRegret},\\\\ S^{\\\*}$
\\\\end{algorithmic}
\\\\end{algorithm}
```

**Complexity.** $O(2^n)$ utility evaluations; tractable only for $n \\le 20$ and used as a *gold-standard diagnostic* on a representative sub-pool, not as a production procedure.



## 8\. System Architecture, Data Flow, and the Input–Process–Output Model

### 8.1 Architectural scope and an explicit disclaimer

For precision we state plainly what is and is not built. **The framework does not train a neural network.** Its constituent models are *pretrained, frozen, decoder-only transformer LLMs accessed as black-box services* through an OpenAI-compatible inference gateway. The architectural contribution is therefore an **orchestration-and-selection architecture** — a statistical estimation pipeline (offline) plus a fusion-and-serving pipeline (online) — not a novel trainable network. Where the term "network architecture" is used, it refers to (a) the *constituent expert models* (whose internal transformer architecture is treated as exogenous), and (b) the *system architecture* that selects, combines, supervises, and routes among them. The single learnable component, optional and lightweight, is the router *policy* of Section 8.5 (a contextual bandit), which has no deep-network weights in its baseline form.

The architecture comprises four interacting subsystems:

```latex
\\\\text{(i) Expert models} \\\\;\\\\longrightarrow\\\\; \\\\text{(ii) Fusion operator} \\\\;\\\\longrightarrow\\\\;
\\\\text{(iii) FMEC estimation \\\\\\\& selection} \\\\;\\\\longrightarrow\\\\; \\\\text{(iv) Optional learned router}
\\\\tag{33}
```

Subsystems (i)–(ii) constitute the *serving path* (how a final answer is produced from a query); subsystem (iii) is the *offline analysis path* (how the ensemble is chosen); subsystem (iv) closes the loop by learning, from realized utility, which action to take per task.

### 8.2 Subsystem (i): constituent expert models

Each expert $M\_i$ is a frozen LLM with attribute tuple (Eq. 1). The illustrative pool of the motivating example consists of mid-tier open-weight and commodity models (e.g., a fast general model, a strong math/coding model, and a long-context/agentic model), with one or more frontier models reserved as *baselines and optional judges*. Models expose a single interface — `chat.completions` with `(system\\\_prompt, user\\\_prompt, temperature, max\\\_tokens, response\\\_format)` — and return text or schema-constrained JSON. **Inputs:** task prompt and decoding settings. **Process:** autoregressive generation (exogenous). **Outputs:** a `ModelAnswer` (free-form) or a `CandidateSolution` (structured: thesis, reasoning, optional code, confidence). The expert layer is the unit over which capability $\\mu\_i$, cost $c\_i$, latency $l\_i$, robustness $r\_i$, and the error vector $E\_i$ are measured.

### 8.3 Subsystem (ii): the fusion operator

The fusion operator $f$ maps a set of expert outputs to a single answer. It has four stages: **fan-out** (send the same task to several experts in parallel), **independent reasoning** (each expert solves without seeing the others), **normalization** (convert each answer to a common schema: conclusion, reasoning, uncertainty, citations, code), and **adjudication/synthesis** (a judge model, deterministic verifier, or voting rule produces the final answer). Schematically,
$$
\\text{Prompt} \\rightarrow {M\_1, M\_2, \\ldots, M\_n} \\rightarrow \\text{Adjudicator} \\rightarrow \\text{Final Answer}. \\tag{33b}
$$
The framework is agnostic to the choice of $f$; three concrete instances are supported and used in the protocol:

1. **Judge-based synthesis.** A judge LLM compares candidate answers under a strict adjudication rubric (Appendix C.2) and returns a `FusionResult` (final answer, winning models, disagreements, confidence, rationale). The judge is instructed not to average and to penalize hallucination.
2. **Cost-aware weighted fusion.** Each expert is assigned a score and normalized weight
$$
s\_i = \\frac{q\_i, r\_i}{c\_i^{,a}, \\ell\_i^{,b}}, \\qquad w\_i = \\frac{s\_i}{\\sum\_{j} s\_j}, \\tag{34}
$$
with exponents $a,b>0$ controlling cost- and latency-sensitivity. These weights instantiate $w$ in (11)–(15) for voting/mixture fusion.
3. **Verifier-augmented fusion.** Candidates are scored *deterministically* (executable code is run in a sandbox; arithmetic, schema, and citation checks are applied) before any LLM judging, and the highest deterministic score wins ties broken by self-reported confidence. This is the most robust instance for verifiable tasks.

**Fusion-quality, cost, and latency relations.** The expected quality of the fused answer is a function of individual qualities and pairwise error correlations,
$$
Q\_F = f\\big(q\_1, \\ldots, q\_n,; \\rho\_{12}, \\rho\_{13}, \\ldots, \\rho\_{ij}\\big), \\tag{35}
$$
the fused cost is additive in the experts plus the judge/synthesizer,
$$
C\_F = \\sum\_{i=1}^{n} C\_i + C\_J, \\tag{36}
$$
and, with parallel fan-out, total latency is dominated by the slowest expert plus the adjudicator,
$$
T\_F \\approx \\max\_i T\_i + T\_J. \\tag{37}
$$
Two limiting calculations make the mechanism precise. If $n=3$ experts have independent errors and equal accuracy $p$, majority-vote correctness is
$$
P(\\text{majority correct}) = 3p^2 - 2p^3, \\tag{38}
$$
e.g., $p=0.75 \\Rightarrow 0.844$. For *verifiable* tasks with iterative repair over $k$ attempts,
$$
P(\\text{correct after } k \\text{ attempts}) = 1 - (1-p)^k, \\tag{39}
$$
e.g., $p=0.70,, k=3 \\Rightarrow 0.973$. Equations (38)–(39) hold under independence; real LLM errors are correlated, which is precisely why FMEC penalizes correlated risk (Eq. 11) and rewards *informative* diversity (Eq. 9) rather than diversity per se.

### 8.4 Subsystem (iii): the FMEC estimation-and-selection pipeline

This is the offline analysis path, organized as five sequential layers; the data transformation at each layer is given in Section 8.6.

* **Layer 1 — Benchmark qualification.** Filter $N$ to models above the percentile threshold (Eq. eligibility, Section 11.2). *In:* public/internal benchmark vectors. *Out:* qualified pool.
* **Layer 2 — Evaluation and annotation.** Run the qualified pool on the held-out task suite; record correctness indicators $A\_{it}$ and multidimensional error vectors $E\_{it}$ (Eq. 2). *Out:* per-(model, task) outcome and error records.
* **Layer 3 — Information-gain estimation.** Estimate $\\mathrm{IG}(M\_j \\mid S)$ from correctness events (Eq. 9). *Out:* per-model information-gain scores.
* **Layer 4 — Risk estimation.** Estimate $\\widehat{\\Sigma}$ (Eq. 10) and marginal risk contributions (Eq. 12). *Out:* covariance matrix and $\\mathrm{MRC}$ vector.
* **Layer 5 — FMEC scoring and constrained optimization.** Combine via (13); run Algorithm 2 to select $S^{\*}$ under budgets (14). *Out:* FMEC table and selected ensemble.

Algorithm 1 implements Layers 2–5 for scoring; Algorithm 2 implements the selection; Algorithm 3 wraps Layers 2–5 in a bootstrap; Algorithm 4 provides the oracle reference for Layer 5.

### 8.5 Subsystem (iv): the optional learned router

A mature system should not always call all experts — that is wasteful. It learns a *policy* $\\pi$ mapping a task to an action (which experts to call, whether to invoke the judge, the verifier, retrieval, etc.):
$$
\\pi(a \\mid x), \\tag{40}
$$
chosen to maximize expected net utility,
$$
\\max\_{\\pi}; \\mathbb{E}\\big\[,Q(x,a) - \\lambda, C(a) - \\mu, L(a) - \\gamma, R(x,a),\\big], \\tag{41}
$$
where $\\lambda,\\mu,\\gamma$ are penalty weights. Online, for task $x\_t$ and action $a\_t$ the system observes reward
$$
r\_t = Q\_t - \\lambda, C\_t - \\mu, L\_t, \\tag{42}
$$
and learns the value-maximizing action,
$$
a\_t = \\arg\\max\_{a} \\mathbb{E}\[,r\_t \\mid x\_t, a,]. \\tag{43}
$$
This is a **contextual bandit**; a baseline $\\varepsilon$-greedy estimator updates action-values incrementally per context (Appendix C.4 gives the reference implementation). The router consumes the FMEC table (Layer 5) as a prior over which experts are worth calling for which task class, closing the loop between offline selection and online serving.

### 8.6 End-to-end Input–Process–Output (IPO) model

The research methodology — *not* the served product — has the following IPO structure. Data transformation proceeds strictly left-to-right.

|Stage|Inputs|Process|Outputs|
|-|-|-|-|
|**0. Qualification**|Benchmark-percentile vectors per candidate; threshold $P\_{\\min}$|Threshold filter; optional stratified sampling by family/scale/cluster|Qualified, de-biased candidate pool $N$|
|**1. Generation**|Held-out task suite $T$; prompt templates; decoding settings|Parallel fan-out to each $M\_i$; structured-output capture|Raw answer records $Y\_{it}$ (text/JSON)|
|**2. Scoring \& annotation**|Raw answers $Y\_{it}$; ground truth $Z\_t$; rubric/verifier|Deterministic scoring + taxonomy labeling|Correctness indicators $A\_{it}$; error vectors $E\_{it}$|
|**3. Coalition sampling**|$A,E$ records; coalition count $K$; utility weights $\\theta$|Monte Carlo coalition draws; utility $v(\\cdot)$ evaluation (Eq. 3)|$\\widehat{\\mathrm{MEC}}\_i$ (Eq. 6)|
|**4. Information \& risk**|$A\_{it}$, $E\_{it}$|Conditional-MI estimation (Eq. 9); covariance estimation (Eq. 10), $\\mathrm{MRC}$ (Eq. 12)|$\\mathrm{IG}\_i$; $\\widehat{\\Sigma}$; $\\mathrm{MRC}\_i$|
|**5. FMEC \& selection**|$\\widehat{\\mathrm{MEC}}, \\mathrm{IG}, \\mathrm{MRC}$; $\\eta,\\lambda$; budgets|Combine (Eq. 13); greedy constrained selection (Alg. 2)|FMEC table; selected ensemble $S^{\*}$; weights $w$|
|**6. Validation**|Tasks $T$; bootstrap count $B$|Bootstrap (Alg. 3); hypothesis tests; oracle regret (Alg. 4)|BCa CIs; $\\mathrm{RSP}$; $p$-values; effect sizes; regret|
|**7. Serving (optional)**|New query $x$; $S^{\*}$/router policy $\\pi$|Fusion operator (Sec. 8.3); router (Sec. 8.5)|Final fused answer + confidence|

### 8.7 Interconnections

The subsystems are tightly coupled. The *error vectors* produced in Stage 2 feed both the covariance estimate (risk, Layer 4) and the correctness-event information-gain estimate (Layer 3). The *utility function* (Eq. 3) is the shared currency linking generation cost/latency telemetry (Stages 1–2) to the optimization objective (Layer 5). The *FMEC table* is consumed by the optimizer (Algorithm 2), the validator (Algorithm 3, via re-estimation), and the router (Eq. 40, as a task-class prior). The *oracle module* (Algorithm 4) calibrates trust in greedy selection by measuring realized regret against (28). Finally, the *fusion operator* (Stage 7) and the *evaluation harness* (Stages 1–2) are deliberately the same machinery, so that the utility measured offline is the utility delivered online — preventing a train/serve mismatch.



## 9\. Data: Raw Input Structures, Example DataFrames, and Derived Structures

This section specifies the *raw, unprocessed* data fed into the methodology and the *derived* structures computed from it. All structures are shown in code blocks; every field is discussed. Example DataFrames are capped at ten rows. Values are illustrative placeholders, not measured results.

### 9.1 Raw input data structures

**(a) Raw model-answer record.** One record per (model, task) generation. This is the immediate output of Stage 1 and the input to scoring.

```python
# Raw, unprocessed answer record (free-form fusion path)
model\\\_answer = {
    "task\\\_id":      "math\\\_0007",       # str  : foreign key to the task record
    "model\\\_id":     "deepseek-v4-pro", # str  : provider/model identifier (exact, from gateway)
    "answer":       "The limit equals e. Proof: ...",  # str : verbatim model output text
    "error":        None,              # Optional\\\[str] : transport/decoding error, else None
    "prompt\\\_tokens":  812,             # int  : billed input tokens (cost telemetry)
    "completion\\\_tokens": 437,          # int  : billed output tokens (cost telemetry)
    "latency\\\_s":    2.41,              # float: wall-clock seconds for this call
    "temperature":  0.20,              # float: decoding setting used (provenance)
    "seed":         20260618,          # int  : RNG seed for reproducibility
}
```

**(b) Raw structured-solution record.** One record per (model, task) on the verifier-augmented path (Section 8.3, instance 3).

```python
# Raw structured candidate (verifier-augmented path)
candidate\\\_solution = {
    "task\\\_id":     "code\\\_0042",
    "model\\\_id":    "kimi-k2.6",
    "thesis":      "Off-by-one in the loop bound.",      # str   : one-line conclusion
    "reasoning":   "The range stops at n-1, so ...",     # str   : chain of reasoning
    "python\\\_code": "def f(n): return sum(range(n+1))",   # Optional\\\[str] : executable code or None
    "confidence":  0.78,                                  # float in \\\[0,1] : self-reported confidence
    "error":       None,                                  # Optional\\\[str]
}
```

**(c) Task record.** One record per evaluation task; defines ground truth and metadata.

```python
task\\\_record = {
    "task\\\_id":        "math\\\_0007",     # str  : primary key
    "domain":         "math",          # str  : {math, code, science, finance, long\\\_context, planning, tool}
    "subdomain":      "real\\\_analysis", # str  : finer label for stratified reporting
    "prompt":         "Evaluate lim ...",  # str : the user-facing task text
    "ground\\\_truth":   "e",             # Any : reference answer / unit tests / rubric key
    "scoring\\\_rule":   "exact\\\_match",   # str  : {exact\\\_match, unit\\\_tests, judge\\\_rubric, numeric\\\_tol}
    "difficulty":     "olympiad",      # str  : optional difficulty stratum
    "is\\\_held\\\_out":    True,            # bool : MUST be True (disjoint from qualification benchmarks)
}
```

### 9.2 Example DataFrames (raw inputs)

**(d) Outcome-and-error tensor** — the central raw matrix from Stage 2 and **the primary released artifact of the entire study** (Section 13.3). One row per (model, task, *replicate*): the replicate axis is mandatory, because run-to-run generation variance is a first-class quantity (Section 11.8) and cannot be recovered from a single draw. Columns `e\\\_\\\*` are the multidimensional error taxonomy (Eq. 2; here the 8-category refinement); the trailing columns carry the provenance needed for drift detection (Section 15.1).

```python
import pandas as pd

# Per-(model, task, replicate) outcome tensor; >=1 replicate per pair (reference
# instantiation uses R=5). Provenance columns make silent model drift detectable
# and make the tensor the deterministic input to every downstream estimator.
outcomes = pd.DataFrame(\\\[
    # task\\\_id     model\\\_id          rep  correct  e\\\_reason e\\\_comp e\\\_halluc e\\\_ctx e\\\_plan e\\\_tool e\\\_safety e\\\_refusal  cost\\\_usd latency\\\_s  provider\\\_version          system\\\_fingerprint  seed
    \\\["math\\\_0007", "gemini-3-flash",  0,  0,       1,       0,     0,       0,    0,     0,     0,       0,         0.0021,  1.18,      "gemini-3-flash@2026-05-30","fp\\\_8a1c...",        20260618],
    \\\["math\\\_0007", "gemini-3-flash",  1,  1,       0,       0,     0,       0,    0,     0,     0,       0,         0.0021,  1.21,      "gemini-3-flash@2026-05-30","fp\\\_8a1c...",        20260619],
    \\\["math\\\_0007", "kimi-k2.6",       0,  1,       0,       0,     0,       0,    0,     0,     0,       0,         0.0039,  2.05,      "kimi-k2.6@2026-06-02",     "fp\\\_2bd9...",        20260618],
    \\\["math\\\_0007", "deepseek-v4-pro", 0,  1,       0,       0,     0,       0,    0,     0,     0,       0,         0.0051,  2.41,      "deepseek-v4-pro@2026-06-01","fp\\\_77e0...",       20260618],
    \\\["code\\\_0042", "gemini-3-flash",  0,  1,       0,       0,     0,       0,    0,     0,     0,       0,         0.0023,  1.22,      "gemini-3-flash@2026-05-30","fp\\\_8a1c...",        20260618],
    \\\["code\\\_0042", "kimi-k2.6",       0,  0,       0,       1,     0,       0,    0,     0,     0,       0,         0.0041,  2.11,      "kimi-k2.6@2026-06-02",     "fp\\\_2bd9...",        20260618],
    \\\["code\\\_0042", "deepseek-v4-pro", 0,  1,       0,       0,     0,       0,    0,     0,     0,       0,         0.0052,  2.55,      "deepseek-v4-pro@2026-06-01","fp\\\_77e0...",       20260618],
], columns=\\\[
    "task\\\_id","model\\\_id","rep","correct","e\\\_reason","e\\\_comp","e\\\_halluc","e\\\_ctx",
    "e\\\_plan","e\\\_tool","e\\\_safety","e\\\_refusal","cost\\\_usd","latency\\\_s",
    "provider\\\_version","system\\\_fingerprint","seed",
])
```

*Column discussion.* `task\\\_id` (str): foreign key to the task record; **never** a qualification-benchmark id. `model\\\_id` (str): exact gateway identifier, preserved verbatim. `rep` (int): replicate index $0..R-1$; the within-pair replicates are what license the generation-variance component $\\sigma^2\_{\\mathrm{gen}}$ of Eq. (20a) and the cluster bootstrap of Section 11.8. `correct` (int $\\in {0,1}$): the correctness indicator $A\_{it}$; the sole input to the outcome-based information-gain estimator (Eqs. 9, 9a). `e\\\_reason, e\\\_comp, e\\\_halluc, e\\\_ctx, e\\\_plan, e\\\_tool, e\\\_safety, e\\\_refusal` (int $\\in {0,1}$): failure-type flags constituting the error vector $E\_{it}$ (Eq. 2); these populate the covariance estimate $\\widehat\\Sigma$ (Eq. 10). `cost\\\_usd` (float): per-call dollar cost derived from `prompt\\\_tokens`, `completion\\\_tokens`, and the provider price; feeds $C(S)$. `latency\\\_s` (float): wall-clock latency; feeds $L(S)$. `provider\\\_version` (str) and `system\\\_fingerprint` (str): the provider's served-version string and per-request fingerprint (exposed by OpenAI-compatible gateways), recorded on **every** call so that a version change between collection and reproduction is detectable rather than silent (Section 15.1). `seed` (int): the per-call decoding seed where the provider honors one. A row is the atomic unit over which all downstream estimators operate.

**Canonicalization and content hash.** The released tensor is canonicalized before hashing — rows sorted by `(task\\\_id, model\\\_id, rep)`, columns fixed to the order above, floats serialized at fixed precision — and a **SHA-256 over the canonical bytes** is published in the manuscript (Section 13.3). A reproduction is *Tier-1 valid* iff it regenerates the published hash; every downstream table (coalition samples, $\\widehat\\Sigma$, FMEC, bootstrap outputs) is then a deterministic function of this hashed object plus the seeds of Section 10.2.

**(e) Benchmark-qualification table** — one row per candidate; used **only** for the Layer-1 eligibility filter, **never** in final evaluation.

```python
qualification = pd.DataFrame(\\\[
    # model\\\_id          family     params\\\_b  mmlu  gpqa  humaneval  swebench  math   arena\\\_elo  pct\\\_rank
    \\\["gemini-3-flash",  "Gemini",  None,     0.84, 0.52, 0.86,      0.41,     0.74,  1287,      0.81],
    \\\["kimi-k2.6",       "Kimi",    None,     0.86, 0.55, 0.83,      0.46,     0.79,  1302,      0.86],
    \\\["deepseek-v4-pro", "DeepSeek",None,     0.88, 0.58, 0.89,      0.52,     0.83,  1330,      0.90],
    \\\["mid-llama-70b",   "Llama",   70,       0.79, 0.44, 0.74,      0.30,     0.61,  1198,      0.62],
    \\\["mid-qwen-72b",    "Qwen",    72,       0.82, 0.48, 0.78,      0.35,     0.68,  1245,      0.73],
], columns=\\\[
    "model\\\_id","family","params\\\_b","mmlu","gpqa","humaneval",
    "swebench","math","arena\\\_elo","pct\\\_rank",
])
```

*Column discussion.* `family` (str): architecture family, used for stratified sampling to prevent over-representation (Section 11.3). `params\\\_b` (Optional\[float]): parameter count in billions where disclosed; `None` for closed models. `mmlu, gpqa, humaneval, swebench, math` (float $\\in \[0,1]$): public benchmark scores from the cited sources (Section 13.2). `arena\\\_elo` (int): a human-preference Elo rating. `pct\\\_rank` (float $\\in \[0,1]$): composite benchmark percentile; a model qualifies iff `pct\\\_rank` $\\ge P\_{\\min}$.

### 9.3 Derived data structures

**(f) Coalition-utility samples** (Stage 3). One row per Monte Carlo draw; consumed by the $\\widehat{\\mathrm{MEC}}$ average (Eq. 6).

```python
coalition\\\_samples = pd.DataFrame(\\\[
    # target\\\_model      k   coalition\\\_ids                          v\\\_with    v\\\_without  delta
    \\\["deepseek-v4-pro", 0,  \\\["gemini-3-flash"],                    0.7421,   0.6010,    0.1411],
    \\\["deepseek-v4-pro", 1,  \\\["gemini-3-flash","kimi-k2.6"],        0.8033,   0.7188,    0.0845],
    \\\["kimi-k2.6",       0,  \\\["deepseek-v4-pro"],                   0.7905,   0.7402,    0.0503],
    \\\["gemini-3-flash",  0,  \\\["kimi-k2.6","deepseek-v4-pro"],       0.8101,   0.8050,    0.0051],
], columns=\\\["target\\\_model","k","coalition\\\_ids","v\\\_with","v\\\_without","delta"])
```

*Column discussion.* `target\\\_model` (str): the model $M\_i$ whose contribution is being estimated. `k` (int): coalition-sample index $1..K$. `coalition\\\_ids` (list\[str]): the sampled coalition $S\_k \\subseteq N\\setminus{M\_i}$. `v\\\_with`, `v\\\_without` (float): utilities $v(S\_k\\cup{M\_i})$ and $v(S\_k)$ from Eq. 3. `delta` (float): the marginal contribution summand of Eq. 6. The mean of `delta` grouped by `target\\\_model` is $\\widehat{\\mathrm{MEC}}\_i$.

**(g) Error-covariance matrix** $\\widehat\\Sigma$ (Stage 4), symmetric $n \\times n$ (or category-blocked), the input to risk (Eq. 11) and $\\mathrm{MRC}$ (Eq. 12):

```python
import numpy as np
models = \\\["gemini-3-flash","kimi-k2.6","deepseek-v4-pro"]
Sigma\\\_hat = pd.DataFrame(
    np.array(\\\[\\\[0.090, 0.018, 0.012],
              \\\[0.018, 0.077, 0.009],
              \\\[0.012, 0.009, 0.061]]),
    index=models, columns=models,
)  # diagonal: per-model failure variance; off-diagonal: co-failure covariance
```

**(h) FMEC results table** (Stage 5), the final ranking primitive and the headline artifact consumed by the optimizer, validator, and router:

```python
fmec\\\_table = pd.DataFrame(\\\[
    # model\\\_id          mec\\\_hat   ig       mrc      fmec     ci\\\_low   ci\\\_high  sel\\\_freq
    \\\["deepseek-v4-pro", 0.1128,   0.0312,  0.0205,  0.1180,  0.094,   0.142,   0.98],
    \\\["kimi-k2.6",       0.0846,   0.0451,  0.0152,  0.0998,  0.071,   0.129,   0.91],
    \\\["gemini-3-flash",  0.0307,   0.0188,  0.0098,  0.0349,  0.012,   0.058,   0.43],
], columns=\\\["model\\\_id","mec\\\_hat","ig","mrc","fmec","ci\\\_low","ci\\\_high","sel\\\_freq"])
```

*Column discussion.* `mec\\\_hat` ($\\widehat{\\mathrm{MEC}}\_i$, Eqs. 6, 6a), `ig` ($\\mathrm{IG}\_i$, Eqs. 9, 9a), `mrc` ($\\mathrm{MRC}\_i$, Eq. 12) combine via Eq. 13 into `fmec`. `ci\\\_low`/`ci\\\_high` are the $95%$ BCa bounds from the **nested** bootstrap (Algorithm 3); `sel\\\_freq` is the **per-model selection frequency** $\\mathrm{SelFreq}\_i$ — the fraction of bootstrap replicates in which $M\_i$ is selected — and is distinct from the recommendation-level Recommendation-Stability Probability $\\mathrm{RSP}$ of Eq. (51), which is reported once for the ensemble as a whole. (The two were conflated in earlier drafts; Algorithm 3 now separates them.) Note the intended phenomenon: a model can have a *lower* `mec\\\_hat` yet a *higher* `fmec` once information gain and risk are accounted for (the empirical signature of Proposition 1).



## 10\. Parameters and Study Configuration

This section enumerates every *non-data-structure* input parameter and then fuses the raw-data schemas and parameters into a single nested study-configuration dictionary. **Security note:** no secret is ever hardcoded; API credentials are read from environment variables at run time, exactly as in the reference implementation. The configuration is sufficient to re-run the entire protocol.

### 10.1 Parameter dictionaries

```python
import os

# 
# Utility-function weights (Eq. 3) — application-dependent preferences.
# Source: deployment policy; defaults below are the "balanced" profile.
# 
UTILITY\\\_WEIGHTS = {
    "alpha": 1.00,   # weight on expected performance Q(S)
    "beta":  0.50,   # weight on correlated-failure risk R(S)
    "gamma": 0.30,   # weight on inference cost C(S)
    "delta": 0.20,   # weight on latency L(S)
}

# 
# FMEC combination hyperparameters (Eq. 13).
# Source: model-selection hyperparameters; tuned via sensitivity analysis.
# 
FMEC\\\_HYPERPARAMS = {
    "eta":    0.75,  # weight on conditional information gain IG\\\_j
    "lambda": 0.60,  # weight on marginal risk contribution MRC\\\_j
}

# 
# Monte Carlo coalition sampling (Eqs. 5-6, 19-21).
# Source: estimator precision plan; K from Hoeffding bound (Eq. 21).
# 
MONTE\\\_CARLO = {
    "K": 512,                       # coalitions per model
    "coalition\\\_sampler": "permutation",  # {permutation (unbiased Shapley, Eq. 6a),
                                    #  size\\\_mixture}; permutation is the default and
                                    #  fixes the estimand unambiguously (Section 4.4)
    "size\\\_mixture\\\_weights": None,   # required ONLY if coalition\\\_sampler=="size\\\_mixture";
                                    #  explicit weight per coalition size, else None
    "sampling\\\_seed": 20260618,      # reproducibility seed
}

# 
# Replicate generation (Sections 9.2, 11.8). The replicate axis is what makes
# the generation-variance component estimable; without it sigma^2\\\_gen is unrecoverable.
# 
REPLICATION = {
    "R": 5,                         # generations per (model, task)
    "adjudication\\\_subsample": 0.20, # fraction of tasks re-judged for sigma^2\\\_adj
    "R\\\_judge": 5,                   # judge replicates on the adjudication subsample
    "replicate\\\_seed\\\_base": 20260618,# per-replicate seeds = base + rep index
}

# 
# Bootstrap validation (Algorithm 3) — NESTED (cluster) bootstrap.
# Source: uncertainty-quantification protocol.
# 
BOOTSTRAP = {
    "B": 10000,                     # bootstrap replications
    "interval": "BCa",              # bias-corrected and accelerated
    "ci\\\_level": 0.95,
    "scheme": "nested\\\_cluster",     # outer: tasks; inner: replicates within task
    "resample\\\_levels": \\\["task", "replicate", "adjudication"],  # all three (Eq. 20a)
    "report\\\_variance\\\_components": True,   # return (sigma2\\\_task, sigma2\\\_gen, sigma2\\\_adj)
    "bootstrap\\\_seed": 13,
}

# 
# Covariance estimator selection (Eq. 10; ablation A5).
# Source: robust-statistics protocol.
# 
COVARIANCE = {
    "estimator": "ledoit\\\_wolf",     # {empirical, ledoit\\\_wolf, bayesian, bootstrap}
    "shrinkage\\\_target": "diagonal", # Ledoit-Wolf target
    "block\\\_by\\\_error\\\_category": True,# estimate per-failure-type blocks
}

# 
# Information-gain estimator selection (Eqs. 8-9, 9a; ablation A6).
# Source: information-theoretic protocol.
# 
INFORMATION\\\_GAIN = {
    "estimator": "outcome\\\_based\\\_cmi",  # {outcome\\\_based\\\_cmi, entropy\\\_reduction,
                                       #  judge\\\_model, agreement\\\_based}
    "discretization": "correctness\\\_indicator",  # A\\\_j in {0,1}
    "conditioning": "correct\\\_count\\\_sufficient\\\_statistic",  # c\\\_S, not full pattern (Eq. 9a)
    "smoothing": "laplace",            # plug-in CMI with Laplace smoothing
    "smoothing\\\_alpha": 0.5,
    "miller\\\_madow\\\_correction": True,   # per-stratum entropy bias correction (Eq. 9a)
}

# 
# Constrained optimization budgets (Eq. 14; Algorithm 2).
# Source: deployment SLOs; expressed in native units.
# 
OPTIMIZATION = {
    "C\\\_max": 0.02,   # max cost per task (USD)
    "L\\\_max": 4.0,    # max latency per task (seconds)
    "R\\\_max": 0.08,   # max ensemble risk (quadratic-form units)
    "weight\\\_simplex": True,   # sum\\\_i w\\\_i = 1, w\\\_i >= 0
    "selection": "greedy\\\_budget\\\_feasible",  # Algorithm 2
}

# 
# Statistical testing (Section 11.9-11.11).
# Source: confirmatory-analysis plan; corrections pre-registered.
# 
STAT\\\_TESTS = {
    "primary\\\_test": "paired\\\_t\\\_or\\\_wilcoxon",  # choose by normality check
    "multiple\\\_comparison": "holm\\\_bonferroni",
    "alpha\\\_level": 0.05,
    "effect\\\_sizes": \\\["cohens\\\_d", "cliffs\\\_delta", "relative\\\_utility\\\_gain"],
    "robustness\\\_draws": 10000,               # Dirichlet(1,1,1,1) utility weights
    "robustness\\\_prior": "dirichlet\\\_1\\\_1\\\_1\\\_1",
    # - PRE-REGISTERED numeric decision thresholds (Section 11.13) -
    "delta\\\_U\\\_rule": "cost\\\_tied",   # H5 margin Delta\\\_U = gamma\\\*(C\\\_frontier - C\\\_open) (Eq. 48a);
    "delta\\\_U\\\_floor": 0.02,         #   with an absolute floor of 2% of |U(frontier)|
    "delta\\\_tau": 0.20,             # H4 requires Kendall tau <= 0.80 (bound away from 1)
    "rsp\\\_floor": 0.90,             # recommendation deemed stable iff RSP >= 0.90 (Eq. 51)
    "kappa\\\_floor": 0.70,           # adjudicator-human Cohen's kappa floor (else re-adjudicate)
    "negligible\\\_d": 0.20,          # Cohen's d below this AND ...
    "negligible\\\_cliffs\\\_delta": 0.147,  # ... Cliff's delta below this => operationally void
}

# 
# Model-drift monitoring (Section 15.1, Limitation 2). Converts "drift invalidates
# Sigma" from a disclaimed limitation into a thresholded operational control.
# 
DRIFT\\\_MONITOR = {
    "capture\\\_provider\\\_version": True,     # log served-version string every call
    "capture\\\_system\\\_fingerprint": True,   # log per-request fingerprint every call
    "canary\\\_task\\\_set\\\_size": 100,          # fixed probe set re-run on a schedule
    "recheck\\\_interval\\\_days": 7,
    "trigger\\\_metric": "correctness\\\_rate\\\_shift",  # |delta accuracy| outside CI ...
    "trigger\\\_threshold": 0.03,            #   ... beyond 3 points triggers re-estimation
    "embedding\\\_distance\\\_check": True,     # secondary distributional-shift probe
}

# 
# Contamination audit (Section 13.2). A precondition for valid inference, not
# an optional robustness check.
# 
CONTAMINATION\\\_AUDIT = {
    "ngram\\\_overlap\\\_n": 13,                # substring/n-gram overlap probe
    "overlap\\\_exclusion\\\_rate": 0.10,       # exclude an item if overlap rate exceeds this
    "canary\\\_probe": True,                 # canary-string / membership-inference probe
    "report\\\_per\\\_benchmark\\\_per\\\_model": True,
    "exclusion\\\_rule": "drop\\\_item\\\_if\\\_any\\\_model\\\_contaminated",
}

# 
# Decoding / inference settings per role (provenance for every LLM call).
# Source: reference implementation defaults.
# 
LLM\\\_SETTINGS = {
    "panel":  {"temperature": 0.20, "max\\\_tokens": 1200, "response\\\_format": None},
    "solver": {"temperature": 0.10, "max\\\_tokens": 2000, "response\\\_format": "json\\\_object"},
    "judge":  {"temperature": 0.00, "max\\\_tokens": 2000, "response\\\_format": "json\\\_object"},
}
```

### 10.2 Fused nested study-configuration dictionary

The following single object fuses raw-data schemas, model identifiers, inference settings, prompt-template references, preprocessing, architecture, and plotting parameters. It is the master configuration for one run of the protocol.

```python
STUDY\\\_CONFIG = {

    # ==========================================================================
    # GATEWAY AND CREDENTIALS  (secrets sourced from the environment ONLY)
    # ==========================================================================
    "gateway": {
        "base\\\_url": "https://openrouter.ai/api/v1",   # OpenAI-compatible endpoint
        "api\\\_key\\\_env\\\_var": "OPENROUTER\\\_API\\\_KEY",       # name of the env var, not the secret
        "api\\\_key": os.environ.get("OPENROUTER\\\_API\\\_KEY"),  # loaded at runtime; never hardcoded
        "discover\\\_models\\\_at\\\_runtime": True,            # resolve exact IDs via /models
    },

    # ==========================================================================
    # MODEL ROSTER  (illustrative; resolved to exact gateway IDs at runtime)
    # ==========================================================================
    "models": {
        "panel": \\\["gemini-3-flash", "kimi-k2.6", "deepseek-v4-pro"],
        "judge": "deepseek-v4-pro",                    # or a reserved frontier model
        "frontier\\\_baselines": \\\["gpt-5.5", "opus-4.8", "fable-5"],  # baselines only
        "id\\\_resolution\\\_keywords": {                    # robust to provider renames
            "gemini-3-flash":  \\\["gemini", "flash"],
            "kimi-k2.6":       \\\["kimi", "k2"],
            "deepseek-v4-pro": \\\["deepseek", "v4", "pro"],
        },
    },

    # ==========================================================================
    # INFERENCE SETTINGS  (see LLM\\\_SETTINGS, Section 10.1)
    # ==========================================================================
    "llm\\\_settings": {
        "panel":  {"temperature": 0.20, "max\\\_tokens": 1200, "response\\\_format": None},
        "solver": {"temperature": 0.10, "max\\\_tokens": 2000, "response\\\_format": "json\\\_object"},
        "judge":  {"temperature": 0.00, "max\\\_tokens": 2000, "response\\\_format": "json\\\_object"},
        "request\\\_timeout\\\_s": 60,
        "max\\\_retries": 2,
        "parallel\\\_fanout": True,
    },

    # ==========================================================================
    # PROMPT TEMPLATES  (full text in Appendix C)
    # ==========================================================================
    "prompts": {
        "panel\\\_system":  "APPENDIX\\\_C1\\\_PANEL\\\_SYSTEM\\\_PROMPT",
        "judge\\\_system":  "APPENDIX\\\_C2\\\_JUDGE\\\_SYSTEM\\\_PROMPT",
        "solver\\\_system": "APPENDIX\\\_C3\\\_STRUCTURED\\\_SOLVER\\\_PROMPT",
        "judge\\\_user\\\_template": "Original prompt:\\\\n{prompt}\\\\n\\\\nCandidate answers:\\\\n{answers\\\_json}",
    },

    # ==========================================================================
    # RAW-DATA SCHEMAS  (Section 9.1-9.2 field contracts)
    # ==========================================================================
    "schemas": {
        "model\\\_answer": \\\["task\\\_id","model\\\_id","answer","error","prompt\\\_tokens",
                          "completion\\\_tokens","latency\\\_s","temperature","seed"],
        "candidate\\\_solution": \\\["task\\\_id","model\\\_id","thesis","reasoning",
                                "python\\\_code","confidence","error"],
        "task\\\_record": \\\["task\\\_id","domain","subdomain","prompt","ground\\\_truth",
                         "scoring\\\_rule","difficulty","is\\\_held\\\_out"],
        "outcomes\\\_columns": \\\["task\\\_id","model\\\_id","correct","e\\\_reason","e\\\_comp",
                              "e\\\_halluc","e\\\_ctx","e\\\_plan","e\\\_tool","e\\\_safety",
                              "e\\\_refusal","cost\\\_usd","latency\\\_s"],
        "error\\\_taxonomy": \\\["e\\\_reason","e\\\_comp","e\\\_halluc","e\\\_ctx",
                            "e\\\_plan","e\\\_tool","e\\\_safety","e\\\_refusal"],
    },

    # ==========================================================================
    # PREPROCESSING PARAMETERS
    # ==========================================================================
    "preprocessing": {
        "qualification\\\_threshold\\\_pct": 0.80,          # P\\\_min (Section 11.2)
        "stratify\\\_by": \\\["family", "param\\\_scale", "capability\\\_cluster"],
        "param\\\_scale\\\_bins": {"small": "<20B", "medium": "20B-70B", "large": ">70B"},
        "n\\\_capability\\\_clusters": 4,                    # k-means on benchmark vectors
        "held\\\_out\\\_disjoint\\\_check": True,               # enforce eval ∩ qual = ∅
        "cost\\\_per\\\_token": {"input": "from\\\_provider", "output": "from\\\_provider"},
    },

    # ==========================================================================
    # ARCHITECTURE PARAMETERS  (Section 8)
    # ==========================================================================
    "architecture": {
        "fusion\\\_operator": "verifier\\\_augmented",       # {judge, weighted\\\_vote, verifier\\\_augmented}
        "weighted\\\_fusion\\\_exponents": {"a": 0.50, "b": 0.25},  # Eq. 34 (cost/latency exponents)
        "sandbox": {"timeout\\\_s": 5, "network": "disabled", "fs": "restricted"},
        "router": {
            "enabled": False,                          # optional learned router (Section 8.5)
            "policy": "epsilon\\\_greedy",                # contextual bandit
            "epsilon": 0.10,
            "reward\\\_weights": {"lambda": 0.50, "mu": 0.02, "gamma": 0.00},  # Eq. 42
            "context\\\_features": \\\["coding","text","quant","general"],
        },
    },

    # ==========================================================================
    # ESTIMATION / OPTIMIZATION / VALIDATION  (Sections 6-7, 10.1)
    # ==========================================================================
    "utility\\\_weights":  {"alpha": 1.00, "beta": 0.50, "gamma": 0.30, "delta": 0.20},
    "fmec\\\_hyperparams": {"eta": 0.75, "lambda": 0.60},
    "monte\\\_carlo":      {"K": 512, "coalition\\\_sampler": "permutation", "sampling\\\_seed": 20260618},
    "replication":      {"R": 5, "adjudication\\\_subsample": 0.20, "R\\\_judge": 5},
    "covariance":       {"estimator": "ledoit\\\_wolf", "block\\\_by\\\_error\\\_category": True},
    "information\\\_gain": {"estimator": "outcome\\\_based\\\_cmi",
                          "conditioning": "correct\\\_count\\\_sufficient\\\_statistic",
                          "smoothing\\\_alpha": 0.5, "miller\\\_madow\\\_correction": True},
    "optimization":     {"C\\\_max": 0.02, "L\\\_max": 4.0, "R\\\_max": 0.08},
    "bootstrap":        {"B": 10000, "interval": "BCa", "ci\\\_level": 0.95,
                          "scheme": "nested\\\_cluster",
                          "resample\\\_levels": \\\["task","replicate","adjudication"],
                          "report\\\_variance\\\_components": True},
    "stat\\\_tests":       {"multiple\\\_comparison": "holm\\\_bonferroni", "alpha\\\_level": 0.05,
                          "delta\\\_tau": 0.20, "rsp\\\_floor": 0.90, "kappa\\\_floor": 0.70,
                          "delta\\\_U\\\_rule": "cost\\\_tied", "delta\\\_U\\\_floor": 0.02,
                          "negligible\\\_d": 0.20, "negligible\\\_cliffs\\\_delta": 0.147},
    "drift\\\_monitor":    {"capture\\\_system\\\_fingerprint": True, "canary\\\_task\\\_set\\\_size": 100,
                          "recheck\\\_interval\\\_days": 7, "trigger\\\_threshold": 0.03},
    "contamination\\\_audit": {"ngram\\\_overlap\\\_n": 13, "overlap\\\_exclusion\\\_rate": 0.10,
                          "report\\\_per\\\_benchmark\\\_per\\\_model": True},

    # ==========================================================================
    # PLOTTING / REPORTING PARAMETERS
    # ==========================================================================
    "plots": {
        "efficient\\\_frontier": {"x": "cost\\\_usd", "y": "quality", "annotate": "ensemble\\\_id"},
        "fmec\\\_ci\\\_forest": {"sort\\\_by": "fmec", "show\\\_ci": True},
        "covariance\\\_heatmap": {"cmap": "viridis", "annot": True},
        "rsp\\\_bar": {"threshold\\\_lines": \\\[0.5, 0.9]},
        "variance\\\_components\\\_bar": {"components": \\\["task","gen","adj"]},  # Eq. 20a
        "dpi": 300, "figure\\\_format": "pdf", "font\\\_scale": 1.1,
    },

    # ==========================================================================
    # GLOBAL REPRODUCIBILITY  (two-tier; Section 13.3)
    # ==========================================================================
    "reproducibility": {
        "global\\\_seed": 20260618,
        "log\\\_model\\\_versions": True,
        "tier1\\\_artifact": "outcome\\\_tensor",          # frozen primary artifact (Section 9.2)
        "tensor\\\_sha256": "<populated upon execution of the reference instantiation>",
        "tier1\\\_claim": "bit\\\_exact\\\_from\\\_tensor\\\_and\\\_seeds",
        "tier2\\\_claim": "instrumented\\\_and\\\_bounded\\\_via\\\_variance\\\_components",  # Eq. 20a
        "persist": \\\["prompts","seeds","outcome\\\_tensor","tensor\\\_sha256",
                    "coalition\\\_samples","covariance","fmec\\\_estimates",
                    "bootstrap\\\_outputs","variance\\\_components","selected\\\_ensemble",
                    "model\\\_version\\\_manifest","contamination\\\_report"],
    },
}
```



## 11\. Experimental Design and Statistical Methodology

This section specifies a **pre-registered, confirmatory** evaluation protocol for the FMEC framework. It is a *proposed* design: no results are reported. The purpose is twofold. First, it renders the central hypothesis falsifiable by committing — in advance of any data collection — to hypotheses, estimators, decision rules, and stopping criteria. Second, it furnishes the downstream reproduction protocol (Section 12) with a complete experimental contract. Every quantity referenced here is defined in Sections 3–10; the design adds no new modeling assumptions, only the apparatus required to test the ones already stated.

The governing methodological commitment is the separation of the **exploratory** and **confirmatory** regimes (Wagenmakers et al., 2012). All hyperparameters — the utility weights $(\\alpha,\\beta,\\gamma,\\delta)$, the FMEC coefficients $(\\eta,\\lambda)$, the Monte Carlo budget $K$, and the qualification threshold $P\_{\\min}$ — are frozen on a development split that is **disjoint** from the evaluation split. The confirmatory analyses below are then executed once on the held-out evaluation tasks. This discipline is what licenses the inferential statements; absent it, the bootstrap intervals and corrected $p$-values would be uninterpretable.

### 11.1 Pre-registered hypotheses

We state five hypotheses. Each is paired with a null, a directional alternative, and the estimator and test that adjudicate it (Sections 11.9–11.11). Let $S\_{\\mathrm{FMEC}}$ denote the ensemble selected by Algorithm 2 under the full objective; $S\_{\\mathrm{Bench}}$ the benchmark-top-$K$ baseline (Baseline 1); $S\_{\\mathrm{NoRisk}}$ the ablation with $\\lambda=0$ (Ablation A2); and $S\_{\\mathrm{NoIG}}$ the ablation with $\\eta=0$ (Ablation A1).

**H1 (Utility dominance).** The FMEC-selected ensemble attains higher expected task utility than the benchmark-ranked ensemble of equal cardinality:

$$
H\_1:\\quad \\mathbb{E}!\\left\[U(S\_{\\mathrm{FMEC}})\\right] ;>; \\mathbb{E}!\\left\[U(S\_{\\mathrm{Bench}})\\right]. \\tag{44}
$$

The null is $H\_1^0:\\ \\mathbb{E}\[U(S\_{\\mathrm{FMEC}})] \\le \\mathbb{E}\[U(S\_{\\mathrm{Bench}})]$. This is the framework's primary claim; its rejection is necessary (not sufficient) for the central thesis.

**H2 (Information–contribution coupling).** Across qualified candidates, conditional information gain is positively associated with marginal ensemble contribution:

$$
H\_2:\\quad \\rho!\\left(\\mathrm{IG}\_i,\\ \\mathrm{MEC}\_i\\right) ;>; 0, \\tag{45}
$$

where $\\rho$ is the population Spearman rank correlation. H2 tests the mechanistic premise that complementarity (not raw competence) drives contribution.

**H3 (Risk reduction).** Incorporating the marginal-risk term yields a lower-risk ensemble than the risk-agnostic selector:

$$
H\_3:\\quad R!\\left(\\mathbf{w}*{\\mathrm{FMEC}}\\right) ;<; R!\\left(\\mathbf{w}*{\\mathrm{NoRisk}}\\right), \\tag{46}
$$

with $R(\\cdot)$ the correlated-failure risk of Eq. (11), evaluated at the optimizer's weight vectors.

**H4 (Insufficiency of benchmark rank).** Benchmark ranking is *not* a sufficient statistic for ensemble contribution:

$$
H\_4:\\quad \\tau!\\left(\\mathrm{rank}*{\\mathrm{Bench}},\\ \\mathrm{rank}*{\\mathrm{FMEC}}\\right) ;<; 1, \\tag{47}
$$

where $\\tau$ is Kendall's rank-correlation coefficient. The substantive (not merely statistical) form requires $\\tau$ to be bounded away from $1$ by a pre-specified margin $\\tau \\le 1-\\Delta\_\\tau$. **We fix $\\Delta\_\\tau = 0.20$** (i.e., the test asserts $\\tau\\le 0.80$), pre-registered before evaluation (`stat\\\_tests.delta\\\_tau`); a value below this would license a "benchmark rank is *almost* sufficient" reading, which the framework declines to claim on a hair's-breadth discordance. Because Kendall's $\\tau$ has sampling variance that depends on the number of qualified models $n$, the protocol includes a pre-registered **power analysis at the realized $n$**: if $n$ is too small for the bound-away-from-one test to achieve $0.80$ power at $\\alpha=0.05$ against $\\tau=0.80$, H4 is reported as underpowered rather than as evidence either way. H4 is the negative claim that motivates the entire framework.

**H5 (Open-weight non-inferiority).** On the evaluated task distribution, an FMEC-selected ensemble drawn from an **open-weight** candidate pool is non-inferior to the strongest single frontier model, within a pre-specified margin $\\Delta\_U>0$:

$$
H\_5:\\quad \\mathbb{E}!\\left\[U(S\_{\\mathrm{FMEC}}^{\\mathrm{open}})\\right] ;\\ge; \\mathbb{E}!\\left\[U(m\_{\\mathrm{frontier}}^{\\star})\\right] - \\Delta\_U. \\tag{48}
$$

H5 is a **non-inferiority** hypothesis and is tested as such (Walker and Nowacki, 2011): the null is *inferiority by more than* $\\Delta\_U$, and rejection supports practical substitutability. The margin $\\Delta\_U$ is a deployment decision and must be *derived*, not guessed. We pre-register a **cost-tied margin**: the open-weight ensemble may give up at most the utility equivalent of the inference cost it saves relative to the frontier model,
$$
\\Delta\_U ;=; \\max!\\Big(\\gamma\\big(C\_{\\mathrm{frontier}} - C\_{\\mathrm{open}}\\big),; 0.02,\\big|U(m\_{\\mathrm{frontier}}^\\star)\\big|\\Big), \\tag{48a}
$$
with $\\gamma$ the cost weight of Eq. (3) and the second term an absolute floor of $2%$ of frontier utility (`stat\\\_tests.delta\\\_U\\\_rule = "cost\\\_tied"`, `delta\\\_U\\\_floor = 0.02`). The derivation makes "non-inferior" mean *"the quality shortfall is no larger than the cost saving is worth,"* which is the economically meaningful sense of substitutability and removes the arbitrariness of a hand-set margin.

The asymmetry across hypotheses is deliberate. H1 and H5 are superiority/non-inferiority claims tested with one-sided procedures; H2 and H3 are mechanistic; H4 is an equivalence-style *bound-away-from-one* claim. Conflating them under a single two-sided test would misstate the scientific content.

### 11.2 Candidate pool and qualification

The candidate pool $\\mathcal{P}$ comprises every model under consideration for deployment, spanning open-weight and API-served frontier systems across multiple families and parameter scales. The pool should be as large and heterogeneous as the compute budget permits; diversity of *failure modes*, not raw count, is the binding resource (Theorem 2).

Not every candidate is eligible for selection on every task. A model $M\_i$ **qualifies** on task domain $t$ if its development-split score clears a relative threshold:

$$
\\text{qualify}(M\_i, t) ;\\iff; \\mathrm{score}*t(M\_i) ;\\ge; P*{\\min}\\cdot \\max\_{j\\in\\mathcal{P}} \\mathrm{score}\_t(M\_j),
$$

with $P\_{\\min}\\in\[0.75,,0.80]$ fixed in configuration (`qualification\\\_threshold\\\_pct`, Section 10.2). Qualification is a *gate*, not a *selector*: it removes models that are non-competitive on a task so that the contribution estimator is not dominated by candidates that never participate in any high-utility coalition. The qualification and selection populations are enforced disjoint from the evaluation set (`held\\\_out\\\_disjoint\\\_check = True`).

### 11.3 Stratified sampling of candidates

To prevent the pool from being dominated by a single lineage — which would induce precisely the correlated failures of Eq. (23) and inflate apparent diversity — candidates are stratified along three axes before any contribution estimation:

1. **Family / lineage.** Provider and base-model genealogy (a proxy for shared pre-training data and tokenizer, hence correlated errors).
2. **Parameter scale.** Binned `small (<20B) / medium (20B–70B) / large (>70B)` (`param\\\_scale\\\_bins`, Section 10.2).
3. **Capability cluster.** $k$-means ($k=4$) on standardized benchmark-score vectors, grouping models by *profile shape* rather than overall level.

Stratification serves the estimator, not cosmetic balance: it ensures the coalition sampler (Algorithm 1) explores combinations that cross lineage and capability boundaries, which is where information-complementarity (Theorem 1) is most likely to be realized.

### 11.4 Task suite

Evaluation spans heterogeneous reasoning regimes so that complementarity is measured, not assumed. The suite covers seven domains, each instantiated by an established public benchmark where one exists (full citations in Section 13):

|Domain|Reference benchmark|Capability probed|
|-|-|-|
|Mathematical reasoning|MATH; grade-school arithmetic word problems|Multi-step symbolic and numeric reasoning|
|Code generation|HumanEval|Functional correctness under unit tests|
|Software engineering|SWE-bench|Repository-level patch synthesis|
|Scientific QA|GPQA|Graduate-level, search-resistant questions|
|Broad knowledge|MMLU|Cross-domain factual breadth|
|Quantitative finance|Domain task set|Numerically grounded financial reasoning|
|Long-context / planning / tool use|Agentic and long-context tasks|Multi-turn decomposition and tool orchestration|

Each domain contributes a held-out evaluation partition disjoint from the qualification partition. Per-domain stratification of items (by difficulty and sub-skill) is performed to support the variance-reduction guarantees of Section 6.

**Corpus fixity (the two non-canonical domains).** Five of the seven domains map to canonical, publicly hosted benchmarks (Section 13.2), so two independent teams evaluate on identical items. The remaining two — **quantitative finance** and **long-context / planning / tool use** — have no single canonical artifact, and leaving them "assembled by the implementer" would make exactly the two domains that exercise the framework's novel error categories (tool-use, long-context) irreproducible across teams. Because complementarity is task-distribution-dependent (Limitation 3, Section 15.1), divergent corpora here would by themselves prevent convergence on H1/H5. The protocol therefore requires **one of two pre-registered resolutions**, in order of preference:

1. **Freeze and release the exact item sets.** The implementer constructs the finance and long-context/agentic corpora once, assigns each item a stable `task\\\_id` and a per-item content hash, and **publishes the frozen item sets with a construction manifest** (source, license, selection procedure, hash list) as part of the reproducibility package (Section 13.3). These two domains are then admitted to the **confirmatory** analysis on the same footing as the canonical five. This is the only resolution consistent with the paper's target reproducibility tier.
2. **Adopt versioned public benchmarks** for these domains where suitable ones exist, pinned by version tag and commit/revision hash and cited as specific artifacts rather than as "a domain task set."

If neither resolution is available for a given run, those two domains are **demoted to exploratory** and **excluded from the confirmatory H1–H5 tests**, with the exclusion stated in the pre-registration. A smaller, fully reproducible confirmatory core is preferred to a larger one whose two newest domains cannot be rebuilt by a third party. "Documented as part of corpus provenance" is necessary but *not* sufficient: a provenance note that one built finance tasks does not let another team reconstruct *those* tasks — only a hashed, released item set does.

### 11.5 Error taxonomy

Risk estimation (Eq. 10) requires *labeled failure modes*, not merely binary correctness, because the covariance that matters is the covariance of **error types**. Every incorrect response is annotated into one of eight mutually exclusive categories:

1. **Factual / knowledge error** — a verifiably false assertion.
2. **Reasoning / logical error** — invalid inference from correct premises.
3. **Arithmetic / computational error** — a numerical or symbolic miscalculation.
4. **Instruction-following / format error** — failure to honor explicit constraints or output schema.
5. **Code-correctness / runtime error** — code that fails tests, raises, or does not compile.
6. **Hallucination / fabrication** — invented citations, APIs, or entities.
7. **Incompleteness / omission** — truncated or partial answers; dropped sub-questions.
8. **Safety / refusal / policy error** — unwarranted refusal or policy-inappropriate output.

Annotation is performed by a rubric-driven adjudication procedure (Appendix C2) with a human-audited subsample; inter-rater agreement is reported (Cohen's $\\kappa$; Cohen, 1960). The error-category indicator vectors populate $E\_i$ (Eq. 2) and hence the covariance $\\Sigma$ (Eq. 10). The covariance is optionally **block-structured by category** (`block\\\_by\\\_error\\\_category = True`) so that the risk term penalizes models that fail *the same way*, which is the failure geometry Theorem 2 identifies as catastrophic.

### 11.6 Baselines and treatment

The treatment is $S\_{\\mathrm{FMEC}}$: the ensemble selected by Algorithm 2 under the full objective $\\mathrm{FMEC}\_j=\\widehat{\\mathrm{MEC}}\_j+\\eta,\\mathrm{IG}\_j-\\lambda,\\mathrm{MRC}\_j$ (Eq. 13). It is compared against six baselines, each isolating a competing selection philosophy and each constrained to the **same cardinality and the same budget envelope** so that comparisons are like-for-like:

|#|Baseline|Selection rule|What it isolates|
|-|-|-|-|
|1|Benchmark top-$K$|Rank by aggregate benchmark score; take top $K$|The prevailing "pick the best models" heuristic (drives H1, H4)|
|2|Random|Uniform random size-matched draw from qualified pool|The null value of *any* principled selection|
|3|Correlation-minimized|Minimize pairwise error correlation only|Diversity *without* contribution or utility|
|4|Shapley-only|Rank by $\\phi\_j$ (Eq. 7); no IG, no risk|Contribution attribution alone|
|5|Information-only|Rank by $\\mathrm{IG}\_j$ (Eq. 8–9); no MEC, no risk|Complementarity alone|
|6|Greedy-utility|Greedily maximize $v(S)$ (Eq. 3) directly, no FMEC decomposition|Whether the FMEC decomposition adds value over naive utility greedy|

Baselines 4 and 5 are the most informative comparators: the gap between the full FMEC objective and each single-term ranker quantifies the marginal value of *combining* contribution, information, and risk rather than optimizing any one in isolation.

### 11.7 Ablations

Where the baselines isolate competing *philosophies*, the ablations isolate the contribution of each *component of FMEC itself*. Six ablations are pre-registered:

|#|Ablation|Modification|Question answered|
|-|-|-|-|
|A1|No information gain|$\\eta = 0$|Does the IG term improve selection beyond contribution and risk?|
|A2|No risk|$\\lambda = 0$|Does the marginal-risk term reduce realized risk (H3)?|
|A3|Leave-one-out MEC|Replace Shapley $\\phi\_j$ by single LOO contribution|Does coalition averaging matter, or does drop-one suffice?|
|A4|Monte Carlo budget|$K\\in{64,128,256,512,1024}$|Estimator stability vs. compute (Eqs. 20–21)|
|A5|Covariance estimator|sample vs. Ledoit–Wolf vs. block-diagonal|Sensitivity of risk to covariance regularization|
|A6|Information-gain estimator|response-based $I(Y\_j;Z\\mid Y\_S)$ vs. outcome-based $I(A\_j;Z\\mid A\_S)$|Which information signal is operative?|

A4 is reported as a convergence curve: the bootstrap standard error of the FMEC estimate as a function of $K$, expected to contract at the $O(1/\\sqrt{K})$ rate predicted by Eq. (20). A3 directly tests whether the (more expensive) Shapley averaging is empirically necessary or whether the leave-one-out approximation of Eq. (30) is adequate on the realized task distribution.

### 11.8 Uncertainty quantification by the nested bootstrap and variance decomposition

All point estimates — $\\widehat{\\mathrm{MEC}}\_j$, $\\mathrm{IG}\_j$, $\\mathrm{MRC}\_j$, and the composite $\\mathrm{FMEC}\_j$ — are statistics computed from a finite, *noisy* evaluation sample. The noise has three distinct sources, and a defensible uncertainty statement must account for all three rather than the one a naive bootstrap captures:

$$
\\mathrm{Var}\\big(\\widehat{\\theta}\\big);=;\\underbrace{\\sigma^2\_{\\mathrm{task}}}*{\\text{which items were sampled}};+;\\underbrace{\\sigma^2*{\\mathrm{gen}}}*{\\text{run-to-run generation noise}};+;\\underbrace{\\sigma^2*{\\mathrm{adj}}}\_{\\text{adjudicator noise}}. \\tag{20a}
$$

A bootstrap that resamples **only tasks** — the single-level scheme — estimates $\\sigma^2\_{\\mathrm{task}}$ alone and is structurally blind to $\\sigma^2\_{\\mathrm{gen}}$ and $\\sigma^2\_{\\mathrm{adj}}$. In a study whose data-generating process is repeated black-box LLM inference at non-zero (and even at zero) temperature, $\\sigma^2\_{\\mathrm{gen}}$ is not negligible: the same model on the same task can return a different answer, flip a correctness flag, and thereby change $\\widehat{\\Sigma}$ and the selected ensemble across honest re-runs. **Reporting a task-only interval and calling it a stability certificate would overstate stability and is the methodological error this section exists to prevent.**

We therefore (i) collect $R\\ge 5$ independent generations per (model, task) — the replicate axis of the outcome tensor (Section 9.2, `replication.R`) — and (ii) quantify uncertainty by the **nonparametric nested (cluster) bootstrap** (Efron and Tibshirani, 1993; Algorithm 3): the outer loop resamples tasks (clusters) with replacement, the inner loop resamples the $R$ replicates *within* each retained task, and a third resampling draws judge replicates on the adjudication subsample (`replication.adjudication\\\_subsample`, `R\\\_judge`). Each level injects its corresponding variance component, and the three are reported separately via the variance-components decomposition of Eq. (20a) (`bootstrap.report\\\_variance\\\_components`). The inner resample adds only a constant factor in compute because it draws from the *already-collected* replicates rather than re-querying providers — the price of capturing $\\sigma^2\_{\\mathrm{gen}}$ is paid once, at collection, and the bootstrap inherits it for free. The fusion-simulation residual $\\widehat\\sigma\_u$ of Section 4.2.1 is propagated as an additional, separately reported term where the simulated-fusion surrogate is used for $Q(S)$.

Confidence intervals use the **bias-corrected and accelerated (BCa)** method (Efron, 1987) rather than the percentile interval, because the FMEC statistics are nonlinear functionals of the data and are generally biased and skewed in finite samples. The BCa endpoints adjust the percentile cutoffs by a bias-correction term $\\hat z\_0$ — estimated from the fraction of bootstrap replicates below the point estimate — and an acceleration term $\\hat a$ estimated by jackknife (here a *delete-one-cluster* jackknife, consistent with the cluster resampling), which corrects for dependence of the variance on the underlying parameter. Where a statistic is approximately pivotal we additionally report the bootstrap-$t$ interval as a sensitivity check; material disagreement between BCa and bootstrap-$t$ is flagged rather than silently resolved. All intervals are reported at the 95% level (`ci\\\_level = 0.95`).

The nested-bootstrap distribution of $\\mathrm{FMEC}\_j$ is the object from which we read the *recommendation-stability* diagnostics of Section 11.10; because it now perturbs all three variance components, it is the honest substrate for the stability claims that make a recommendation actionable, rather than an optimistic one that ignores the dominant run-to-run source.

### 11.9 Confirmatory tests and multiplicity control

Each hypothesis maps to a specific test on bootstrap-derived quantities:

* **H1 / H5** (utility superiority / non-inferiority): one-sided test on the paired per-task utility differences $U(S\_{\\mathrm{FMEC}})-U(S\_{\\mathrm{Bench}})$ (H1) and on the non-inferiority margin (H5), using the bootstrap distribution of the mean difference. Pairing is by evaluation task to remove task-difficulty variance.
* **H2** (coupling): test that the Spearman $\\rho(\\mathrm{IG},\\mathrm{MEC})$ exceeds $0$, with the null distribution obtained by permuting the pairing between IG and MEC across candidates.
* **H3** (risk reduction): one-sided test on $R(\\mathbf{w}*{\\mathrm{NoRisk}})-R(\\mathbf{w}*{\\mathrm{FMEC}})>0$ over bootstrap resamples.
* **H4** (insufficiency of benchmark rank): test that Kendall's $\\tau$ is bounded above by $1-\\Delta\_\\tau$, combined with a sign-based count of rank inversions between benchmark and FMEC orderings.

Because five hypotheses are tested on one evaluation corpus, the family-wise error rate is controlled by the **Holm–Bonferroni** step-down procedure (Holm, 1979). With ordered $p$-values $p\_{(1)}\\le\\cdots\\le p\_{(5)}$ and family level $\\alpha=0.05$ (`stat\\\_tests`, Section 10.1), $H\_{(k)}$ is rejected iff $p\_{(j)} \\le \\alpha/(5-j+1)$ for all $j\\le k$. Holm is chosen over plain Bonferroni for its uniform power advantage at no cost in validity, and over false-discovery-rate procedures because the setting is confirmatory (a small fixed family of pre-registered hypotheses), where strong family-wise control is the appropriate guarantee.

### 11.10 Effect sizes and recommendation stability

Statistical significance is necessary but not sufficient; a deployment decision turns on **magnitude** and **stability**. Three effect-size and stability measures are reported alongside every test.

**Standardized mean difference (Cohen's $d$; Cohen, 1988).** For the paired utility differences,

$$
d ;=; \\frac{\\bar{U}*{\\mathrm{FMEC}} - \\bar{U}*{\\mathrm{Bench}}}{s\_{\\mathrm{pooled}}}, \\tag{49}
$$

with $s\_{\\mathrm{pooled}}$ the pooled standard deviation. We report $d$ with its bootstrap CI and interpret it on conventional thresholds while acknowledging their domain-dependence.

**Cliff's $\\delta$ (ordinal dominance; Cliff, 1993).** Because utilities are not guaranteed interval-scaled, we report the nonparametric dominance statistic

$$
\\delta ;=; \\frac{#{U(S\_{\\mathrm{FMEC}}) > U(S\_{\\mathrm{Bench}})} ;-; #{U(S\_{\\mathrm{FMEC}}) < U(S\_{\\mathrm{Bench}})}}{n\_{\\mathrm{F}}, n\_{\\mathrm{B}}}, \\tag{50}
$$

which is robust to the scale and to outliers and answers the practitioner's question directly: how often does FMEC win?

**Recommendation-Stability Probability (RSP).** The decision-relevant quantity is the probability that the *recommended ensemble does not change* under resampling. With $S^\\star$ the ensemble selected on the full sample and $S^{\\star(b)}$ the ensemble selected on **nested-bootstrap** replicate $b$ (Algorithm 3, resampling tasks, generation replicates, and adjudication draws),

$$
\\mathrm{RSP} ;=; \\frac{1}{B}\\sum\_{b=1}^{B}\\mathbb{1}!\\left\[,S^{\\star(b)} = S^{\\star},\\right]. \\tag{51}
$$

The resampling in (51) **perturbs all three variance components of Eq. (20a)**; an RSP computed over a task-only bootstrap would mechanically inflate, because it holds generation and adjudication noise fixed at their observed draw. A companion **per-model selection frequency** — the fraction of replicates in which each candidate is selected — is reported as a forest plot (`fmec\\\_ci\\\_forest`, Section 10.2). A recommendation is treated as stable only if $\\mathrm{RSP}\\ge 0.90$ (`stat\\\_tests.rsp\\\_floor`, pre-registered); below the floor the framework is *required* to report that it cannot issue a stable recommendation for that deployment, regardless of the point FMEC of the selected ensemble. RSP and selection frequency thereby convert the abstract recommendation into a calibrated statement: a model selected in $49%$ of resamples is not a robust recommendation, and the framework says so by construction.

### 11.11 Robustness to utility specification

The utility weights $(\\alpha,\\beta,\\gamma,\\delta)$ encode a deployment's preferences and are not estimated from data. A recommendation that flips under small perturbations of these weights is fragile. We therefore conduct a **preference-robustness analysis**: $10{,}000$ weight vectors are drawn from a symmetric Dirichlet, $(\\alpha,\\beta,\\gamma,\\delta)\\sim\\mathrm{Dirichlet}(1,1,1,1)$, the full selection pipeline is re-run for each draw, and the distribution of selected ensembles is summarized. The reported quantity is the **selection-stability surface**: the probability mass that each ensemble accumulates across the preference simplex. This is the empirical instantiation of the utility-calibration / Dirichlet-sensitivity analysis of the theoretical section; it certifies whether a recommendation is a property of the *evidence* or merely of one arbitrary weight choice. Robustness to the FMEC coefficients $(\\eta,\\lambda)$ is assessed analogously on a grid.

### 11.12 Oracle-regret evaluation

For candidate pools small enough to enumerate ($n\\le 20$, where the $O(2^n)$ exhaustive search of Algorithm 4 is tractable), we compute the **true** optimal coalition $S^\\star$ (Eq. 28) and the realized **regret** of the FMEC selection (Eq. 29),

$$
\\mathrm{Regret}(S\_{\\mathrm{FMEC}}) ;=; U(S^\\star) - U(S\_{\\mathrm{FMEC}}),
$$

reporting its bootstrap distribution. This is the only analysis that measures *absolute* selection quality against ground truth rather than relative performance against a baseline, and it directly probes the looseness of the greedy approximation guarantee (Section 6.9) on realized instances. The regret distribution, alongside the submodular worst-case bound, brackets the framework's selection optimality from both the empirical and the theoretical side.

### 11.13 Falsifiability and failure conditions

The design is constructed so that the framework can lose. Every failure condition below carries a **pre-registered numeric threshold** (the values fixed in `STAT\\\_TESTS`, Section 10.1); a falsification criterion with an unstated boundary is not a criterion. The central thesis is **rejected** under any of the following pre-committed outcomes:

* **H1 not rejected** after Holm correction at family level $\\alpha=0.05$ — the FMEC ensemble does not beat benchmark-top-$K$ on expected utility. This is the primary falsification.
* **H4 rejected in reverse** — benchmark rank and FMEC rank are statistically *not* distinguishable from identical, i.e. the data are consistent with $\\tau > 0.80$ (the complement of the $\\Delta\_\\tau=0.20$ bound, `stat\\\_tests.delta\\\_tau`), implying benchmark scores already suffice and the machinery is unnecessary.
* **Negligible effect sizes** — even if $p<\\alpha$, if $\\text{Cohen's }d < 0.20$ **and** $\\text{Cliff's }\\delta < 0.147$ (`negligible\\\_d`, `negligible\\\_cliffs\\\_delta`), or $\\mathrm{RSP} < 0.90$ (`rsp\\\_floor`), the recommendation is operationally void.
* **Instability under utility perturbation** — if the Dirichlet analysis (Section 11.11) shows no ensemble accumulating dominant selection mass (no ensemble at or above the same $0.90$ stability bar across the preference simplex), the framework cannot issue a stable recommendation for that deployment.
* **Ablation nullity** — if A1 and A2 show that setting $\\eta=0$ and $\\lambda=0$ does not degrade utility by a margin exceeding the H1 effect (no significant loss at $\\alpha=0.05$ after Holm), then the information and risk terms — the framework's distinctive content — add nothing, and FMEC collapses to plain utility greedy (Baseline 6).
* **Adjudication-quality floor breached** — if the adjudicator–human Cohen's $\\kappa$ falls below $0.70$ (`kappa\\\_floor`) on the audited subsample and cannot be remediated by re-adjudication, the correctness labels are deemed too unreliable to support inference on the affected (open-ended) tasks, which are then excluded or the analysis is declared inconclusive for them.

Enumerating the conditions under which the framework fails is not a hedge; it is the property that distinguishes a scientific proposal from an unfalsifiable one. A method that cannot specify its own failure modes — *with numbers* — cannot be trusted when it succeeds.



## 12\. Reproduction Protocol

This section specifies, as an ordered procedure and *without source code*, exactly how to reproduce the FMEC pipeline end-to-end. Each step names the equations, algorithms, configuration blocks, prompt templates, and decoding settings that govern it, so that an independent team can re-implement the method from this document alone. The protocol assumes the configuration object of Section 10.2 (`STUDY\\\_CONFIG`) and the prompt templates of Appendix C as the single source of truth for every constant and string.

**Step 0 — Environment, provenance, drift instrumentation, and contamination audit.**
Fix the global random seed (`reproducibility.global\\\_seed = 20260618`). Record the exact version identifier of every candidate model (`reproducibility.log\\\_model\\\_versions = True`) and, on **every** call, capture the provider's served-version string and per-request `system\\\_fingerprint` (`drift\\\_monitor.capture\\\_system\\\_fingerprint`), writing both into the outcome tensor (Section 9.2) so that a version change between collection and any later reproduction is *detectable rather than silent*. Stand up the drift monitor: a fixed canary probe set of `drift\\\_monitor.canary\\\_task\\\_set\\\_size = 100` items re-run every `recheck\\\_interval\\\_days = 7`, with a correctness-rate shift beyond `trigger\\\_threshold = 0.03` mandating re-estimation. **Run the contamination audit before any inference is used for inference**: for each (benchmark, model version), compute $n$-gram/substring overlap (`contamination\\\_audit.ngram\\\_overlap\\\_n = 13`) between evaluation items and available training-corpus proxies, and run a canary/membership-inference probe; report the per-(benchmark, model) contamination rate and drop any item exceeding `overlap\\\_exclusion\\\_rate = 0.10` under the pre-registered exclusion rule. Establish gateway access through the environment variable named in `gateway.api\\\_key\\\_env\\\_var` (`OPENROUTER\\\_API\\\_KEY`); no key is ever written to disk or configuration. Persist the artifact set listed in `reproducibility.persist`, including the outcome tensor, its SHA-256, the model-version manifest, and the contamination report.

**Step 1 — Assemble and stratify the candidate pool.**
Enumerate the candidate models (`models` roster, Section 10.2). Stratify them by family, parameter scale (`param\\\_scale\\\_bins`), and capability cluster ($k$-means, $k=4$) as specified in Section 11.3. This fixes the population over which contribution is estimated.

**Step 2 — Construct the task corpus and enforce disjointness.**
Partition each task domain (Section 11.4) into a **development** split (for freezing hyperparameters), a **qualification** split, and a held-out **evaluation** split. Enforce `preprocessing.held\\\_out\\\_disjoint\\\_check`: the evaluation split must be disjoint from both development and qualification. For the two non-canonical domains, apply the corpus-fixity resolution of Section 11.4 (freeze + hash + release the item sets, or demote them to exploratory). All confirmatory analyses (Step 9 onward) touch the evaluation split exactly once.

**Step 3 — Qualify candidates per domain.**
For each domain $t$, compute development-split scores and admit model $M\_i$ iff $\\mathrm{score}*t(M\_i)\\ge P*{\\min}\\cdot\\max\_j \\mathrm{score}*t(M\_j)$, with $P*{\\min}$ from `preprocessing.qualification\\\_threshold\\\_pct` (Section 11.2). The output is the per-domain qualified set, the input to selection.

**Step 4 — Collect model responses, with replicates.**
For every qualified model and every evaluation item, draw **$R$ independent generations** (`replication.R = 5`, per-replicate seeds `replicate\\\_seed\\\_base + rep`) using the role-specific decoding settings in `llm\\\_settings` (Section 10.1): panel/candidate generation at `temperature = 0.2`, structured solving at `0.1`, and adjudication at `0.0`. The replicate axis is mandatory: it is the only way to estimate the generation-variance component $\\sigma^2\_{\\mathrm{gen}}$ of Eq. (20a). Candidate generation uses the panel system template (Appendix C1) and, for structured tasks, the structured-solver template (Appendix C3). Persist raw responses in the `model\\\_answer` / `candidate\\\_solution` structures of Section 9.1 and populate the per-(model, task, replicate) outcome tensor of Section 9.2.

**Step 5 — Adjudicate correctness and label error modes (with replicated judging).**
Score each response for correctness and, when incorrect, assign exactly one of the eight error categories (Section 11.5) using the judge system template (Appendix C2) at `temperature = 0.0`. On a fraction `replication.adjudication\\\_subsample = 0.20` of tasks, run the judge `R\\\_judge = 5` times to expose the adjudication-variance component $\\sigma^2\_{\\mathrm{adj}}$ (Eq. 20a). Audit a human-labeled subsample and report inter-rater agreement; **if Cohen's $\\kappa < 0.70$ (`stat\\\_tests.kappa\\\_floor`) and cannot be remediated, the affected open-ended tasks are excluded per Section 11.13**. The labeled outcomes populate the correctness matrix and the error-category vectors $E\_i$ (Eq. 2).

**Step 6 — Estimate marginal contribution (MEC).**
Execute **Algorithm 1** to estimate $\\widehat{\\mathrm{MEC}}\_j$ via Monte Carlo coalition sampling (Eqs. 4–6), with $K$ from `monte\\\_carlo.K` and the coalition seed fixed. The utility of each sampled coalition is computed from Eq. (3) using `utility\\\_weights` $(\\alpha,\\beta,\\gamma,\\delta)$. Optionally compute the Shapley value $\\phi\_j$ (Eq. 7) where the design calls for full coalition averaging rather than the leave-one-out approximation (Eq. 30).

**Step 7 — Estimate information gain (IG).**
Compute the conditional information gain for each candidate (Eqs. 8–9) using the estimator named in `information\\\_gain.estimator` (outcome-based conditional mutual information by default), with Laplace smoothing `smoothing\\\_alpha`. This quantifies each model's *complementary* signal given the responses already in the coalition.

**Step 8 — Estimate the risk model.**
Form the error-mode covariance $\\Sigma$ (Eq. 10) using the estimator in `covariance.estimator` (Ledoit–Wolf shrinkage by default), optionally block-structured by error category (`block\\\_by\\\_error\\\_category`). Marginal risk contribution $\\mathrm{MRC}\_j=(\\Sigma\\mathbf{w})\_j$ follows from Eq. (12).

**Step 9 — Compose FMEC.**
Combine the three signals into the composite score $\\mathrm{FMEC}\_j=\\widehat{\\mathrm{MEC}}\_j+\\eta,\\mathrm{IG}\_j-\\lambda,\\mathrm{MRC}\_j$ (Eq. 13), with $(\\eta,\\lambda)$ from `fmec\\\_hyperparams`. This produces the per-candidate FMEC table of Section 9.3.

**Step 10 — Solve the constrained selection.**
Run **Algorithm 2** (greedy constrained selection) to maximize the portfolio utility $U(\\mathbf{w})=\\sum\_i w\_i,\\mathrm{FMEC}*i$ (Eqs. 14–16) subject to the simplex constraint and the cost, latency, and risk budgets in `optimization` ($C*{\\max},L\_{\\max},R\_{\\max}$). The output is the selected ensemble $S\_{\\mathrm{FMEC}}$ and its weights. Record which constraints are active at the optimum.

**Step 11 — Quantify uncertainty.**
Run **Algorithm 3** (bootstrap validation) with $B$ resamples from `bootstrap.B`, recomputing Steps 6–10 on each resample to obtain BCa confidence intervals (`bootstrap.interval`), the per-model selection frequencies, and the Recommendation-Stability Probability (Eq. 51).

**Step 12 — Confirmatory tests.**
Test the five pre-registered hypotheses (Eqs. 44–48) using the procedures of Section 11.9, controlling the family-wise error rate by Holm–Bonferroni at `stat\\\_tests.alpha\\\_level`. Report effect sizes (Eqs. 49–50) with bootstrap intervals.

**Step 13 — Robustness analyses.**
Execute the preference-robustness analysis (Section 11.11): draw $10{,}000$ weight vectors from $\\mathrm{Dirichlet}(1,1,1,1)$, re-run Steps 9–10 per draw, and report the selection-stability surface. Repeat on a grid over $(\\eta,\\lambda)$.

**Step 14 — Oracle-regret (small pools).**
For pools with $n\\le 20$, run **Algorithm 4** to compute the optimal coalition $S^\\star$ (Eq. 28) by exhaustive search and report the regret distribution (Eq. 29) of the FMEC selection.

**Step 15 — Reporting.**
Generate the standard figures (`plots`, Section 10.2): the cost–quality efficient frontier, the FMEC confidence-interval forest plot, the error-mode covariance heatmap, the RSP/selection-frequency bar chart, and the variance-components bar chart (Eq. 20a), all at `dpi = 300` in the configured figure format. Reconcile every reported number against the persisted artifacts of Step 0, and verify that the recomputed pipeline regenerates the published tensor SHA-256.

This is the precise sense in which the two reproducibility tiers differ. **Tier 1 (computational):** given the released, hash-pinned outcome tensor of Section 9.2 and the seeds of Section 10.2, Steps 6–15 are *fully deterministic* and reproduce the published numbers bit-for-bit — there is no model call on this path, so provider non-determinism cannot enter. **Tier 2 (empirical):** Steps 4–5 (re-collection) are *not* deterministic, because they query mutating black-box endpoints; the protocol does not pretend otherwise but instruments the irreducible variability — the replicate axis quantifies $\\sigma^2\_{\\mathrm{gen}}$, the replicated judge quantifies $\\sigma^2\_{\\mathrm{adj}}$ (Eq. 20a), and the drift monitor and version manifest make any between-collection change in the subjects detectable. Reproducibility is thus *bounded and measured* on the one path where it cannot be made exact, and *exact* on the path where it can.

### 12.1 Reference instantiation (the executed minimal instance)

The reproduction protocol above is exercised by a single, fully specified **reference instantiation** whose purpose is to convert the reproducibility claim from a promise into a verifiable procedure. It is deliberately minimal — small enough to run cheaply and to admit the exact oracle of Algorithm 4 — yet complete enough to release every artifact a third party needs for Tier-1 reproduction.

**Fixed configuration of the reference instantiation.**

* **Pool.** $n \\le 20$ qualified models (so the $O(2^n)$ oracle enumeration of Algorithm 4 is tractable), drawn to cross family and capability-cluster boundaries per Section 11.3; an explicitly open-weight sub-pool is retained for the H5 test.
* **Domains.** The five canonical, publicly hosted benchmarks of Section 13.2 (MMLU, MATH/GSM8K, GPQA, HumanEval, SWE-bench). The two non-canonical domains (finance, long-context/agentic) are included **only** if their item sets are frozen, hashed, and released per Section 11.4; otherwise they are reported as exploratory and excluded from the confirmatory tests.
* **Estimation.** $K = 512$ coalitions via the permutation sampler (Eq. 6a); outcome-based CMI with count-sufficient conditioning and Miller–Madow correction (Eq. 9a); Ledoit–Wolf covariance, block-structured by error category (Eq. 10).
* **Replication and uncertainty.** $R = 5$ generations per (model, task); $20%$ adjudication subsample with $R\_J = 5$ judge replicates; $B = 10{,}000$ nested-cluster bootstrap replicates with the variance-components decomposition of Eq. (20a).
* **Selection and validation.** Greedy constrained selection (Algorithm 2) under the budgets of Section 10.2; oracle-regret (Algorithm 4) computed exactly; the five hypotheses tested under Holm–Bonferroni against the pre-registered thresholds of Section 11.13.

**Released artifacts (the reproducibility package).** Upon execution, the package contains: the frozen outcome tensor (Section 9.2) and its **SHA-256**, the model-version manifest, the contamination report, the frozen non-canonical item sets and hashes (if used), all seeds (`STUDY\\\_CONFIG`), and every persisted intermediate in `reproducibility.persist` (coalition samples, $\\widehat\\Sigma$, FMEC estimates, bootstrap outputs, variance components, selected ensemble). A reader regenerates the FMEC table, the selected ensemble, the variance components, and the hypothesis decisions from the tensor and seeds alone, and confirms Tier-1 reproduction by matching the published hash.

**Honesty about status.** This manuscript reports **no results**: the artifacts above are specified and their schemas fixed, but the execution-derived values — the tensor hash, the per-(benchmark, model) contamination rates, the realized variance components, the realized inter-rater $\\kappa$, and the realized FMEC/selection/decision outputs — are produced *only by running the instantiation* and are referenced throughout as artifacts-to-be-populated, never as fabricated numbers. The amendments of this revision make the path to those artifacts complete and reproducible; they do not, and cannot, substitute for executing it.



## 13\. Essential Sections, Datasets, and Code Availability

This section serves three purposes for an implementer: it maps each reproduction step to the sections of this paper that are *essential* to executing it, it enumerates the public datasets and benchmarks the evaluation depends on with references and access information, and it states the code- and data-availability posture.

### 13.1 Reproduction-step to essential-section map

The reproduction protocol of Section 12 is executable using only the following sections. The map is intended to let a reader locate the precise definitional content behind each operational step rather than re-reading the paper linearly.

|Reproduction step (Section 12)|Essential sections / artifacts|
|-|-|
|0. Environment \& provenance|§10.2 (`gateway`, `reproducibility`); §8.1|
|1. Pool assembly \& stratification|§3; §11.2–11.3; §10.2 (`models`)|
|2. Corpus \& disjointness|§11.4; §10.2 (`preprocessing`)|
|3. Qualification|§11.2; Eq. (1)–(2)|
|4. Response collection|§9.1; §10.1 (`llm\\\_settings`); Appendix C1, C3|
|5. Adjudication \& error labeling|§11.5; Eq. (2); Appendix C2|
|6. MEC estimation|§4 (Eqs. 4–7); Algorithm 1; §6.2–6.4|
|7. Information-gain estimation|§4 (Eqs. 8–9); §10.1 (`information\\\_gain`)|
|8. Risk estimation|§4 (Eqs. 10–12); §10.1 (`covariance`)|
|9. FMEC composition|§4 (Eq. 13)|
|10. Constrained selection|§5 (Eqs. 14–18); Algorithm 2; §6.9|
|11. Uncertainty quantification|§11.8, §11.10 (Eq. 51); Algorithm 3|
|12. Confirmatory tests|§11.1 (Eqs. 44–48), §11.9–11.10 (Eqs. 49–50)|
|13. Robustness|§11.11|
|14. Oracle-regret|§6.13 (Eqs. 28–29); Algorithm 4|
|15. Reporting|§10.2 (`plots`)|

The theoretical sections (§6) are not required to *run* the pipeline but are required to *interpret* it: the convergence guarantees (Eqs. 19–21) justify the Monte Carlo budget, the submodularity result (§6.9) justifies the greedy selector, and the regret analysis (§6.13) sets expectations for the oracle comparison.

### 13.2 Datasets and benchmarks

The evaluation draws on established, publicly available benchmarks, one per capability domain where a canonical artifact exists. The table gives the reference and the canonical public host; URLs were current as of access (mid-June 2026) and should be re-verified at use, as hosting and naming for community benchmarks are subject to change.

|Domain|Benchmark|Reference|Canonical public host|
|-|-|-|-|
|Broad knowledge|MMLU|Hendrycks et al. (2021a)|`github.com/hendrycks/test`; HF `cais/mmlu`|
|Mathematical reasoning|MATH|Hendrycks et al. (2021b)|`github.com/hendrycks/math`|
|Grade-school math|GSM8K|Cobbe et al. (2021)|`github.com/openai/grade-school-math`|
|Scientific QA|GPQA|Rein et al. (2023)|`github.com/idavidrein/gpqa`; HF `Idavidrein/gpqa`|
|Code generation|HumanEval|Chen et al. (2021)|`github.com/openai/human-eval`|
|Software engineering|SWE-bench|Jiménez et al. (2024)|`github.com/princeton-nlp/SWE-bench`|
|Human-preference ranking|Chatbot Arena|Chiang et al. (2024)|operated as *Arena* / LMArena|

A note on the last entry: the human-preference platform introduced as Chatbot Arena moved to a dedicated domain, lmarena.ai, in September 2024 and subsequently incorporated as an independent company. It is referenced here as a *source of comparative human-preference signal* and a model of pairwise adjudication, not as a fixed-item dataset; an implementation that uses it should record the platform version and date, since its model roster and methodology evolve continuously. The quantitative-finance and long-context/agentic domains (Section 11.4) are instantiated from domain-specific task sets assembled by the implementer to the specification of Section 11.4; no single canonical public benchmark is mandated for them, and the choice should be documented as part of the corpus provenance (Step 2).

Use of any benchmark must respect its license and, critically, its **contamination status**: a benchmark whose items appear in a candidate model's pre-training data yields invalid contribution *and* complementarity estimates, because both the model's measured competence and its measured failure structure are corrupted. The disjointness check of Step 2 defends only against qualification/evaluation leakage *within this study*; it does **not** defend against pre-training leakage, which is the real threat, so the audit cannot be left to the implementer's discretion. The protocol therefore pre-registers a concrete contamination audit (`CONTAMINATION\\\_AUDIT`, Section 10.1), run in Step 0 before any outcome is used for inference:

1. **Lexical overlap probe.** For each (benchmark item, model version), compute $n$-gram / verbatim substring overlap at `ngram\\\_overlap\\\_n = 13` against available training-corpus proxies and public dumps; flag any item whose overlap rate exceeds `overlap\\\_exclusion\\\_rate = 0.10`.
2. **Behavioral probe.** Apply a canary-string and/or membership-inference-style test (e.g., perplexity or exact-completion gaps between held-out items and known-public items) per model version, to catch contamination not visible lexically.
3. **Reporting and exclusion.** Report the contamination rate **per (benchmark, model)** (`report\\\_per\\\_benchmark\\\_per\\\_model = True`) in the reproducibility package, and apply the pre-registered exclusion rule (`drop\\\_item\\\_if\\\_any\\\_model\\\_contaminated`): an item flagged for *any* candidate is removed for *all*, so that the surviving corpus is uncontaminated uniformly across the pool. Contamination auditing is a precondition for valid inference, not an optional robustness check; an unaudited run is reported as such and its inferential claims withheld.

**Pinned item sets for the two non-canonical domains.** Per the corpus-fixity requirement of Section 11.4, the quantitative-finance and long-context/agentic domains are admitted to the confirmatory analysis **only** as frozen, released artifacts: each item carries a stable `task\\\_id` and a content hash, the full hash list and a construction manifest (source, license, selection and filtering procedure, generation provenance) are published in the reproducibility package (Section 13.3), and the released item set is itself covered by the audit above. Absent a frozen, hashed release, these domains are reported as exploratory and excluded from H1–H5, so that no confirmatory claim ever rests on a corpus a third party cannot reconstruct byte-for-byte.

### 13.3 Code, data, and artifact availability (two-tier)

This paper reports no *experimental results*; its empirical contribution is the **reproducibility apparatus**, and that apparatus is concrete rather than promissory. The reference implementation — inference-gateway client, structured-output schemas, adjudication procedure, weighted- and verifier-augmented fusion operators, sandboxed code-execution path, the four algorithms of Section 7, and the optional contextual-bandit router — mirrors the architecture of Section 8 and the configuration of Section 10.2, and is released as a deterministic, seed-driven pipeline whose Stage 6–15 outputs are a pure function of released data.

**The primary released artifact is the frozen outcome tensor** of Section 9.2: the per-(model, task, replicate) correctness-and-error matrix with full provenance columns. Everything the framework computes — $\\widehat{\\mathrm{MEC}}$ (Eqs. 6, 6a), IG (Eqs. 9, 9a), $\\widehat\\Sigma$ (Eq. 10), the FMEC table (Eq. 13), the constrained selection (Algorithm 2), the nested bootstrap and variance components (Algorithm 3, Eq. 20a), and the oracle regret (Algorithm 4) — is a deterministic transform of this tensor and the seeds of Section 10.2. Releasing it is therefore what makes the study reproducible at all, and it is what separates this revision from a specification that can only be re-implemented but never re-checked.

The package is organized by reproducibility tier.

* **Tier 1 — computational reproducibility (bit-exact).** The released archive contains: (i) the frozen outcome tensor and its **published SHA-256 over the canonicalized bytes** (`reproducibility.tensor\\\_sha256`); (ii) the configuration object (`STUDY\\\_CONFIG`, Section 10.2) with all seeds; (iii) the frozen prompt templates (Appendix C); (iv) every persisted *stage-wise intermediate* in `reproducibility.persist` — coalition samples, covariance estimate, FMEC estimates, bootstrap outputs, variance components, and the selected ensemble — so a reader can checkpoint-verify at each stage rather than only end-to-end; (v) the frozen, hashed item sets and construction manifests for the two non-canonical domains (Section 11.4), where used. A reproduction is **Tier-1 valid iff re-running the released pipeline on the released tensor regenerates the published SHA-256 and reproduces every downstream table**. No model is called on this path, so provider non-determinism cannot enter and the result is exact.
* **Tier 2 — empirical replicability (bounded, not exact).** Re-collecting the tensor by re-querying the models (Steps 4–5) is *not* deterministic, because the subjects are mutating, black-box commercial endpoints. The package supports a disciplined re-collection rather than pretending it is exact: the **model-version manifest** and per-call `system\\\_fingerprint`/`provider\\\_version` records (Section 9.2) make any between-collection drift detectable; the replicate axis and replicated judging make the generation and adjudication variance components ($\\sigma^2\_{\\mathrm{gen}}$, $\\sigma^2\_{\\mathrm{adj}}$ of Eq. 20a) *measurable* on the new collection; and the contamination report (Section 13.2) is regenerated for the new model versions. A re-collection that lands within the reported variance envelope is a successful Tier-2 replication; one that does not is, by design, traced to a named source — drift, contamination, or genuine effect change — rather than left as an unexplained failure.

In keeping with double-blind reviewing norms, repository and author-identifying information are withheld at submission and replaced by an anonymized placeholder; a permanent, versioned archive (with a DOI) accompanies the de-anonymized release. No API credentials are included in any artifact — only the *name* of the environment variable from which the gateway key is read (Section 10.2) — so the package is shareable without exposing secrets. Execution-derived quantities (the tensor hash, the contamination rates, the realized variance components, the realized $\\kappa$, and all FMEC/selection/decision outputs) are populated **only upon execution of the reference instantiation** (Section 12.1) and are never reported here as fabricated values.



## 14\. Required Disciplines and Skills

Implementing FMEC at a professional standard is an inherently *interdisciplinary* undertaking: the framework borrows its risk machinery from quantitative finance, its attribution machinery from cooperative game theory, its complementarity machinery from information theory, and its evaluation machinery from modern statistical practice, then realizes all of it in production LLM-serving infrastructure. This section enumerates the disciplines a practitioner must command and the concrete skills within each, mapped to the parts of the framework they govern. It is organized as a competency map for a Master's- or PhD-level implementer and is the basis for the disciplines-and-skills synthesis requested downstream.

**1. Mathematics of finance and portfolio theory.** The entire risk-and-selection layer is a transposition of Markowitz mean–variance portfolio construction onto a portfolio of models. The practitioner must understand mean–variance optimization, the efficient frontier and Pareto-dominance (Section 5.2), risk budgeting, and the interpretation of a covariance matrix as the object that governs diversification. Concretely: formulating and solving the constrained allocation of Eqs. (14)–(16); reading the marginal-risk contribution $(\\Sigma\\mathbf{w})\_j$ (Eq. 12) as the sensitivity of portfolio risk to a position; and reasoning about quality-adjusted cost (Eq. 17) as a risk-adjusted-return analogue. Without this lens, the risk term is an opaque penalty rather than a principled diversification control.

**2. Cooperative game theory.** Marginal contribution is defined game-theoretically. The practitioner must understand coalitional games, the marginal contribution of a player to a coalition (Eq. 4), the Shapley value and its axiomatic justification — efficiency, symmetry, null-player, additivity (Eq. 7) — and the computational intractability that motivates Monte Carlo estimation. The required skill is to recognize *why* a model's value is its averaged marginal contribution across coalitions rather than its standalone score, and to know when the cheaper leave-one-out approximation (Eq. 30) is defensible.

**3. Information theory.** Complementarity is quantified by conditional mutual information. The practitioner must command entropy, mutual information, conditional mutual information, and the chain rule of mutual information — the last being the engine of the information-complementarity theorem (Eq. 22). The operative skills are estimating conditional mutual information from finite categorical data with appropriate smoothing (Eqs. 8–9, Section 10.1), and distinguishing response-level from outcome-level information (Ablation A6), which determines what "new information" a model is credited with.

**4. Monte Carlo methods and computational statistics.** Both contribution and its uncertainty are estimated by sampling. Required: Monte Carlo estimator construction and its bias/variance properties (Eqs. 5–6), convergence at the $O(1/\\sqrt{K})$ rate (Eq. 20), sample-complexity reasoning via concentration inequalities (Eq. 21, Hoeffding), and the nonparametric bootstrap with bias-corrected-and-accelerated intervals (Section 11.8). The practitioner must be able to set the Monte Carlo budget $K$ from a target precision rather than by convention, and to read a bootstrap distribution as the sampling distribution of a nonlinear statistic.

**5. Robust multivariate statistics.** The risk model stands or falls on the quality of the covariance estimate, which is high-dimensional relative to the number of evaluation tasks. Required skills: recognizing the instability of the sample covariance in this regime; applying shrinkage estimation (Ledoit–Wolf) and understanding the bias–variance trade-off it strikes (Eq. 10, Ablation A5); and exploiting structure — here, block structure by error category — to regularize the estimate toward the failure geometry that Theorem 2 (Eq. 23) identifies as decisive.

**6. Submodular and constrained optimization.** Selection is a constrained combinatorial optimization. The practitioner must understand monotone submodularity, the greedy algorithm and its approximation guarantees, and — crucially — *which* guarantee applies under *which* constraint: $(1-1/e)$ under a cardinality constraint, $\\tfrac{1}{2}(1-1/e)$ for naive greedy under a knapsack/budget constraint, $(1-1/e)$ via partial-enumeration greedy, and $(1-e^{-\\gamma})$ under $\\gamma$-weak submodularity (Section 6.9). The required skill is to match the algorithm to the constraint structure of Eq. (14) and to know the bound it earns, rather than asserting an unconditional optimality the method does not possess.

**7. Statistical inference and experimental design.** The evaluation is a confirmatory study and must be conducted as one. Required: pre-registration and the exploratory/confirmatory distinction (Section 11); one-sided, paired, and **non-inferiority** testing (H1, H5); rank-correlation inference (H2, H4); family-wise error control by Holm–Bonferroni (Section 11.9); effect-size estimation and interpretation — Cohen's $d$, Cliff's $\\delta$ (Eqs. 49–50); and stratified sampling for variance control (Section 11.3). The skill that ties these together is designing an experiment that *can fail* and specifying in advance what failure looks like (Section 11.13).

**8. LLM systems and API engineering.** The framework is realized against live model-serving infrastructure. Required skills: programming an OpenAI-compatible inference gateway with the base URL and environment-variable-sourced credentials of Section 10.2; eliciting and validating **structured outputs** against typed schemas (Section 9.1); controlling decoding parameters per role — generation, solving, adjudication temperatures (Section 10.1); and reasoning about the cost, latency, and throughput of a multi-model serving path (Eqs. 36–37). Secure execution is a sub-competency in its own right: sandboxing untrusted model-generated code with resource and network restrictions (Section 8.3) is mandatory wherever a verification path executes code.

**9. Reinforcement learning and contextual bandits.** The optional learned router is a contextual-bandit problem. Required: the explore–exploit trade-off, the $\\epsilon$-greedy policy, reward design that internalizes cost and risk (Eq. 42), and online policy update from bandit feedback (Eqs. 40–43). The skill is to recognize routing as sequential decision-making under partial feedback, and to design a reward that does not silently optimize quality at unbounded cost.

**10. Reproducible research and scientific-software engineering.** Finally, the practitioner must be able to make the entire pipeline *reproducible*: global seeding, model-version manifesting, deterministic adjudication, and disciplined persistence of every intermediate artifact (Section 10.2, Section 12). Adjacent to this is rigorous prompt and rubric engineering — designing the panel, solver, and judge templates (Appendix C) and the eight-category error rubric (Section 11.5) so that adjudication is consistent and auditable.

The competency profile is therefore neither "an ML engineer" nor "a quant" in isolation: it is the intersection. The hardest skill is not any single discipline but the *translation* between them — seeing that a model roster is a portfolio, that contribution is a Shapley value, that diversity is conditional mutual information, and that a deployment recommendation is only as credible as the resampling that certifies its stability.



## 15\. Discussion and Limitations

### 15.1 Limitations

A framework that proposes to displace a simple and widely used heuristic — pick the highest-scoring model — must be candid about where it is costly, fragile, or inapplicable. We identify five substantive limitations, none of which is incidental.

**1. Estimation cost.** FMEC is expensive to compute. The contribution estimator costs $O(nK + n^2 m)$ per the complexity analysis of Eq. (32), the bootstrap multiplies the entire pipeline by $B=10{,}000$, and the preference-robustness analysis adds $10{,}000$ further re-solves. This revision *adds* cost in two places, and we state it plainly rather than hide it: collecting $R=5$ generations per (model, task) multiplies generation cost roughly fivefold, and the replicated judging on the adjudication subsample adds judge calls. The trade is deliberate — that fivefold collection cost is what buys an honest generation-variance estimate (Eq. 20a) and is paid *once* at collection, after which the nested bootstrap reuses the stored replicates at no extra inference cost. For a deployment that will serve a single model on a handful of queries, none of this overhead is recoverable. The framework is justified only when the selection is *amortized* — when the chosen ensemble will serve a large query volume, so that a one-time estimation cost is repaid by sustained gains in utility. The break-even analysis between estimation cost and downstream benefit is itself a decision the practitioner must perform; FMEC does not assume it away.

**2. Non-stationarity and model drift.** The contribution, information, and risk estimates are valid only for the *specific model versions* audited. Providers update served models, sometimes silently; an update changes a model's error distribution and therefore its covariance with the rest of the pool, invalidating $\\Sigma$ and the selection that depends on it. This revision converts drift from a disclaimed hazard into a *monitored, thresholded control*: every call records the served-version string and per-request `system\\\_fingerprint` (Section 9.2), and a fixed canary probe set is re-run on a schedule with a correctness-rate-shift trigger (`DRIFT\\\_MONITOR`, Section 10.1) that *mandates* re-estimation when exceeded. This makes drift not merely *detectable* but *actionable* — yet it still cannot make estimates *durable*: monitoring tells you when to re-estimate, it does not remove the need to. Periodic re-estimation is mandatory, and the cadence is a function of the volatility of the provider ecosystem, which is presently high. This is also the dominant reason Tier-2 (empirical) replication is bounded below Tier-1 (computational) reproduction (Section 13.3): the released tensor is exactly reproducible forever; the models that produced it are not.

**3. Task-distribution dependence.** Complementarity is not an intrinsic property of a model; it is a property of a model *relative to a task distribution*. An ensemble that is optimal on the evaluation corpus may be suboptimal on a deployment whose query distribution differs — a model that contributes unique information on mathematical reasoning may be redundant on code. External validity therefore depends on the evaluation corpus being representative of deployment, a condition that must be argued, not assumed, and re-examined when the deployment distribution shifts.

**4. Utility-weight specification.** The weights $(\\alpha,\\beta,\\gamma,\\delta)$ that define $v(S)$ (Eq. 3) are exogenous preferences, not estimable quantities, and the selected ensemble depends on them. The preference-robustness analysis (Section 11.11) characterizes this dependence and can certify when a recommendation is stable across the simplex, but it cannot substitute for genuine elicitation of a deployment's true trade-offs between quality, cost, latency, and risk. Where those preferences are themselves uncertain or contested, the framework's output is correspondingly conditional.

**5. Adjudication dependence.** Correctness labels and error-category assignments are produced by an automated adjudicator (Appendix C2) with a human-audited subsample. Every downstream quantity — $\\widehat{\\mathrm{MEC}}$, $\\mathrm{IG}$, and $\\Sigma$ — inherits the adjudicator's errors and biases. For verifiable tasks (code with tests, problems with checkable answers), verification-augmented fusion (Section 8.3) grounds the labels in execution and sharply reduces this dependence; for open-ended generation it does not, and the framework's estimates are then only as reliable as the judge. This revision strengthens the treatment in two ways. First, the adjudication-variance component $\\sigma^2\_{\\mathrm{adj}}$ is now *measured*, not merely acknowledged: the judge is run $R\_J$ times on an adjudication subsample and that variance is propagated through the nested bootstrap (Eq. 20a), so the judge's non-determinism enters the reported uncertainty rather than hiding inside a single label. Second, a pre-registered Cohen's $\\kappa$ floor of $0.70$ (Section 11.13) *gates* the analysis: open-ended tasks whose adjudicator–human agreement falls below it are excluded or the analysis declared inconclusive for them. Reporting and propagating inter-rater agreement bounds and quantifies this exposure; for genuinely open-ended generation it does not eliminate it, and that residual is the honest limit of the approach where execution cannot ground the label.

A sixth concern, **benchmark contamination** (Section 13.2), cuts across limitations 3 and 5: if evaluation items leaked into a candidate's training data, both its measured competence and its measured complementarity are corrupted. Contamination auditing for the specific versions in use is a precondition for valid inference, not an optional robustness check.

### 15.2 The economics of orchestration (contextual)

The following framing situates FMEC within the microeconomics of AI serving. It is **context, not a tested claim**: the relations below are standard economic models invoked to motivate *why* a risk-and-cost-aware selection layer is economically rational. No empirical economic estimates are made or implied, and the constants are illustrative.

The orchestration layer is best understood as an economic agent that arbitrages quality against cost. Demand for "intelligence" (useful task completions) is price-elastic; a constant-elasticity model writes consumed quantity $Q$ as a function of unit price $P$,

$$
Q ;=; A,P^{-\\varepsilon},\\qquad \\varepsilon > 0, \\tag{52}
$$

with $A$ a scale constant and $\\varepsilon$ the price elasticity. Revenue (or, symmetrically, total spend) is then

$$
R ;=; P,Q ;=; A,P^{,1-\\varepsilon}, \\tag{53}
$$

so that in the elastic regime $\\varepsilon>1$ a *fall* in the unit price of intelligence raises total expenditure — consumption grows faster than price declines. The same logic, applied to the cost of compute per unit of delivered capability, is a Jevons-style observation: letting $g$ denote delivered capability per unit cost and $D(g)$ the induced demand for compute, an elasticity

$$
\\frac{d \\ln D}{d \\ln g} ;>; 1 \\tag{54}
$$

means that making each unit of capability cheaper *increases* total compute consumed rather than decreasing it. A widely circulated industry observation captures the same intuition qualitatively: cheaper intelligence does not shrink the market for intelligence, it enlarges it.

The implication for FMEC is direct. In a world where the unit cost of capability is falling and total demand is consequently rising, the binding question is not "which single model is best" but "which *allocation* of a model portfolio delivers capability per unit cost on the efficient frontier." That is precisely the quantity FMEC optimizes: the quality-adjusted cost of Eq. (17) is the per-unit-capability price the orchestration layer pays, and the constrained selection of Eqs. (14)–(16) chooses the allocation that sits on the cost–quality Pareto frontier (Section 5.2). The verification economics of the serving layer reinforce this — the majority-correctness and independent-verification relations (Eqs. 38–39) show how spending marginal compute on redundancy or verification buys reliability — but whether that spend is worthwhile is, again, an economic decision the framework quantifies rather than presumes. The orchestration layer, in short, is where the economics of cheap-and-rising intelligence demand is actually transacted, and FMEC is a principled policy for transacting it.



## 16\. Conclusion

This paper has argued that the prevailing approach to assembling large-language-model systems — rank candidates by benchmark score and deploy the best — rests on a hidden and false premise: that standalone competence is a sufficient statistic for a model's value within a system. We have replaced that premise with a measurable quantity, the **Final Marginal Ensemble Contribution**, which decomposes a model's value into three estimable parts: its Shapley-style marginal contribution to ensemble utility (Eq. 7), its conditional information gain given the models already present (Eqs. 8–9), and its marginal contribution to correlated-failure risk (Eq. 12). The composite (Eq. 13) is then the objective of a constrained portfolio optimization (Eqs. 14–16) that selects and weights an ensemble subject to explicit cost, latency, and risk budgets.

The framework is, deliberately, a synthesis rather than an invention of components: it transposes Markowitz portfolio theory, Shapley attribution, and information-theoretic complementarity onto the model-selection problem, and it inherits from each the theoretical guarantees that make the synthesis rigorous — consistency and sample-complexity bounds for the estimator (Eqs. 19–21), the information-complementarity and correlated-failure theorems that formalize *when* ensembling helps (Eqs. 22–23), and the submodularity result that licenses greedy selection with a stated approximation guarantee (Section 6.9). We were careful to state that guarantee precisely: $(1-1/e)$ under cardinality constraints, and weaker, constraint-specific bounds under knapsack budgets and weak submodularity — a correction to the looser claims that sometimes accompany greedy ensemble methods.

We have not reported results. The contribution is the framework, its theory, and a pre-registered, falsifiable evaluation protocol (Section 11) that specifies in advance the hypotheses, estimators, multiplicity control, effect sizes, and — critically — the *numeric* conditions under which the framework should be rejected (Section 11.13): a non-inferiority margin tied to cost saving (Eq. 48a), a rank-discordance margin $\\Delta\_\\tau=0.20$, a recommendation-stability floor $\\mathrm{RSP}\\ge 0.90$, an adjudication-agreement floor $\\kappa\\ge 0.70$, and negligible-effect floors on Cohen's $d$ and Cliff's $\\delta$. Equally deliberate is the reproducibility posture. We separate computational reproducibility — made bit-exact by releasing the frozen outcome tensor, its published hash, the seeds, and every stage-wise intermediate, so the entire chain downstream of generation is an independently re-runnable function of released data — from empirical replicability, which is bounded by the mutating black-box endpoints that produce the data and which we therefore *instrument and bound* through a variance-components design (Eq. 20a) that measures generation and adjudication noise rather than ignoring it. The protocol is exercised by a hash-pinned reference instantiation (Section 12.1), and wherever a quantity can only come from executing it, the manuscript names it as an artifact-to-be-populated rather than inventing a value. That combination is the point: a model-selection methodology earns trust not by being clever but by being able to lose, by saying beforehand and *with numbers* what losing would look like, and by making the path to its own verification exact where it can be and honestly bounded where it cannot.

If the central hypothesis survives that test, the consequence for practice is concrete. The right question when building an LLM system is not *which model is best*, but *which portfolio of models delivers the most task utility per unit cost at acceptable risk* — and that is a question with a quantitative, reproducible, and risk-aware answer. FMEC is a proposal for how to compute it.



## 17\. Applications and Use-Cases

The framework targets any setting in which more than one capable model is available and the cost of choosing badly is non-trivial. We enumerate representative practical use-cases and then state a single all-encompassing one.

**Practical use-cases.**

1. **Production inference-gateway composition.** An organization operating an OpenAI-compatible gateway uses FMEC to decide *which* models enter its routing pool and with *what* weights, subject to per-request cost and latency SLOs (Eqs. 14, 36–37), replacing ad-hoc "use the top model" routing with a risk-budgeted portfolio.
2. **Cost-constrained enterprise assistant.** A firm with a fixed quality bar and a binding cost ceiling uses FMEC to assemble the cheapest ensemble that meets the bar on its own task mix, directly operationalizing the open-weight non-inferiority claim of H5 and the quality-adjusted-cost objective of Eq. (17).
3. **High-stakes verification pipelines.** In domains where correlated failure is catastrophic — quantitative finance, code that ships, compliance-adjacent reasoning — FMEC's risk term (Eqs. 10–12, Theorem 2) is used to select a panel whose members fail *differently*, paired with a verification-augmented fusion operator (Section 8.3) that grounds correctness in execution where possible.
4. **On-premises / data-residency-constrained deployment.** Where API models are precluded by privacy or residency requirements, FMEC selects an on-prem **open-weight** ensemble that approximates frontier quality on the deployment's tasks, again per H5.
5. **Model-portfolio procurement and budgeting.** Treating model subscriptions as portfolio assets, FMEC provides a quantitative basis for deciding which models are worth paying for — a model that tops a leaderboard but adds no marginal contribution to the existing portfolio (a redundancy exposed by Eqs. 7–9) is a procurement that should not be made.
6. **Benchmark-to-deployment auditing.** FMEC is used as a diagnostic to test, for a *specific* deployment, whether leaderboard rank actually predicts contribution (H4); when it does not, the audit prevents over-paying for benchmark-topping models that contribute nothing at the margin.
7. **Research instrument for ensemble composition.** As a measurement apparatus, FMEC quantifies complementarity and correlated failure across model families and scales, supporting empirical study of *when and why* ensembling LLMs helps.

**Singular all-encompassing use-case.**

> The FMEC framework is used by an LLM-orchestration system to select and weight a cost-, latency-, and risk-constrained ensemble of large language models — scoring each candidate by its final marginal ensemble contribution, the sum of its Shapley-style marginal utility and conditional information gain net of its marginal correlated-failure risk — so that the system deploys the model portfolio that maximizes expected task utility per unit cost rather than simply deploying the single highest-scoring model.



## Declaration on the Use of Generative AI and AI-Assisted Technologies

**Tool and provider.** During the preparation of this manuscript, the author used a generative artificial-intelligence assistant — Claude, a large language model developed by Anthropic (Claude Opus 4 model family) — as a writing and editing aid.

**Origin of the ideas.** The intellectual content of this paper originated with the author. The central thesis — that benchmark rank is an incomplete proxy for deployment value — together with the Final Marginal Ensemble Contribution and its constituent terms, the covariance-aware risk formulation, the constrained-optimization framing, and the experimental and statistical protocol, were all conceived by the author and existed, in substance, across several separate working documents before any AI tool was involved. The AI system did not originate the research problem, the framework, the central claims, or the conclusions.

**Nature and scope of AI assistance.** The author's starting materials were multiple disjointed and technically dense documents, each developing a different facet of the work. Generative AI was used to:

* synthesize and fuse these separate documents into a single, internally consistent manuscript with unified notation, structure, and narrative;
* critically review the author's arguments — surfacing gaps, ambiguities, unstated assumptions, and points requiring stronger justification — for the author to then address;
* assist in expressing the author's ideas in formal mathematical notation and in a professional scholarly register; and
* improve the readability, organization, and clarity of the exposition.

AI assistance was confined to the articulation, integration, and presentation of pre-existing ideas. It was not used to produce the scientific insights, to design the framework, or to draw the paper's conclusions, consistent with the principle that such tools should support — rather than replace — the author's core scholarly work.

**Verification and human oversight.** The author critically reviewed, edited, and validated all AI-assisted output. Every substantive claim, mathematical statement, and bibliographic reference was checked against primary sources; this included an independent verification of the complete reference list and a separate audit of the originality of the paper's core contribution, the results of which the author reviewed and confirmed. No AI-generated text, result, or citation was incorporated without author verification.

**Accountability.** The author takes full and sole responsibility for the entire content of this manuscript — including any portions developed with AI assistance — and for its accuracy, originality, and integrity. Consistent with the positions of the Committee on Publication Ethics (COPE) and the International Committee of Medical Journal Editors (ICMJE), a generative AI tool cannot accept responsibility for a work or be accountable for it and therefore does not meet the criteria for authorship; accordingly, no AI system is listed as an author or co-author. Records of the AI-assisted workflow are available from the author on reasonable request.

**Scope note.** This is a concept-and-methodology paper that reports no empirical results. AI was not used to generate, analyze, or fabricate data; the protocol of Sections 11–13 is specified for subsequent execution.



## Appendix A. Notation

The following table collects the principal symbols. Where a symbol is overloaded by convention, its disambiguation is noted.

|Symbol|Meaning|
|-|-|
|$M\_i$|Candidate model $i$, described by the tuple of Eq. (1)|
|$\\mu\_i, c\_i, l\_i, r\_i$|Model $i$'s quality (competence), unit cost, latency, and **reliability** score|
|$E\_i$|Model $i$'s error-mode indicator vector over the taxonomy of Eq. (2)|
|$n$|Number of models in the (qualified) candidate pool|
|$S$|A coalition (subset) of models, $S\\subseteq{1,\\dots,n}$|
|$v(S)$|Utility of coalition $S$ (Eq. 3)|
|$Q(S), C(S), L(S)$|Coalition quality, cost, and latency components of $v(S)$|
|$R(\\cdot)$|**Correlated-failure risk**; $R(S)$ at coalition level (Eq. 3), $R(\\mathbf{w})=\\mathbf{w}^\\top\\Sigma\\mathbf{w}$ at weight level (Eq. 11)|
|$\\alpha,\\beta,\\gamma,\\delta$|Utility preference weights on quality, risk, cost, latency (Eq. 3)|
|$\\Delta\_j(S)$|Marginal ensemble contribution of model $j$ to coalition $S$ (Eq. 4)|
|$\\mathrm{MEC}\_j$|Expected marginal ensemble contribution of $j$ (Eq. 5); $\\widehat{\\mathrm{MEC}}\_j$ its Monte Carlo estimate (Eq. 6)|
|$\\phi\_j$|Shapley value of model $j$ (Eq. 7)|
|$K$|Monte Carlo coalition-sample budget|
|$Y\_j, A\_j$|Model $j$'s response and outcome (correctness) random variables|
|$Z$|Ground-truth / target signal|
|$\\mathrm{IG}\_j$|Conditional information gain of $j$ (response-based Eq. 8; outcome-based Eq. 9)|
|$\\Sigma$|Error-mode covariance matrix (Eq. 10)|
|$\\mathbf{w}$|Ensemble weight vector on the simplex|
|$\\mathrm{MRC}\_j$|Marginal risk contribution $(\\Sigma\\mathbf{w})\_j$ (Eq. 12)|
|$\\eta,\\lambda$|FMEC combination coefficients on information gain and risk (Eq. 13)|
|$\\mathrm{FMEC}\_j$|Final marginal ensemble contribution of $j$ (Eq. 13)|
|$U(\\mathbf{w})$|Portfolio objective $\\sum\_i w\_i,\\mathrm{FMEC}\_i$ (Eq. 15)|
|$C\_{\\max},L\_{\\max},R\_{\\max}$|Cost, latency, and risk budgets (Eq. 14)|
|$\\mathrm{QAC}$|Quality-adjusted cost (Eq. 17)|
|$\\mathrm{RUG}$|Risk-adjusted utility gain (Eq. 18)|
|$S^\\star$|Oracle-optimal coalition (Eq. 28); regret defined in Eq. (29)|
|$s\_i$|Raw weighted-fusion score of model $i$ (Eq. 34); $q\_i, r\_i$ its quality and reliability, $a,b$ cost/latency exponents|
|$Q\_F, C\_F, T\_F$|Fused-system quality, cost, and latency (Eqs. 35–37)|
|$\\pi(a\\mid x)$|Router policy over actions $a$ given context $x$ (Eq. 40)|
|$r\_t$|Router reward at step $t$ (Eq. 42)|
|$B$|Bootstrap resample count|
|$R,,R\_J$|Generations per (model, task); judge replicates on the adjudication subsample (Section 11.8)|
|$\\sigma^2\_{\\mathrm{task}},\\sigma^2\_{\\mathrm{gen}},\\sigma^2\_{\\mathrm{adj}}$|Variance components: task-sampling, generation, adjudication (Eq. 20a)|
|$\\widehat{Q}\_{\\mathrm{maj}}(S)$|Simulated-fusion (majority-vote) quality surrogate for $Q(S)$ (Eq. 3a)|
|$c\_S$|Correct-count sufficient statistic $\\sum\_{i\\in S}A\_{S,i}$ for CMI conditioning (Eq. 9a)|
|$\\mathrm{RSP}$|Recommendation-Stability Probability (Eq. 51); reported stable iff $\\ge 0.90$|
|$\\mathrm{SelFreq}\_i$|Per-model bootstrap selection frequency (distinct from $\\mathrm{RSP}$; Section 9.3)|
|$\\Delta\_U,,\\Delta\_\\tau$|Pre-registered non-inferiority margin (Eq. 48a) and rank-discordance margin ($=0.20$)|
|$P\_{\\min}$|Qualification threshold (Section 11.2)|
|$\\varepsilon$|Price elasticity of intelligence demand (Eq. 52)|



## Appendix B. Complete Equation Index

This index enumerates **every** numbered equation in the paper, grouped by role, to support exhaustive tallying. The paper contains **54 numbered equations** (1–54) plus **six sub-tagged display equations** introduced for precision in this revision — the schematic **(33b)** and the amendment equations **(3a)** simulated-fusion quality, **(6a)** permutation Shapley estimator, **(9a)** outcome-based CMI with Miller–Madow correction, **(20a)** variance-components decomposition, and **(48a)** cost-tied non-inferiority margin — for **60 tagged display expressions in total**. Numbered-equation numbering is sequential and unique throughout; the sub-tagged equations attach to the nearest related numbered equation and are listed in their respective groups below.

**B.1 Methodology — the FMEC core (Eqs. 1–13).**

|Eq.|Role|
|-|-|
|1|Model descriptor tuple $M\_i=(\\mu\_i,c\_i,l\_i,r\_i,E\_i)$|
|2|Error-mode taxonomy vector $E\_i$|
|3|Coalition utility $v(S)=\\alpha Q-\\beta R-\\gamma C-\\delta L$|
|3a|Simulated-fusion quality $\\widehat{Q}\_{\\mathrm{maj}}(S)$ (majority-vote surrogate)|
|4|Marginal ensemble contribution $\\Delta\_j(S)=v(S\\cup{j})-v(S)$|
|5|Expected MEC over coalitions|
|6|Monte Carlo MEC estimator|
|6a|Permutation (unbiased Shapley) estimator $\\widehat{\\phi}\_j$|
|7|Shapley value $\\phi\_j$|
|8|Information gain (response-based CMI) $I(Y\_j;Z\\mid Y\_S)$|
|9|Information gain (outcome-based CMI) $I(A\_j;Z\\mid A\_S)$|
|9a|Outcome-based CMI plug-in with count-conditioning + Miller–Madow|
|10|Error-mode covariance $\\Sigma=\\mathrm{Cov}(E)$|
|11|Portfolio risk $R(\\mathbf{w})=\\mathbf{w}^\\top\\Sigma\\mathbf{w}$|
|12|Marginal risk contribution $\\mathrm{MRC}\_j=(\\Sigma\\mathbf{w})\_j$|
|13|**FMEC** $=\\widehat{\\mathrm{MEC}}\_j+\\eta,\\mathrm{IG}\_j-\\lambda,\\mathrm{MRC}\_j$|

**B.2 Constrained optimization and efficiency (Eqs. 14–18).**

|Eq.|Role|
|-|-|
|14|Constrained selection program (simplex + $C,L,R$ budgets)|
|15|Portfolio objective $U(\\mathbf{w})=\\sum\_i w\_i,\\mathrm{FMEC}\_i$|
|16|Expanded objective|
|17|Quality-adjusted cost $\\mathrm{QAC}=C/Q$|
|18|Risk-adjusted utility gain $\\mathrm{RUG}$|

**B.3 Theoretical analysis (Eqs. 19–31).**

|Eq.|Role|
|-|-|
|19|Estimator consistency (SLLN)|
|20|Estimator variance $\\sigma^2/K$|
|20a|Variance-components decomposition $\\sigma^2\_{\\mathrm{task}}+\\sigma^2\_{\\mathrm{gen}}+\\sigma^2\_{\\mathrm{adj}}$|
|21|Sample complexity $K=O(\\log(1/\\delta)/\\varepsilon^2)$ (Hoeffding)|
|22|Theorem 1 — information complementarity (chain rule)|
|23|Theorem 2 — correlated-failure risk|
|24|Proposition 1 — FMEC dominance|
|25|Generalization bound (Hoeffding), part 1|
|26|Generalization bound, part 2|
|27|Ranking consistency|
|28|Oracle-optimal coalition $S^\\star$|
|29|Oracle regret|
|30|Approximate additivity / recoverability|
|31|Decision-theoretic optimal action $a^\\star$|

**B.4 Algorithmic complexity (Eq. 32).**

|Eq.|Role|
|-|-|
|32|FMEC estimation complexity $O(nK+n^2 m)$|

**B.5 System architecture and serving (Eqs. 33, 33b, 34–43).**

|Eq.|Role|
|-|-|
|33|Four-subsystem orchestration schematic|
|33b|Fusion-operator schematic|
|34|Weighted-fusion score $s\_i$ and normalized weight $w\_i$|
|35|Fused quality $Q\_F$|
|36|Fused cost $C\_F=\\sum\_i C\_i + C\_J$|
|37|Fused latency $T\_F\\approx\\max\_i T\_i + T\_J$|
|38|Majority correctness $3p^2-2p^3$|
|39|Independent-verification reliability $1-(1-p)^k$|
|40|Router policy $\\pi(a\\mid x)$|
|41|Router objective|
|42|Router reward $r\_t$|
|43|Bandit action selection|

**B.6 Statistical evaluation (Eqs. 44–51).**

|Eq.|Role|
|-|-|
|44|H1 — utility dominance|
|45|H2 — information–contribution coupling|
|46|H3 — risk reduction|
|47|H4 — insufficiency of benchmark rank|
|48|H5 — open-weight non-inferiority|
|48a|Cost-tied non-inferiority margin $\\Delta\_U$|
|49|Cohen's $d$|
|50|Cliff's $\\delta$|
|51|Recommendation-Stability Probability|

**B.7 Economic context (Eqs. 52–54).**

|Eq.|Role|
|-|-|
|52|Constant-elasticity demand $Q=A,P^{-\\varepsilon}$|
|53|Revenue / spend $R=P,Q=A,P^{1-\\varepsilon}$|
|54|Jevons-style compute-demand elasticity $d\\ln D/d\\ln g>1$|

The methodology equations (B.1) and the constrained-optimization equations (B.2) constitute the *defining* equations of the framework; the theory (B.3) supports them; the architecture (B.5), evaluation (B.6), and economic (B.7) groups govern, respectively, serving, validation, and motivation.



## Appendix C. Prompt Templates and Router Reference Implementation

This appendix gives the frozen prompt templates referenced throughout the paper (Sections 8, 10.2, 11.5, 12) and a reference implementation of the optional contextual-bandit router (Section 8.5). The templates are the *exact* strings an implementation should version and persist; placeholders in braces are filled at call time. They are written to be model-agnostic and to elicit schema-validated outputs.

### C.1 Panel / candidate-generation system prompt (`panel\\\_system`)

Used at candidate-generation temperature (Section 10.1) to elicit an independent solution from each panel model.

```text
You are an expert problem-solver participating in a multi-model panel. You will be
given a task. Produce your single best, self-contained answer.

Requirements:
- Reason carefully and show the key steps of your reasoning before the final answer.
- Do not assume other panelists' answers; solve the task independently.
- If the task is verifiable (code, math, structured output), make your final answer
  precise and machine-checkable.
- Clearly delimit your final answer using the exact markers:
  <final\\\_answer> ... </final\\\_answer>
- If you are uncertain, state the uncertainty explicitly inside the reasoning; do not
  fabricate facts, citations, APIs, or results.

Task:
{task\\\_prompt}
```

### C.2 Adjudicator / judge system prompt (`judge\\\_system`)

Used at adjudication temperature `0.0` (deterministic) both to score correctness and to assign exactly one error category from the eight-category taxonomy (Section 11.5). Output is strict JSON validated against the schema below; any non-conforming output is rejected and re-requested.

```text
You are a strict, impartial adjudicator. You will be given (1) a task, (2) a reference
solution or verifiable criterion, and (3) a candidate answer. Judge the candidate
answer ONLY against the task and the reference/criterion. Do not reward fluency,
confidence, or verbosity.

Decision procedure:
1. Determine whether the candidate answer is CORRECT (fully satisfies the task and
   matches the reference / passes the criterion) or INCORRECT.
2. If INCORRECT, assign EXACTLY ONE primary error category from this closed set:
   - "factual"            : a verifiably false assertion
   - "reasoning"          : invalid inference from correct premises
   - "arithmetic"         : numerical or symbolic miscalculation
   - "instruction"        : failure to follow explicit constraints or output format
   - "code"               : code that fails tests, raises, or does not compile
   - "hallucination"      : invented citations, APIs, entities, or data
   - "incompleteness"     : truncated or partial answer; dropped sub-questions
   - "safety"             : unwarranted refusal or policy-inappropriate content
3. Provide a brief, specific justification grounded in the task and reference.
4. Report a calibrated confidence in \\\[0,1].

Output requirements:
- Respond with a SINGLE JSON object and NOTHING else (no prose, no code fences).
- Conform exactly to this schema:
  {
    "is\\\_correct": <boolean>,
    "error\\\_category": <one of the eight strings above, or null if is\\\_correct is true>,
    "confidence": <number between 0 and 1>,
    "justification": <string, <= 60 words>
  }

Task:
{task\\\_prompt}

Reference / criterion:
{reference}

Candidate answer:
{candidate\\\_answer}
```

A *fusion* variant of the adjudicator (used by the judge-based fusion operator of Section 8.3 rather than by evaluation) replaces the correctness decision with a selection among $k$ candidate answers, returning `{"winner\\\_index": <int>, "justification": <string>}` under the same strict-JSON discipline. The two variants share the impartiality and no-fabrication constraints; only the decision target differs.

### C.3 Structured-solver system prompt (`structured\\\_solver`)

Used at solving temperature `0.1` for tasks requiring a typed, machine-parseable result (e.g., the `candidate\\\_solution` schema of Section 9.1).

```text
You are an expert solver that returns STRICTLY structured output. Solve the task and
return your result as a single JSON object conforming exactly to the provided schema.

Rules:
- Output ONLY the JSON object; no surrounding prose or code fences.
- Populate every required field; use null only where the schema permits it.
- Place any free-form reasoning in the "reasoning" field, not outside the JSON.
- For code tasks, the "code" field must contain a complete, runnable solution with no
  placeholders.
- Do not invent values to satisfy the schema; if a value is genuinely unknown, follow
  the schema's null/optional convention and explain in "reasoning".

Schema:
{json\\\_schema}

Task:
{task\\\_prompt}
```

### C.4 Contextual-bandit router (reference implementation)

A reference $\\epsilon$-greedy contextual-bandit router corresponding to Eqs. (40)–(43) and the `router` configuration block (Section 10.2). It is illustrative scaffolding for the *optional* learned-routing layer, not a required component of FMEC selection. Reward design follows Eq. (42): quality net of cost and risk penalties.

```python
import math
import random
from collections import defaultdict
from typing import Hashable, Sequence

class EpsilonGreedyRouter:
    """Contextual epsilon-greedy router over a fixed set of model 'arms'.

    Context is a discrete feature key (e.g., task category). Each (context, arm)
    pair maintains an incremental mean reward. Reward internalizes cost and risk
    per Eq. (42): r = quality - lambda\\\*risk - mu\\\*cost - gamma\\\*latency.
    """

    def \\\_\\\_init\\\_\\\_(self, arms: Sequence\\\[Hashable], epsilon: float = 0.10,
                 reward\\\_weights: dict | None = None, seed: int = 20260618):
        self.arms = list(arms)
        self.epsilon = float(epsilon)
        # Eq. (42) reward weights; gamma (latency) optional, defaults to 0.
        self.w = reward\\\_weights or {"lambda": 0.50, "mu": 0.02, "gamma": 0.00}
        self.\\\_rng = random.Random(seed)
        self.\\\_counts: dict = defaultdict(lambda: {a: 0 for a in self.arms})
        self.\\\_means: dict = defaultdict(lambda: {a: 0.0 for a in self.arms})

    def select(self, context: Hashable) -> Hashable:
        """Eq. (43): explore with prob. epsilon, else exploit the argmax mean."""
        if self.\\\_rng.random() < self.epsilon:
            return self.\\\_rng.choice(self.arms)
        means = self.\\\_means\\\[context]
        best = max(means.values())
        # random tie-break among argmax arms
        top = \\\[a for a, m in means.items() if math.isclose(m, best)]
        return self.\\\_rng.choice(top)

    @staticmethod
    def reward(quality: float, cost: float, risk: float, latency: float,
               w: dict) -> float:
        """Eq. (42): scalar reward internalizing cost and risk."""
        return (quality
                - w.get("lambda", 0.0) \\\* risk
                - w.get("mu", 0.0) \\\* cost
                - w.get("gamma", 0.0) \\\* latency)

    def update(self, context: Hashable, arm: Hashable, reward: float) -> None:
        """Incremental mean update for the (context, arm) estimate."""
        self.\\\_counts\\\[context]\\\[arm] += 1
        n = self.\\\_counts\\\[context]\\\[arm]
        m = self.\\\_means\\\[context]\\\[arm]
        self.\\\_means\\\[context]\\\[arm] = m + (reward - m) / n
```

The router's role is strictly subordinate to FMEC selection: FMEC determines *which models are in the pool and their portfolio weights*; the router, when enabled, performs *per-request* arbitration within that pool. Disabling the router (`router.enabled = False`, the default) recovers pure FMEC-weighted serving.



## Appendix D. Discipline–Skill–Section Matrix

A reformatting of Section 14 into a compact competency matrix. Each row is a discipline; the middle column lists the operative skills; the right column names the governing sections and equations.

|Discipline|Operative skills|Governing sections / equations|
|-|-|-|
|Mathematics of finance \& portfolio theory|Mean–variance optimization; efficient frontier; risk budgeting; marginal-risk reading|§5; Eqs. 11–12, 14–17|
|Cooperative game theory|Coalitional value; marginal contribution; Shapley axioms; LOO approximation|§4; Eqs. 4–7, 30; §6.9|
|Information theory|Entropy; mutual information; conditional MI; chain rule; CMI estimation|§4; Eqs. 8–9; Thm 1 (Eq. 22)|
|Monte Carlo \& computational statistics|MC estimator design; convergence; sample complexity; nonparametric bootstrap; BCa|§4, §6; Eqs. 5–6, 19–21; §11.8|
|Robust multivariate statistics|Sample-covariance instability; shrinkage (Ledoit–Wolf); block structure|§4; Eq. 10; A5|
|Submodular \& constrained optimization|Monotone submodularity; greedy; constraint-specific guarantees|§6.9; Alg. 2; Eq. 14|
|Statistical inference \& experimental design|Pre-registration; one-sided / paired / non-inferiority tests; rank correlation; Holm–Bonferroni; effect sizes|§11; Eqs. 44–50|
|LLM systems \& API engineering|Gateway integration; structured outputs; decoding control; cost/latency reasoning; sandboxing|§8, §10; Eqs. 34–37; §8.3|
|Reinforcement learning / contextual bandits|Explore–exploit; $\\epsilon$-greedy; reward design; online update|§8.5; Eqs. 40–43; App. C.4|
|Reproducible research \& SW engineering|Seeding; version manifesting; deterministic adjudication; artifact persistence; rubric/prompt engineering|§10.2, §12; §11.5; App. C|



## Appendix E. References

References follow the Harvard (Cite Them Right) author–date convention. In-text citations appear as (Author, Year); entries are listed alphabetically by author surname below.

Breiman, L. (1996) 'Bagging predictors', *Machine Learning*, 24(2), pp. 123–140.

Breiman, L. (2001) 'Random forests', *Machine Learning*, 45(1), pp. 5–32.

Castro, J., Gómez, D. and Tejada, J. (2009) 'Polynomial calculation of the Shapley value based on sampling', *Computers \& Operations Research*, 36(5), pp. 1726–1730.

Chen, L., Zaharia, M. and Zou, J. (2023) 'FrugalGPT: how to use large language models while reducing cost and improving performance', arXiv \[preprint]. Available at: https://arxiv.org/abs/2305.05176 (Accessed: 18 June 2026).

Chen, M. et al. (2021) 'Evaluating large language models trained on code', arXiv \[preprint]. Available at: https://arxiv.org/abs/2107.03374 (Accessed: 18 June 2026).

Chiang, W.-L. et al. (2024) 'Chatbot Arena: an open platform for evaluating LLMs by human preference', *International Conference on Machine Learning (ICML 2024)*.

Cliff, N. (1993) 'Dominance statistics: ordinal analyses to answer ordinal questions', *Psychological Bulletin*, 114(3), pp. 494–509.

Cobbe, K. et al. (2021) 'Training verifiers to solve math word problems', arXiv \[preprint]. Available at: https://arxiv.org/abs/2110.14168 (Accessed: 18 June 2026).

Cohen, J. (1960) 'A coefficient of agreement for nominal scales', *Educational and Psychological Measurement*, 20(1), pp. 37–46.

Cohen, J. (1988) *Statistical Power Analysis for the Behavioral Sciences*. 2nd edn. Hillsdale, NJ: Lawrence Erlbaum Associates.

Cover, T.M. and Thomas, J.A. (2006) *Elements of Information Theory*. 2nd edn. Hoboken, NJ: Wiley-Interscience.

Das, A. and Kempe, D. (2011) 'Submodular meets spectral: greedy algorithms for subset selection, sparse approximation and dictionary selection', *Proceedings of the 28th International Conference on Machine Learning (ICML)*, pp. 1057–1064.

Dietterich, T.G. (2000) 'Ensemble methods in machine learning', in *Multiple Classifier Systems (MCS), Lecture Notes in Computer Science 1857*. Berlin: Springer, pp. 1–15.

Du, Y., Li, S., Torralba, A., Tenenbaum, J.B. and Mordatch, I. (2023) 'Improving factuality and reasoning in language models through multiagent debate', arXiv \[preprint]. Available at: https://arxiv.org/abs/2305.14325 (Accessed: 18 June 2026).

Efron, B. (1987) 'Better bootstrap confidence intervals', *Journal of the American Statistical Association*, 82(397), pp. 171–185.

Efron, B. and Tibshirani, R.J. (1993) *An Introduction to the Bootstrap*. New York: Chapman \& Hall/CRC.

Freund, Y. and Schapire, R.E. (1997) 'A decision-theoretic generalization of on-line learning and an application to boosting', *Journal of Computer and System Sciences*, 55(1), pp. 119–139.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D. and Steinhardt, J. (2021a) 'Measuring massive multitask language understanding', *International Conference on Learning Representations (ICLR 2021)*.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D. and Steinhardt, J. (2021b) 'Measuring mathematical problem solving with the MATH dataset', *Advances in Neural Information Processing Systems: Datasets and Benchmarks Track (NeurIPS 2021)*.

Hoeffding, W. (1963) 'Probability inequalities for sums of bounded random variables', *Journal of the American Statistical Association*, 58(301), pp. 13–30.

Holm, S. (1979) 'A simple sequentially rejective multiple test procedure', *Scandinavian Journal of Statistics*, 6(2), pp. 65–70.

Jiang, D., Ren, X. and Lin, B.Y. (2023) 'LLM-Blender: ensembling large language models with pairwise ranking and generative fusion', *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 14165–14178.

Jiménez, C.E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O. and Narasimhan, K. (2024) 'SWE-bench: can language models resolve real-world GitHub issues?', *International Conference on Learning Representations (ICLR 2024)*.

Khuller, S., Moss, A. and Naor, J. (1999) 'The budgeted maximum coverage problem', *Information Processing Letters*, 70(1), pp. 39–45.

Kim, M., Ray, E.L. and Reich, N.G. (2024) 'Beyond forecast leaderboards: measuring individual model importance based on contribution to ensemble accuracy', arXiv \[preprint]. Available at: https://arxiv.org/abs/2412.08916 (Accessed: 18 June 2026).

Kuncheva, L.I. and Whitaker, C.J. (2003) 'Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy', *Machine Learning*, 51(2), pp. 181–207.

Ledoit, O. and Wolf, M. (2004) 'A well-conditioned estimator for large-dimensional covariance matrices', *Journal of Multivariate Analysis*, 88(2), pp. 365–411.

Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J. and Glance, N. (2007) 'Cost-effective outbreak detection in networks', *Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, pp. 420–429.

Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems (NeurIPS 2017)*.

Markowitz, H. (1952) 'Portfolio selection', *The Journal of Finance*, 7(1), pp. 77–91.

Miller, G.A. (1955) 'Note on the bias of information estimates', in H. Quastler (ed.) *Information Theory in Psychology: Problems and Methods*. Glencoe, IL: Free Press, pp. 95–100.

Nemhauser, G.L., Wolsey, L.A. and Fisher, M.L. (1978) 'An analysis of approximations for maximizing submodular set functions—I', *Mathematical Programming*, 14(1), pp. 265–294.

Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J.E., Kadous, M.W. and Stoica, I. (2024) 'RouteLLM: learning to route LLMs with preference data', arXiv \[preprint]. Available at: https://arxiv.org/abs/2406.18665 (Accessed: 18 June 2026).

Rein, D., Hou, B.L., Stickland, A.C., Petty, J., Pang, R.Y., Dirani, J., Michael, J. and Bowman, S.R. (2023) 'GPQA: a graduate-level Google-proof Q\&A benchmark', arXiv \[preprint]. Available at: https://arxiv.org/abs/2311.12022 (Accessed: 18 June 2026).

Rozemberczki, B. and Sarkar, R. (2021) 'The Shapley value of classifiers in ensemble games', *Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM)*, pp. 1558–1567.

Shapley, L.S. (1953) 'A value for n-person games', in H.W. Kuhn and A.W. Tucker (eds.) *Contributions to the Theory of Games II*. Princeton, NJ: Princeton University Press, pp. 307–317.

Sviridenko, M. (2004) 'A note on maximizing a submodular set function subject to a knapsack constraint', *Operations Research Letters*, 32(1), pp. 41–43.

Wagenmakers, E.-J., Wetzels, R., Borsboom, D., van der Maas, H.L.J. and Kievit, R.A. (2012) 'An agenda for purely confirmatory research', *Perspectives on Psychological Science*, 7(6), pp. 632–638.

Walker, E. and Nowacki, A.S. (2011) 'Understanding equivalence and noninferiority testing', *Journal of General Internal Medicine*, 26(2), pp. 192–196.

Wang, J., Wang, J., Athiwaratkun, B., Zhang, C. and Zou, J. (2024) 'Mixture-of-Agents enhances large language model capabilities', arXiv \[preprint]. Available at: https://arxiv.org/abs/2406.04692 (Accessed: 18 June 2026).

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A. and Zhou, D. (2023) 'Self-consistency improves chain of thought reasoning in language models', *International Conference on Learning Representations (ICLR 2023)*.

Wolpert, D.H. (1992) 'Stacked generalization', *Neural Networks*, 5(2), pp. 241–259.



*End of paper.*

