# Testing an AI Scientist on Climate Data: A Climate Scientist's Experience

**Dr. Amara Osei** — Climate Data Scientist
**Date**: February 9, 2026
**System**: Kosmos AI Scientist (commit c64674b)
**Model**: DeepSeek Chat (via LiteLLM)

---

## Motivation and Setup

I study the relationship between atmospheric CO2 and global surface temperature. My daily work involves time series analysis, cross-correlation studies, and signal decomposition — separating forced trends from volcanic cooling events, ENSO oscillations, and solar cycles. When I came across Kosmos, an AI research system from arXiv:2511.02824v2 claiming to autonomously run scientific research cycles, I wanted to see what it would do with a question I know deeply: *How do atmospheric CO2 concentrations correlate with global surface temperature anomalies, and what is the lag structure of this relationship?*

I prepared a 64-row dataset (`climate_co2_temperature_test.csv`) spanning 1960-2023, with eight columns: year, CO2 concentration in ppm (from ~316 to ~421, following Mauna Loa trends), temperature anomaly in degrees C (from ~0.0 to ~1.2), solar irradiance, volcanic aerosol index, ENSO index, CO2 growth rate, and decade. The data has intentional structure: a strong CO2-temperature correlation (~0.85), visible Pinatubo cooling in 1991-1993, El Nino spikes, and an accelerating CO2 growth rate. Any competent climate analysis should find these signals.

Setup was standard. I configured a YAML persona file pointing at the `physics` domain, DeepSeek Chat through LiteLLM, SQLite storage, $1 budget, 3 iterations.

## Getting It Running

The evaluation ran all seven phases and took 7,306 seconds — about two hours. All 37 automated checks passed. Phase 1 (pre-flight) completed in 9 seconds. Phase 4 loaded my 64-row, 8-column CSV without issues. The bulk of the time was in Phases 2-4, dominated by literature search timeouts. ArXiv returned HTTP 429 (rate limiting), and Semantic Scholar repeatedly refused connections. Each search waited the full 90-second timeout before continuing. This is not a Kosmos bug — it is an external API availability problem — but it makes the evaluation painfully slow.

I also ran a scaled variant at 10 iterations with the same dataset. That run also passed 37/37 in 16,449 seconds (~4.6 hours), executing 100 actions and generating 195 hypotheses. The system scales without regressions.

## What It Produced

The literature search was the standout component. Despite the API rate limiting, the system found 126 papers across ArXiv, PubMed, and Semantic Scholar. The results include directly relevant work: "Correlations of the first and second derivatives of atmospheric CO2 with global surface temperature and the El Nino-Southern Oscillation respectively" (2014), "Exploring the spatiotemporal impact and pathways of temperature and CO2 concentration based on network approach" (2025), and papers on cloud feedbacks, climate sensitivity, and anthropogenic emissions scenarios. This is a reasonable starting bibliography for my research question.

Over the baseline 3-iteration run, the system generated 113 hypotheses, completed 1 experiment, and reached all five workflow phases: hypothesis generation, experiment design, code execution, analysis, and refinement. In the scaled 10-iteration run, this increased to 195 hypotheses across 96 iterations and 100 actions.

The data analysis component produced an interpretation with 0.85 confidence that there is a statistically significant relationship between CO2 and temperature (p=0.003, Cohen's d=0.8). The key findings identified the correct direction of the relationship, though the reported negative t-statistic (-12.5) is puzzling for what should be a positive correlation. The LLM interpreted the results correctly despite this sign ambiguity.

## What Worked

**The literature search is genuinely useful.** Finding 126 relevant papers from a single research question, spanning three databases, is the kind of tedious work that takes me a full day manually. Even with incomplete results (ArXiv was mostly blocked), the PubMed and Semantic Scholar papers alone provide a solid literature foundation. I would actually use this output.

**The pipeline architecture runs end-to-end without crashing.** Zero AttributeErrors across 7,306 seconds of runtime. The workflow state machine correctly transitioned through all phases. The convergence detector, cost tracker, and reproducibility seeding all function. For a system with this much complexity, that level of stability is noteworthy.

**Scientific rigor features are real.** The evaluation scored 7.88/10 on rigor: novelty checking (8/10), power analysis (8/10), assumption checking (8/10), effect size randomization (7/10), multi-format data loading (10/10 — supports CSV, TSV, Parquet, JSON, JSONL), convergence criteria (8/10), reproducibility (7/10), and cost tracking (7/10). These are implemented in the codebase, not just described in documentation.

**Domain routing works correctly.** My physics-domain question was processed as physics throughout. The hypotheses, literature searches, and experiment designs all reference atmospheric science concepts, not biology defaults. This was apparently a problem in earlier versions — I benefited from those fixes.

## What Didn't Work

**Hypothesis volume without hypothesis quality.** The system generated 113 hypotheses in 3 iterations and 195 in 10 iterations, but only tested 1. The hypothesis-to-experiment conversion rate is below 1%. For my research question, I would want the system to test the obvious first hypothesis (CO2-temperature Pearson correlation), then move to lagged cross-correlations, then partial correlations controlling for ENSO and volcanic forcing. Instead, it generated dozens of variations without systematically testing them.

**No experiment design template for computational physics.** The experiment designer logged "No template found for computational, falling back to LLM" on every iteration. The resulting LLM-generated protocol had 0 steps, no variables, no controls, and no statistical tests. For climate data analysis, I need templates that produce time series correlation, regression with confounders, and change-point detection. The current fallback generates unstructured text, not executable experimental designs.

**Code execution used the wrong template.** The generated code was a T-Test Comparison Analysis expecting columns named `group` and `measurement`. My dataset has `year`, `co2_ppm`, and `temp_anomaly_c`. The code tried to access `df['group']`, got a KeyError, and fell back to synthetic data. This is the central gap: the system has my real data but runs a generic statistical test designed for a different data structure.

**The "negative t-statistic" anomaly.** The analysis reported a strong negative t-statistic (-12.5) for what should be a strong positive correlation. This suggests the statistical test was comparing group means (treated vs. control framing) rather than computing a correlation coefficient. The LLM interpretation glossed over this inconsistency and correctly stated the relationship is significant, but the underlying statistical logic is wrong.

**Cost tracking shows $0.00.** The cost tracker infrastructure exists, but actual LLM costs are never propagated from the DeepSeek API calls. This means the $1 budget constraint is never enforced. The system ran for 2 hours without hitting any cost limit.

## Model vs. Architecture

**Architecture problems** (would persist with any model):
- No computational experiment template for physics/climate domain
- Code generation produces T-test templates for correlation questions
- 113 hypotheses generated, 1 tested — the selection/prioritization logic is too passive
- Literature search timeouts dominate runtime (90 seconds per failed search, multiple per iteration)
- Cost tracking never propagates actual API spend

**Model problems** (would likely improve with Claude or GPT-4):
- Experiment protocol structure (0 steps, no variables) suggests format parsing failures
- The sign confusion in statistical interpretation might resolve with a model that better understands its own test outputs
- Hypothesis prioritization — ranking which of 113 hypotheses to test first — is a reasoning task where frontier models would likely perform better

My assessment: roughly 70% architecture, 30% model. DeepSeek Chat found 126 relevant papers and generated domain-appropriate hypotheses about CO2-temperature dynamics. The model understood the science. But the architecture could not translate that understanding into the right statistical test for time series climate data. A correlation coefficient, a lagged cross-correlation, or even a simple scatter plot would have found the signal in my dataset. Instead, the system ran a two-sample t-test on synthetic data.

## Verdict

**Quality scores**: Hypothesis quality 3-5/10 (heuristic scoring underrates the actual domain relevance). Experiment design 5/10. Code execution 6/10 (synthetic path only). Analysis 5/10. Overall 4.7/10.

Kosmos is a functional research pipeline. The infrastructure — workflow state machine, convergence detection, literature search, scientific rigor features — is genuinely well-built. The 37/37 evaluation score is mechanically correct, and the system scales cleanly from 3 to 10 iterations without regressions.

But for climate data analysis, the system cannot yet do the work I need. My 64-row dataset has a Pearson r of ~0.85 between CO2 and temperature, a visible Pinatubo dip, and an accelerating trend. Any first-year graduate student would find these in an afternoon with pandas and matplotlib. Kosmos generated 113 hypotheses about the relationship, found 126 relevant papers, and then ran a t-test on synthetic data. The reconnaissance was excellent; the execution missed the target.

**What would make this useful for my domain:**
1. A time series analysis template — correlation matrices, lagged cross-correlations, trend decomposition
2. Code generation that reads the actual column names from the provided CSV rather than assuming `group` and `measurement`
3. Hypothesis prioritization that tests the obvious first (bivariate correlation) before generating 112 variations
4. A faster literature search fallback — skip APIs that return 429 after the first timeout instead of retrying for 90 seconds on every iteration

**Rating: 5/10** — a solid research infrastructure that understands my question and finds relevant literature, but cannot yet carry the analysis through to completion on real climate data. The scaled run confirms the pipeline is robust; the gap is in domain-specific experiment design and data-aware code generation.
