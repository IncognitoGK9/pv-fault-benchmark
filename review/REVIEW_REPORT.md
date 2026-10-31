# Hostile review: *Benchmarking Classical Machine Learning Models for Shading-Fault Detection in Rooftop PV Systems Using Real Sensor Data* (PV_Fault_Detection_Final_v2.docx)

Reviewed 31 Oct 2026, against the code, the `results/` files, `data/reunion_subsample.csv` and `data/subsample_metadata.json` at commit `14e38a1`. I did not re-run the 7.4 h benchmark. The new numbers below come from `review/reanalysis.py`, which writes `review/reanalysis_results.json`. They are cheap post-hoc checks that use the stored results, or a fixed Random Forest under the paper's own outer CV.

**Overall verdict.** Every table value matches the stored results. The prose has four factual errors. Three of the paper's central claims do not hold up against its own data:

1. The "two-tier" ranking.
2. Va as the physical shading signal.
3. The cross-model SHAP summation.

On top of that, the no-fault class is sampled under different operating conditions from the fault class. That confound was not disclosed, and it is measurable. An arXiv preprint can go out once the claims are re-calibrated and the confound is disclosed (that is what the `arxiv/` version does). A journal resubmission should first re-run with matched no-fault sampling (R1 below).

---

## A. Major issues (these change the conclusions)

### M1. The no-fault class is confounded with time of day and operating level (not disclosed)
* Every fault slice is 400 rows (33–53 min) in the middle of the day (10:00–17:00, median hour 12.6). Each no-fault slice is a random run of 2,000 consecutive rows, so it covers several hours. Five of the seven no-fault slices include dawn or dusk (earliest 05:51, latest 18:11). The fault class has an hour s.d. of 1.7 h; the no-fault class has 3.7 h.
* Median `Pg` is 274 W for no-fault and 982 W for fault. Median `Va` is 184 V for no-fault and 151 V for fault.
* `Eg` is the **daily cumulative energy counter**: it resets at midnight and rises through the day (checked in the subsample). It therefore encodes time of day and irradiance history, not shading. Yet it ranks second for DT, kNN and XGBoost.
* **Probe** (Random Forest with the fixed modal configuration, same 3×5 StratifiedGroupKFold, no SMOTE):

  | Features | Accuracy | MCC | AUC |
  |---|---|---|---|
  | clock hour only | 0.732 ± .114 | 0.443 ± .260 | 0.792 |
  | `Eg` only | 0.616 | 0.196 | 0.735 |
  | `Va` only | **0.880 ± .075** | **0.724 ± .158** | 0.913 |
  | 6 raw channels without `Eg` | **0.888 ± .084** | **0.744 ± .185** | 0.922 |
  | Paper's tuned 29-feature RF + SMOTE (Table 7) | 0.838 ± .156 | 0.526 ± .416 | 0.859 |

  Clock time alone, with no electrical information, beats kNN, the MLP and the SVM on MCC under the primary condition. A single feature (`Va`) beats every tuned 29-feature model. So the benchmark mostly measures how well each model copes with redundant or confounded inputs and with SMOTE. It is not a clean measure of shading discrimination.
* The paper's physical explanation (bypass diodes, MPPT shifting to a lower voltage) is plausible, but it cannot be separated from the fact that module voltage falls as temperature and irradiance rise at midday. The no-fault rows are cooler, lower-irradiance rows.
* **Required for a journal:** re-sample no-fault rows from the same clock-time window, and ideally the same irradiance and temperature band (the companion meteorological file allows this). Drop `Eg` or convert it to a rate. Re-run.
* **For arXiv:** disclose the confound and the probe as a limitation. Done in the `arxiv/` version (Section 5.6).

### M2. The "two-tier" structure is not supported by the tests
* Logistic Regression is not significantly different from kNN, the SVM or the MLP (accuracy p = .055, .055, .277; MCC p = .062, .110, .286). The Decision Tree is not significantly different from the SVM or the MLP on either metric. Only the two ensembles separate from the lower group.
* Failing to reject a difference does not show equivalence. "LR/DT are statistically indistinguishable from the ensembles" is absence of evidence, not evidence of parity. No equivalence test (e.g. TOST) was run.
* The defensible claim is narrower: the ensembles beat kNN and the MLP, and LR and DT cannot be separated from either group.

### M3. No multiple-comparison control, and the paper says the opposite
* The paper tests and reports all 21 pairs per metric (Table 10). Yet §5.9 says testing "is limited to the comparisons the research questions require, which avoids … a multiple-comparisons correction", and §2 cites García & Herrera (2008) for exactly that requirement. Both statements are false for this design.
* Holm-adjusted over the 21 pairs (`review/reanalysis_results.json`):
  * **Accuracy:** only RF > MLP (p_Holm = .001) and RF > kNN (.009) survive. XGB > kNN goes to .17 and XGB > MLP to .18.
  * **MCC:** nothing survives. The smallest p_Holm is .070 (RF and XGB versus kNN).
* The 15 folds are not independent, because the three repeats re-use the same 17 episodes. A Wilcoxon test on them treats them as independent and is anti-conservative. Use the Nadeau–Bengio corrected resampled t-test, or test on one repeat, or use a Friedman test with post-hoc analysis (Demšar 2006).
* **Per-fold values were not saved** (`summary_metrics.json` keeps only mean and s.d.), so no reader can recompute any test. `run_experiment.py` now writes `per_fold_metrics.json` on future runs.

### M4. The cross-model SHAP comparison is invalid as reported
* Units are mixed across explainers. `LinearExplainer` (LR) and `TreeExplainer` on XGBoost return **log-odds**; RF and DT return **probability**; the sampling explainer for kNN, SVM and MLP returns probability. The total mean |SHAP| is 40.8 for LR, 14.1 for XGB and 0.58–1.07 for the rest. Fig. 3 plots these raw magnitudes, so five models are invisible. The "summed across classifiers" ranking is just LR plus XGB. The paper's claim that SHAP is "defined identically across model families" holds only within a common output scale.
* **The explained models are not the evaluated models.** `shap_analysis.py` calls `build_pipeline()`, which uses **default** hyperparameters (LR C = 1, an unpruned DT, RF with 100 fully grown trees, default XGB). It does not use the tuned or modal ones. The fit is on all 20,800 rows, so the attributions are in-sample. None of this is disclosed.
* kNN, SVM and MLP are explained on 300 rows; the other models are explained on all rows.
* `V_g − V_a` is almost exactly `−V_a + 233` (the grid voltage is nearly constant). kNN's top feature is therefore `V_a` information under another name.
* **Fix applied:** each model's attribution is normalised to a share of its total (`review/fig_shap_normalised.py` → `figures/fig_shap_normalised.png`). That ordering is `Va` 0.23, `Eg` 0.09, `V_g − V_a` 0.08, then rolling `I_a`/`P_g` means. The disclosure of default hyperparameters goes in the text. **Journal:** recompute SHAP on the modal configurations, out of fold, on one output scale.
* `visualize.py` does **not** produce the paper's SHAP figure. It still draws the legacy LR-coefficient/DT-importance chart. The README says otherwise.

### M5. The per-fault-type analysis treats episodes as samples; its explanations are wrong
* Per-code "recall" is a per-episode hit rate. Each episode is classified almost all-or-nothing (Random Forest, share of rows correct, `reanalysis_results.json`):
  * Code 1.0: episode 7 = 1.00, episode 9 = 0.00, giving "0.501".
  * Code 4.0: episodes 11, 17, 21 ≈ 1.0; episodes 19 and 23 ≈ 0.00, giving "0.595".
  * Code 2.3: its single episode = 0.007.

  Reporting 0.501 to three decimals implies a precision that n = 2 episodes cannot support. Report hits as *k / n episodes*.
* §5.8 blames the code-2.3 failure on sparsity (3,287 rows). This is contradicted by the design. Every fault episode contributes exactly 400 rows, and code 2.1 has fewer raw rows (2,148) yet is detected perfectly. The data show the actual reason. Episode 5 (code 2.3) was recorded at low output (median `Pg` 217 W) with `Va` = 181 V, which is inside the no-fault `Va` range (median 184 V). The missed code-4.0 episode 19 matches the same pattern (late afternoon, `Pg` 179 W, `Va` 169 V). The missed episode 23 has near-normal `Va` (176 V). Both uniform-shading episodes inject zero power. The detected one (episode 7) also has `Va` = 0, while the missed one (episode 9) holds `Va` ≈ 150 V, a value typical of fault rows at full output.
* Intermittent episodes run up to 22,988 rows. Keeping a random 400-row slice (2–15 % of the episode) may capture unshaded intervals that still carry the fault label. This is label noise caused by subsampling, and it should be stated.
* Tables 12–14 use a **fixed modal configuration chosen from the outer-fold searches**, so outer-test data informed it. Tables 13 and 14 do not disclose this. It is mildly optimistic, and it means Table 14's 0.768 (pooled) cannot be compared with the 0.741 fold mean in Table 11.

### M6. The feature ablation does not isolate what it claims
* Configuration B ("temporal only") contains **no `V_a`**, because the rolling features cover `P_g`, `I_a`, `V_g` and `F_g` only. B's 16.2-point MCC loss therefore measures removing `V_a`, not "temporal versus raw".
* The four instantaneous derived features are never ablated on their own.
* "Configuration C reproduces Table 7, validating the protocol" overstates what that shows. It shows consistency between a fixed and a tuned configuration; it validates nothing.
* The probe shows that raw channels *without* `Eg` and *without* SMOTE (MCC 0.744) beat configuration C (0.528). So the engineered block and SMOTE together *hurt* the RF. The paper's statement that the engineered block is "additive" is not supported.

### M7. The label-scheme ablation (Table 11) is a SMOTE artefact
Table 11 shows the with-SMOTE condition only. Without SMOTE, the tier-scheme leaders are the **Decision Tree (acc 0.763, MCC 0.527) and LR (0.758, 0.483)**, ahead of RF (0.749, 0.467). DT then loses only 6 accuracy points from binary to tier, not 20. The claim that the SVM is the most robust to the finer-grained task holds only under SMOTE. Report both conditions.

---

## B. Factual or numerical errors in the text (all corrected in `arxiv/`)

| # | Location | Manuscript says | Data say |
|---|---|---|---|
| E1 | §4.3 | "all p ≥ .599 among these four" (LR, DT, RF, XGB) | RF vs XGB p = .421, so "all p ≥ .42" |
| E2 | Abstract, §5.10 | partial-shading codes recall "≥ 0.95" / "mostly ≥ 0.95" | code 3.1 = 0.947; four of five codes ≥ 0.94; code 2.3 = 0.007 |
| E3 | §3.1, §5.9, §6 | fault window "about 3.5 months" | 8 Feb – 19 May 2022 = 100 days (≈ 3.3 months) |
| E4 | §3.5 | "each repeat using the seed plus its repeat index" for SMOTE and every classifier | seed + repeat applies **only** to the outer split; SMOTE, the classifiers and the inner search all use 42 |
| E5 | §3.5 vs README | SHAP took "1.5–2 h" | README says 15–25 min. The authors chose to keep 1.5–2 h in the paper. See HANDOVER §5 for the measured re-run time |
| E6 | §3.1 | "Fault-labelled rows (2.83%) cluster into 35 contiguous same-state runs, 18 no-fault runs and 17 fault episodes" | the *record* segments into 35 runs (17 fault, 18 no-fault) |
| E7 | References | "Ombiro, H., 2026" | the author is Z. Ombiro, and the entry is never cited in the text |
| E8 | §4.3 note | "\|r\| ≥ .8 is conventionally large" | there is no such convention for rank-biserial r; drop it or cite one (common thresholds are .1/.3/.5) |
| E9 | §2 | Wainer & Cawley (2021) cited as *justifying* nested CV | their paper argues nested CV is usually *unnecessary*; cite them as showing nested CV is conservative |
| E10 | §2 | effect sizes "answer the Bayesian critique" (Benavoli 2017) | they only partly address it; reword |
| E11 | Table 6 | "Python 3 (author's workstation)", versions given as "≥" | report the exact versions and the CPU/RAM used (left as an author action) |
| E12 | `data/subsample_metadata.json` | tier counts Severe = 2,000, Constant = 1,600 | bug (dictionary overwrite); true counts are 2,800 and 4,000, as the paper states. Code and file fixed |
| E13 | §4.1 | MCC deficit "roughly twice" the accuracy deficit | accuracy and MCC are on different scales, so comparing raw deficits means nothing; removed |
| E14 | §3.4 | accuracy is the "primary" metric | with 32.7 % prevalence the majority baseline already scores 0.673 accuracy (MCC 0); the inner search optimises macro-F1. Make MCC primary and report the baseline |

## C. Verified numbers (no change needed)
All of the following, plus every new number in the arXiv text, are re-checked automatically by `review/verify_claims.py` (45 checks, all passing).
* Table 2 (all eight counts and shares), 2,988,798 rows, 1 Oct 2021 – 13 Apr 2023, 559 days, 97.17 % / 2.83 %.
* Subsample: 20,800 rows, 24 episodes (17 fault), 14,000 / 4,000 / 2,800, prevalence 32.7 %. Per-code episode rows sum exactly to the Table 2 counts.
* Tables 7, 8 and 11 match `all_runs_summary.csv` cell by cell (mean and s.d. to 3 d.p.).
* Table 9 (all 28 cells), 62,400 predictions per model. The FP/FN statements are correct.
* Table 10: all 21 p-values and the four effect sizes. The MCC text p-values (.003, .006, .026, .021, .016, .041) are correct.
* Table 5 equals `most_common_hyperparameters.json`. Table 4 equals `models.py`.
* Table 12 equals `feature_ablation_rf.json`. Tables 13 and 14 equal `fault_type_error_analysis.json`.
* SMOTE deltas in §4.5 and §5.2 (−0.064, −0.051, −0.004, +0.014; tier −14.4, −18.9, +0.017). The 7.2 / 20.0 / 17.0 accuracy-point drops.
* SHAP per-model `Va` ranks (1st for DT, RF, XGB and SVM; 3rd kNN; 4th MLP; 5th LR).

## D. Tone and writing
The draft reads as template-generated. Specific tells:
* "gap" rhetoric repeated about seven times ("To bridge this gap", "Consequently, an important research gap remains", "This work closes that gap", "bridges this gap by combining all four", "Despite this growing body of work").
* Stock contrastive constructions: "qualifies rather than undermines", "follows established practice rather than introducing a new device", "a deployability trade-off rather than an omission", "defensibly preferable".
* A four-bullet "lack of approaches that" list that restates the introduction, then a Table 1 whose rightmost column repeats the same contribution eight times.
* Textbook equations for every classifier and for accuracy, precision, recall and F1 add length but no information. MCC and the protocol are what matter.
* The §2 paragraph citing wind-turbine, battery and bearing SHAP papers is citation padding.

The `arxiv/` text removes these, states findings once, and puts hedges only where the evidence needs them.

## E. Recommended actions before journal resubmission
* **R1:** matched no-fault sampling (same clock window and irradiance band), with `Eg` removed or differenced, then re-run.
* **R2:** a corrected resampled t-test or Friedman + Holm. Persist per-fold values (code done).
* **R3:** SHAP on the modal, out-of-fold models with a common output (probability).
* **R4:** report per-episode results and leave-one-episode-type-out results.
* **R5:** a full ablation grid ({raw, raw−Eg, raw+derived, raw+temporal, all} × {SMOTE, none}).
* **R6:** add trivial baselines (majority class; clock hour; single-feature `Va` threshold).
* **R7:** exact software versions and hardware in Table 6.
* **R8:** verify the 2025–2026 references (Gaaloul 2025, Khandeparkar 2025, Mouleloued 2025, Quiles-Cucarella 2025, Krishna Veni 2025). Crossref is blocked in this session. Toche Tchio (2024) **was verified** by web search: it compares Extra Trees with LR, kNN, SVM, DT, AdaBoost and RF, so the original description was fair, and the arXiv text now names those classifiers.
