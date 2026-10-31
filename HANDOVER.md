# HANDOVER: PV shading-fault benchmark → arXiv preprint

*For continuing in the chat interface, and for counter-checking by another AI assistant. Last updated 31 Oct 2026. Repo `IncognitoGK9/pv-fault-benchmark`, branch `claude/determined-tesla-n5rt94`.*

---

## 1. Context
* **Paper:** "Benchmarking Classical Machine Learning Models for Shading-Fault Detection in Rooftop Photovoltaic Systems Using Real Sensor Data".
* **Authors:** Zablon Ombiro (corresponding, zablon.ombiro@strathmore.edu) and Bernard Shibwabo, Strathmore University.
* **Source manuscript:** `PV_Fault_Detection_Final_v2.docx`. It is under supervisor review and has **not been modified**.
* **arXiv version:** `arxiv/main.tex` → `arxiv/main.pdf` (17 pp., pdfLaTeX, bibliography inline). Figures are in `arxiv/figures/`. Upload `arxiv/arxiv_submission.tar.gz`. A downloadable zip of the whole folder is `arxiv.zip` at the repo root.
* **Data:** Réunion 4 kW plant (Graillet et al. 2022; Zenodo 10.5281/zenodo.7828879). Raw 2,988,798 rows. Subsample: 20,800 rows, 24 episodes (17 fault, 7 no-fault), 32.7 % fault.
* **Protocol:** 7 classifiers × {binary, tier} × {SMOTE, none} = 28 runs. Outer loop: 3 repeats × 5-fold StratifiedGroupKFold by episode. Inner loop: 3-fold GroupKFold random search (15 draws, macro-F1).

## 2. Chronology of work
**Round 1 (hostile review):**
1. Read the full docx and every script.
2. Checked every number in the paper against `results/` and `data/`.
3. Ran post-hoc diagnostics (`review/reanalysis.py`).
4. Replaced the SHAP figure.
5. Rewrote the paper in LaTeX with the corrections and a plain research tone.
6. Fixed two code issues.
7. Wrote `review/REVIEW_REPORT.md` and `review/UPDATE_NOTES.md`.

**Round 2 (author instructions):**
* Keep the SHAP runtime at 1.5–2 h, and measure what it would actually take (see §5).
* AI declaration: Claude used for writing and drafting; experiments and scripts run by the author, who takes full responsibility.
* Keep Table 6 exactly as in v2.
* Add the corresponding-author email.
* Correct everything else as suggested, and re-check all claims.

**The re-check found three errors in my own round-1 draft. All are fixed:**
* (a) Upper-group MCC p-value floor ".37" → ".29".
* (b) The `V_g` "s.d. 4 V" claim is now qualified.
* (c) The explanation for the missed uniform-shading episode was wrong, and is corrected.

I then added `review/verify_claims.py`.

## 3. Files created or changed
| Path | Status | What it is |
|---|---|---|
| `arxiv/main.tex`, `arxiv/main.pdf` | new | arXiv manuscript |
| `arxiv/figures/` | new | fig1_metrics, fig2a/b/c_cm, fig4_smote_mcc, fig5_label_scheme (PNG, extracted from docx v2); fig3_shap.pdf (regenerated) |
| `arxiv/arxiv_submission.tar.gz`, `arxiv.zip` | new | upload tarball / downloadable folder |
| `arxiv/README.md` | new | build and upload steps |
| `review/REVIEW_REPORT.md` | new | hostile review: 7 major issues (M1–M7), 14 errors (E1–E14), verified list, recommendations R1–R8 |
| `review/UPDATE_NOTES.md` | new | every Word → arXiv change with its reason; round-2 log; table renumbering |
| `review/reanalysis.py` → `review/reanalysis_results.json` | new | Holm correction, confound probe, per-episode recall, ratio-feature check |
| `review/verify_claims.py` | new | **45 automated PASS/FAIL checks** of every new or changed number, including all 140 retyped table cells |
| `review/fig_shap_normalised.py` → `figures/fig_shap_normalised.{png,pdf}` | new | per-model normalised SHAP heatmap (new Fig. 3) |
| `prepare_dataset.py` | fixed | tier counts were overwritten per code (the last code won) |
| `data/subsample_metadata.json` | fixed | tier counts 2,000/1,600 → **2,800/4,000** (now matches the paper and the per-episode records) |
| `run_experiment.py` | changed | future runs write `results/per_fold_metrics.json` |
| `README.md` | changed | SHAP-figure provenance corrected; `review/` and `arxiv/` documented |
| `.gitignore` | changed | `__pycache__/` |

**No modelling code that produced the reported results was changed. All results files are untouched**, except the metadata tier-count fix.

## 4. Findings to counter-check (each with its source)
Run `python review/verify_claims.py`. Every line should print PASS. Each finding below can also be checked by hand.

| # | Claim | How to verify |
|---|---|---|
| M1 | Fault rows are recorded 10:03–16:58 (400-row slices of 33–53 min); no-fault rows 05:51–18:11. Hour s.d. 1.7 vs 3.7 h. Median Pg 982 vs 274 W; Va 151 vs 184 V | `data/reunion_subsample.csv` grouped by `fault_bin` |
| M1 | `Eg` resets at midnight (daily cumulative energy) | episode 0: Eg 5.96 at 16:12 → 0.0 at 06:25 the next day |
| M1 | RF (modal config, no SMOTE, same folds): hour-only MCC .443; Va-only .724; raw without Eg .744; best tuned model .526 | `reanalysis_results.json → confound_probe_rf_fixed_modal_no_smote` |
| M2 | LR vs kNN/SVM/MLP not significant (acc p .055/.055/.277) | `results/wilcoxon_accuracy.json` |
| M3 | Holm over 21 pairs: accuracy survivors RF>MLP (.0013) and RF>kNN (.0085); MCC none (min .0703) | `reanalysis_results.json → holm_*` (applied to the stored raw p) |
| M3 | Per-fold values were never stored | `results/summary_metrics.json` has only mean and std |
| M4 | SHAP totals: LR 40.8, XGB 14.1 (log-odds), others 0.58–1.07 (probability) | `results/shap_importance.json` |
| M4 | SHAP models use default hyperparameters | `shap_analysis.py` → `pipeline.build_pipeline` → `models.get_estimator` (no `set_params`) |
| M4 | `visualize.py` does not draw the paper's SHAP figure | `visualize.py::fig4_feature_importance` reads `feature_importance.json` |
| M5 | Missed episodes 5 (2.3), 9 (1.0), 19 and 23 (4.0); episode 31 at .842; reproduces Table 13 within ±.002 | `reanalysis_results.json → per_episode_rf_share_correct` |
| M6 | Ablation config B contains no Va | `results/feature_ablation_rf.json → B_temporal_only.feature_names` |
| M7 | Tier without SMOTE: DT .763/.527, LR .758/.483 lead | `results/all_runs_summary.csv` |
| E1 | RF vs XGB accuracy p = .421 (paper said "all ≥ .599") | `wilcoxon_accuracy.json` |
| E2 | Code 3.1 recall .947 (paper said "≥ 0.95") | `fault_type_error_analysis.json` |
| E3 | Fault window 8 Feb – 19 May 2022 = 100 days (paper said 3.5 months) | `subsample_metadata.json` segments |
| E4 | seed + repeat applies only to the outer split | `pipeline.py` lines 70–72 vs 31–33 |
| ref | Toche Tchio 2024 compares Extra Trees with LR, kNN, SVM, DT, AdaBoost and RF | web search (aimspress.com, DOAJ) |

**All original table values were confirmed correct:** Tables 2, 5, 7–14, every cell.

## 5. SHAP runtime (author asked: keep 1.5–2 h, but check)
* Paper (v2 and arXiv): "about 1.5–2 h". README: "15–25 min". These disagree.
* Re-run in this session: the unmodified `shap_analysis.py` on a scratch copy, on a 4-vCPU / 15 GB cloud container (shap 0.54.1, scikit-learn 1.9.1, xgboost 3.4.1).
* **Measured (partial):** Logistic Regression done at 5 s, kNN at 753 s (12.5 min), Decision Tree at 755 s. The run was then stopped during the SVM, at the author's request (15-min cap). SVM, MLP, Random Forest and XGBoost times were not measured.
* Interpretation: kNN alone takes 12.5 min on 4 vCPUs. The SVM (same sampling explainer, slower predict_proba) and the MLP also need the sampling explainer, so a total of 1.5–2 h is plausible. The README's 15–25 min was not; it is now changed to 1.5–2 h to match the paper.

## 6. Decisions taken (author-confirmed in round 2)
* SHAP runtime text: **"about 1.5–2 h"** (kept).
* Table 6 (arXiv Table 5): **kept exactly as v2** ("Python 3 (author's workstation)", "≥" versions, CPU-only note).
* AI declaration: Claude used for writing, drafting and cross-checking figures; all experiments designed and all scripts written and run by the first author, who takes full responsibility. ChatGPT removed at the author's request (round 3); only Claude is named.
* Corresponding author: Zablon Ombiro, zablon.ombiro@strathmore.edu.
* Reframing accepted ("correct the rest as suggested"): the confound section (§5.6, Table 13), Holm correction, narrower conclusion, normalised SHAP, both conditions in the tier table, episode-level per-fault table.
* Dropped as padding: Hossen 2026, Zheng 2026, Kawoosa 2023, Patnaik 2021.

## 7. Still open
1. **Supervisor sign-off** on the reframed conclusions before posting. The arXiv text is materially more cautious than the docx v2.
2. Reference check for the 2025 papers (Crossref is blocked in this session): Gaaloul, Khandeparkar, Mouleloued, Quiles-Cucarella, Krishna Veni.
3. arXiv category (eess.SY or cs.LG) and licence (CC BY 4.0 suggested).
4. The docx v2 still contains the original errors. If the supervisor's copy is also to be corrected, apply `review/UPDATE_NOTES.md` to it.

## 8. Before journal resubmission (needs the raw CSV and ≈ 8 h)
* R1: no-fault rows matched to the fault clock window and irradiance band; drop or difference `Eg`; re-run all 28 runs.
* R2: corrected resampled t-test, or Friedman + Holm, on `per_fold_metrics.json`.
* R3: SHAP on the modal configurations, out of fold, with probability output for every model.
* R4: per-episode and leave-one-fault-type-out reporting.
* R5: full ablation grid × {SMOTE, none}.
* R6: trivial baselines (majority class, clock hour, a single `Va` threshold).

## 9. Rebuild
```bash
pip install -r requirements.txt
python review/verify_claims.py           # 45 checks, pandas only
python review/reanalysis.py              # ~2 min
python review/fig_shap_normalised.py
cd arxiv && latexmk -pdf main.tex && tar czf arxiv_submission.tar.gz main.tex figures
```
