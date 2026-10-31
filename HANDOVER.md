# HANDOVER: PV shading-fault benchmark → arXiv preprint

*Paste this file into a new chat to continue. Written 31 Oct 2026. Branch `claude/determined-tesla-n5rt94`, repo `IncognitoGK9/pv-fault-benchmark`.*

## 1. Context
* Paper: "Benchmarking Classical Machine Learning Models for Shading-Fault Detection in Rooftop PV Systems Using Real Sensor Data". Authors: Z. Ombiro (corresponding) and B. Shibwabo, Strathmore University.
* The source manuscript is `PV_Fault_Detection_Final_v2.docx` (under professor review; unchanged). The arXiv version is `arxiv/main.tex`, compiled to `arxiv/main.pdf` (17 pp.).
* Data: Réunion 4 kW rooftop plant (Graillet et al. 2022, Zenodo 10.5281/zenodo.7828879). Raw 2,988,798 rows. Subsample: 20,800 rows, 24 episodes (17 fault), 32.7 % fault.
* Protocol: 7 classifiers × {binary, tier} × {SMOTE, none}, 3×5 StratifiedGroupKFold by episode, inner 3-fold GroupKFold random search (15 draws, macro-F1).

## 2. What was done
1. A full hostile review, written up in `review/REVIEW_REPORT.md`. Every number in the paper was checked against `results/`. **All table values match.** Four text errors were found and fixed.
2. Post-hoc reanalysis in `review/reanalysis.py`, with output in `review/reanalysis_results.json`. It ran Holm correction, a confound probe, per-episode recall and a ratio-feature check.
3. A new SHAP figure, `figures/fig_shap_normalised.{png,pdf}`, normalised per model.
4. The arXiv LaTeX package in `arxiv/`. Upload `arxiv/arxiv_submission.tar.gz`.
5. A change log, Word → arXiv, in `review/UPDATE_NOTES.md`.
6. Code fixes: the metadata tier-count bug; per-fold values are now persisted on future runs; README corrections. No modelling code was changed.

## 3. Key findings (numbers to remember)
| Issue | Evidence |
|---|---|
| Operating-condition confound | Fault slices are all 10:00–17:00; no-fault slices run 05:51–18:11. Median Pg 982 W vs 274 W. Clock-hour-only RF MCC **0.443**; `Va`-only **0.724**; raw-without-`Eg` **0.744**; paper's best tuned RF 0.526 |
| `Eg` | Daily cumulative energy counter (resets at midnight), i.e. a time proxy. Ranks 2nd in SHAP share |
| Multiplicity | Holm over 21 pairs: accuracy, only RF>MLP (.001) and RF>kNN (.009) survive; MCC, none (min .070) |
| "Two tiers" | Not supported. LR vs kNN/SVM/MLP p = .055/.055/.277 |
| SHAP | Mixed units (log-odds for LR and XGB, probability for the rest). Explained models use **default** hyperparameters, in-sample |
| Per-episode | Misses: episode 5 (code 2.3), episode 9 (1.0), episodes 19 and 23 (4.0); episode 31 (3.1) at 0.84. All low-output or near-normal-`Va` |
| Text errors | p ≥ .599 → .42; "≥ 0.95" → 3.1 is 0.947; 3.5 months → 100 days; seed + repeat only for the outer split |
| Not stored | Per-fold metric values, so no Friedman or corrected t-test can be computed without a re-run |

## 4. Open items for the authors [AUTHOR]
1. **Approve the reframing** (abstract, §4.3, §5.1, §5.6, §6). It adds the confound diagnostic and narrows the "two-tier / interpretable models suffice" claim. The supervisor should see this before posting.
2. Table 6: insert the exact Python/package versions and the CPU/RAM.
3. State the measured SHAP runtime (the paper said 1.5–2 h; the README says 15–25 min).
4. Confirm the AI declaration wording (ChatGPT + Claude).
5. Verify the 2025–26 references and the Toche Tchio (2024) description. Decide on the four dropped references (Hossen, Zheng, Kawoosa, Patnaik).
6. Choose the arXiv category (eess.SY or cs.LG) and licence (CC BY 4.0 suggested).

## 5. Recommended before journal resubmission (needs the raw CSV and about 8 h)
* R1: re-sample no-fault rows matched to the fault clock window and irradiance band (meteorological companion file); drop or difference `Eg`; re-run all 28 runs.
* R2: Nadeau–Bengio corrected t-test or Friedman + Holm on `per_fold_metrics.json`.
* R3: SHAP on the modal configurations, out of fold, with probability output for all models.
* R4: per-episode and leave-one-fault-type-out reporting.
* R5: ablation grid {raw, raw−Eg, raw+derived, raw+temporal, all} × {SMOTE, none}.
* R6: baselines: majority class, clock hour, a single `Va` threshold.

## 6. How to rebuild
```bash
pip install -r requirements.txt
python review/reanalysis.py          # ~2 min, uses data/reunion_subsample.csv
python review/fig_shap_normalised.py
cd arxiv && latexmk -pdf main.tex && tar czf arxiv_submission.tar.gz main.tex figures
```
