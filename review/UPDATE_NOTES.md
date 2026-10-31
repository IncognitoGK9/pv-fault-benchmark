# Update notes: Word v2 → arXiv version (`arxiv/main.tex`)

Each change is listed with its reason. The reason codes (M = major, E = error) refer to `review/REVIEW_REPORT.md`. **[AUTHOR]** marks a decision the authors must confirm before posting.

## Changes to claims (substance)

| Section | v2 (Word) | arXiv | Reason |
|---|---|---|---|
| Abstract | "significantly exceeding kNN, MLP and, on MCC, SVM"; "two performance tiers" | Only RF > kNN and RF > MLP on accuracy survive Holm; LR and DT cannot be separated from either group | M2, M3 |
| Abstract | Severe recall 0.50–0.60 vs partial "≥ 0.95" | Episode-level framing: 4 of 17 episodes missed (1/2 uniform, 2/5 intermittent, the single 2.3 episode) | M5, E2 |
| Abstract | — | Added the operating-condition confound (clock-hour MCC 0.44; `Va`-only 0.72) **[AUTHOR]** | M1 |
| Abstract | "statistically validated basis for choosing interpretable classifiers" | Removed; matched sampling identified as the prerequisite | M1, M2 |
| §3.4 | Accuracy primary; uncorrected Wilcoxon | MCC primary; majority baseline (0.673) stated; Holm over 21 pairs; non-independence of repeats stated | M3, E14 |
| §3.5 | Seed + repeat for SMOTE and classifiers | Seed + repeat for the outer split only | E4 |
| §4.3 | "all p ≥ .599 among these four" | "p ≥ .42 (accuracy), ≥ .37 (MCC)"; Holm results paragraph; non-significance ≠ equivalence | E1, M2, M3 |
| Table 10 | Raw p only; "\|r\| ≥ .8 large" | Bold = survives Holm; convention removed; sign of r stated | M3, E8 |
| §4.4 / Fig. 3 | Raw mean \|SHAP\| summed across models | Per-model normalised shares (new figure); default-hyperparameter and in-sample caveats; `V_g − V_a` ≈ `V_a`; `Eg` flagged | M4 |
| Table 11 | With SMOTE only; "SVM most robust" | Both conditions; without SMOTE, DT and LR lead on tier | M7 |
| §4.6 | "temporal features lose 16.2 MCC points"; "C validates the protocol" | B has no `V_a`; modal configuration is mildly optimistic | M6 |
| Table 13 / §4.7 / §5.8 | Sparsity and MPPT hypotheses | Episodes-detected column; observed operating signature of missed episodes; sparsity ruled out (2.1 has fewer rows) | M5 |
| §5.1, §5.4, §6 | "two-tier structure"; "interpretable models favoured" | Narrower claim; recommendation conditional on non-significance | M2 |
| §5.3 | `Va` dominance "has a physical basis" | Physical mechanism given, plus temperature/irradiance alternative; not separable here | M1 |
| New §5.6 | — | Operating-condition confound with probe table (new Table 15) **[AUTHOR]** | M1 |
| §5.7 | Meteorological exclusion presented as a deployability trade-off | Removed as a section. The companion file is now proposed for matched sampling | M1 |
| §5.9 | "testing limited to required comparisons, no correction needed" | Holm applied; remaining limitations listed | M3 |

## Corrections and tone

* Fault window "3.5 months" → "about 100 days (8 Feb – 19 May 2022)" (E3). Run segmentation sentence rewritten (E6).
* Sample rate "≈ 0.2 Hz" → "one sample every 5–8 s" (measured median 6 s). `Eg` described as the *daily cumulative* energy counter.
* Wainer & Cawley (2021) and Benavoli (2017) now cited for what they actually say (E9, E10). Benavoli volume corrected to 18(77).
* SHAP runtime: kept at "about 1.5–2 h" in §3.5, as the authors decided in round 2 (the measured re-run time is recorded in HANDOVER §5).
* Table 6 (now Table 5): **kept exactly as in v2**, as the authors decided in round 2.
* Reference "Ombiro, H." → "Ombiro, Z.", now cited in Data availability (E7).
* Dropped as citation padding: Hossen et al. 2026, Zheng et al. 2026, Kawoosa et al. 2023, Patnaik et al. 2021. **[AUTHOR: restore any you consider essential]**
* Removed: per-classifier textbook equations (LR, kNN, DT, SVM, MLP, RF, XGB) and the accuracy/precision/recall/F1 formulas. Only MCC is kept. The qualitative Table 3 is merged into the related-work text, and Table 1 is shortened to "data and task" and "difference".
* Gap rhetoric (about seven instances), the "lack of approaches that" bullet list, and stock contrastive phrasing are removed. The text was rewritten throughout in plain declarative style.
* CRediT for B. Shibwabo normalised to CRediT roles.
* AI declaration: names Claude for writing, drafting and cross-checking figures, and states that all experiments and scripts were designed and run by the first author, who takes full responsibility. ChatGPT is retained because v2 declared it **[AUTHOR: delete "ChatGPT (OpenAI) and" if it was not used for this version]**.

## Round 2 (after author feedback)
* Corresponding author added: Zablon Ombiro, zablon.ombiro@strathmore.edu (title-page footnote with a mailto link).
* Toche Tchio (2024) now described precisely: Extra Trees vs LR, kNN, SVM, DT, AdaBoost and RF (verified by web search).
* Self-check found and fixed three errors in the round-1 arXiv draft:
  1. Upper-group MCC p-value floor: "≥ .37" → "≥ .29" (LR vs XGB = .286).
  2. "`V_g` nearly constant (s.d. 4 V)" holds only for no-fault rows; it is now qualified as "whenever the inverter is connected".
  3. Missed uniform-shading episode: zero output does not explain the miss, because the detected one also has zero output. The text now gives the actual difference (`V_a` = 0 for the detected episode vs ≈ 150 V for the missed one).
* Fault-slice duration "about 33 min" → "33–53 min".
* `review/verify_claims.py` added: 45 automated checks, including all 140 retyped table cells. All pass.

## Table/figure numbering, Word → arXiv
Word T1–T14 → LaTeX T1, T2, (T3 removed), T4→T3, T5→T4, T6→T5, T7→T6, T8→T7, T9→T8, T10→T9, T11→T10 (extended), T12→T11, T13→T12 (+ column), T14 folded into the §4.7 text, plus new T13 (confound probe). Figures: 1, 2 unchanged; 3 replaced; 4 and 5 merged into one two-panel figure.

## Code and data changes in this commit
* `prepare_dataset.py`: the tier-count bug is fixed. `data/subsample_metadata.json`: tier counts corrected to 14,000 / 4,000 / 2,800 (verifiable from the per-episode records in the same file).
* `run_experiment.py`: per-fold metric values are now written to `results/per_fold_metrics.json` on future runs. This does not affect existing results.
* `review/reanalysis.py`: Holm correction, confound probe, per-episode recall and ratio-feature check. Output goes to `review/reanalysis_results.json`.
* `review/fig_shap_normalised.py` → `figures/fig_shap_normalised.{png,pdf}`.
* README: the SHAP-figure and visualize claims are corrected.
* No modelling code that produced the reported results was changed.
