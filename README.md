# Diabetes Classification

Exploratory analysis and classification on a public Chinese diabetes dataset. The goal is to detect undiagnosed prediabetes/diabetes using only non-glycemic features (anthropometrics, blood pressure, lipids, liver/kidney function, blood count, lifestyle).

**Current result in one line.** Without any glucose test, undiagnosed DM can be ranked moderately well (CV AUROC about 0.80 with routine labs, 0.77 for the clinical CDRS score), while prediabetes cannot be separated from normal glucose regulation in a useful way.

## Dataset

| | |
|---|---|
| Source | Li J., Zheng H., Zhou Y., …, Jiang F. *A bimodal dataset for diabetes research.* Scientific Data 13, 652 (2026). [doi:10.1038/s41597-026-06923-y](https://doi.org/10.1038/s41597-026-06923-y) |
| Data | Zenodo, [doi:10.5281/zenodo.18270337](https://doi.org/10.5281/zenodo.18270337) (CC BY 4.0) |
| Origin | Dept. of Endocrinology & Metabolism, Shanghai Sixth People's Hospital (Shanghai Jiao Tong University), China |
| Period | 10 Feb – 7 Apr 2012 (cross-sectional) |
| Size | 5,922 records × 190 columns, 1 row = 1 person, no duplicate IDs |
| Encoding | GBK (`pd.read_csv(..., encoding="gbk")`); blank cells are whitespace strings |

### Files (data/)

- Dataset for diabetes research.csv: the raw data.
- Supplementary Information.docx: Supplementary Table 1, the data dictionary (variable, label, value coding).
- processed/: output of `01-preprocessing.ipynb` (prepared dataset with split and targets, feature sets, dropped-column report, leakage ablation, single-feature audit).

### Column groups

Demographics (age, gender) · anthropometrics (height, weight, BMI, waist, hip, WHR, body fat) · blood pressure and heart rate · glycemia (FPG, PG2h, HbA1c, HPLC-HbA1c, glycated albumin) · insulin/C-peptide and HOMA indices · liver function · lipids · kidney function (creatinine, uric acid, urine albumin, eGFR) · complete blood count · thalassemia flags · medical history, smoking, drinking, tea · complications (retinopathy, neuropathy, nephropathy, cardiovascular) · family history · questionnaire items · many derived categorical cut-offs.

Three columns are text: Data (survey date), EscDrinkD, EyeDia.

### Target

`PGclass3`: 0 = NGR (normal), 1 = prediabetes, 2 = diabetes (WHO 1999).

## Known Issues in the Dataset

- **Population is ambiguous.** The paper calls the records "diabetes patients", and Zenodo calls them "follow-up records". However, 75.7% of the records are NGR and only 49 people have a prior diabetes diagnosis. The cohort looks more like a screening or OGTT-referral population. No inclusion criteria are published.
- **Gender coding mismatch.** The dictionary says 0/1, but the data uses 1/2. Height, hemoglobin and body fat indicate **1 = male, 2 = female**.
- **ID gaps.** IDs go up to 6,571 but there are only 5,922 rows, so 649 IDs are missing.
- **Undocumented leaking columns.** `VAR00001` has no description. See the leakage section.
- **Implausible kidney values.** 330 people have creatinine < 35 µmol/L, which in turn produces 376 eGFR values > 200.
- **"9 = unknown" coding.** 689 people answered 9 on all 15 disease-history items. This creates spurious correlations of r ≥ 0.9 between those items.

## Main EDA Findings (`eda-diabetes.ipynb`)

**Screening subset.** Excluding people already diagnosed with diabetes (`DMhis = 1`) leaves **5,873** people:

| Class | n | % |
|---|---|---|
| NGR | 4,444 | 75.7 |
| Prediabetes | 1,121 | 19.1 |
| DM | 308 | 5.2 |

The classes are imbalanced (NGR : DM ≈ 14 : 1).

**Label leakage.**

1. **The label is deterministic.** Applying WHO 1999 rules to `FPG`/`PG2h`/`DMhis` reproduces `PGclass3` for **5,922 / 5,922** rows. Any model that sees glucose is only relearning a threshold rule.
2. **`VAR00001` (undocumented) leaks the label.** Its value is 1 or 9, and 9 never occurs in NGR (0 / 4,444) but does occur in prediabetes (484) and DM (111).
3. **`ThalussemiaAND` leaks the label.** A value of 1 appears in only 2 NGR records versus 371 prediabetes/DM records. It appears to be filled only for people with abnormal glucose.

As a result, 17 columns are dropped as leakage. These are the glucose columns, derived indices (HOMA, ISI, OGTT insulin/C-peptide), the glucose flags, the other label versions, `VAR00001` and `ThalussemiaAND`. A further 20 columns are dropped because they are other glycemic tests (HbA1c, GA), study markers, complications, or mostly-unknown questionnaire items. After also removing columns that are >50% missing or constant, **135 candidate features** remain.

**Signal without glucose (median NGR → Pre → DM).**

| Feature | NGR | Pre | DM |
|---|---|---|---|
| Age | 38 | 43 | 47 |
| Waist (cm) | 73 | 76 | 79 |
| Systolic BP | 110 | 120 | 120 |
| GGT | 24 | 34 | 61 |
| ALT | 21 | 24 | 30 |
| Male (%) | 43.6 | 52.5 | 69.5 |
| High BP (%) | 18.8 | 30.2 | 51.9 |

DM separates well from NGR. Prediabetes differs only slightly from NGR and is expected to be the hardest class.

**Redundancy.** There are 56 feature pairs with |Spearman r| ≥ 0.9. Six BMI variants are cut-offs of the same BMI. Other pairs include Age/AgeGroup, GFR category variants, %/count blood cells, TBIL/IDBIL and HGB/HCT. These will be pruned before modeling.

## Repository Structure

| Path | Content |
|---|---|
| `eda-diabetes.ipynb` | Exploratory analysis (purely descriptive, the raw file is never modified) |
| `preprocessing.py` | All preprocessing code: `prepare()`, `stats_cleaner`, `buat_split`, feature sets |
| `01-preprocessing.ipynb` | Runs and checks `prepare()`, leakage evidence and ablation, locked temporal split, saves `data/processed/` |
| `02-modeling.ipynb` | Untuned baselines with repeated stratified CV on the training part, checkpointing, result export |
| `data/` | Raw CSV and data dictionary (unchanged) |
| `data/processed/` | Prepared dataset with split and targets, feature sets, dropped-column report, ablation, single-feature audit |
| `results/` | All CV outputs (see below) |
| `figures/` | Proposed framework diagram |

The preprocessing pipeline (`prepare()`, `stats_cleaner`, split schemes, leakage ablation) was written by Hana Azizah Nurhadi on the `feat/preprocess` branch and moved into `preprocessing.py` when the branches were merged.

## Preprocessing (`preprocessing.py`, `01-preprocessing.ipynb`)

Preprocessing has two stages. The first uses fixed rules only, so it is safe to run before splitting. The second learns statistics from the data, so it is fitted on the training folds only, as the first step of every model pipeline.

**Stage 1, `prepare()` (fixed rules).**

1. **Population.** People with a known diabetes diagnosis (`DMhis = 1`) are excluded, leaving 5,873 people.
2. **Recoding.** Gender becomes `male` (1 = male). The "9 = unknown" code in the 15 disease-history items, `WeightLoss` and `Cancer` becomes missing, and the 689 people who answered 9 to all 15 items get one flag, `riwayat_tidak_diisi`.
3. **Structural blanks.** Flag columns that only hold 1 or blank (`HypertenHis`, `DyslipideHis`, …) are read as 0. `SmokingRange`, `DrinkYears` and `Drink50g` are blank for people who do not smoke or drink, so they become 0 for those people. The notebook verifies this against the smoking and drinking answers.
4. **Implausible values become missing.** These are creatinine below 35 µmol/L (with the eGFR computed from it), eGFR above 200, height below 120 cm, systolic not above diastolic pressure, and LDL above total cholesterol.
5. **Dropped columns.** These are glucose and glucose-derived columns (including fasting insulin and C-peptide), other glycemic tests, label versions, study markers, diabetes complications, the diabetes-awareness questionnaire, the two proven leaks (`VAR00001`, `ThalussemiaAND`), three suspicious columns (`ThalussemiaOR`, `ThalussemiaHPLC`, `DMhighrisk`), categorical cut-offs of numeric columns, arithmetic duplicates (`ALT.1`, `GLO` = TP − ALB, `AG` = ALB/GLO, `IDBIL` = TBIL − DBIL) and per-relative family-history answers (summarised by `DMfamilyHistory`). 93 columns remain.

**Stage 2, `stats_cleaner` (fitted per fold).** It drops columns more than 50% missing, binary columns with fewer than 10 minority cases and one of each pair with Spearman |ρ| ≥ 0.95. It then applies log1p to skewed non-negative columns and imputes (median for numeric columns and the training mode for binary ones, or no imputation for models that handle missing values). On the full training part it keeps about 80 columns.

**Leakage evidence.** `VAR00001` equals 9 only when fasting glucose is at least 6.1 mmol/L, and `ThalussemiaAND` matches the rule "MCV < 80 and abnormal glucose" in 99.7% of people, so both are derived from the label. Adding either one back raises macro-AUC by 0.09 to 0.13, far above the fold-to-fold variation (0.01 to 0.02). The three suspicious columns change it by at most 0.008, so they add nothing and stay dropped. A single-feature audit finds no other column with an AUROC above 0.75, and missingness carries no class signal.

**Feature sets.** F1 (non-lab, 42 input columns) holds what can be collected without drawing blood, namely demographics, anthropometrics, blood pressure, lifestyle and history. F2 (combined, 93 input columns) adds the routine non-glycemic labs (liver, lipids, kidney, urine, complete blood count). The CDRS set holds the six components of the Chinese Diabetes Risk Score.

**Locked test set (temporal).** Record IDs follow the survey date almost exactly (Spearman 0.9995), so the first 70% of IDs form the training part (4,111 people, 10 Feb – 12 Mar 2012) and the last 30% form the test part (1,762 people, 12 Mar – 7 Apr 2012). **The test part has not been used yet.** The test block has a lower prevalence than the training block (DM 3.9% vs 5.8%, prediabetes 15.3% vs 20.7%).

## Experimental Setup (`02-modeling.ipynb`)

| | |
|---|---|
| Tasks | `y_dm` DM vs not DM · `y_abn` prediabetes or DM vs NGR · `y3` three classes |
| Feature sets | F1 non-lab · F2 combined · CDRS components |
| Models | Logistic Regression, Random Forest, XGBoost, TabPFN, all with **default hyperparameters** |
| Clinical baseline | Chinese Diabetes Risk Score (CDRS) with the standard cut-off of 25, not fitted to the data, binary tasks only |
| Imbalance | `none` and `class_weight` (balanced weights, XGBoost via sample weights) |
| Validation | Repeated stratified 5-fold × 3 on the training part only. Folds are identical across all runs, so comparisons are paired by fold |
| Pipeline | `stats_cleaner` (median imputation for LR/RF/XGBoost, none for TabPFN) → StandardScaler (LR only) → model |

The primary metrics are AUROC and AUPRC, because they do not depend on a threshold. Threshold-based metrics use 0.5 (argmax for three classes), which is not meaningful for imbalanced data without class weights. `sens_at_spec80` reports sensitivity when specificity is held at 80%, as a rough picture of screening use.

### Result files (`results/`)

| File | Content |
|---|---|
| `cv_summary.csv` | Mean and std of every metric per run |
| `cv_fold_metrics.csv` | Every metric per run per fold (use this for paired tests) |
| `rq1_lab_vs_nonlab.csv` | Paired F2 − F1 difference per model |
| `rq3_model_comparison.csv` | Paired difference of each model against LR |
| `oof/<run_id>.csv` | Out-of-fold predicted probabilities per person |
| `importance/<run_id>.csv` | Feature importance averaged over folds, with the number of folds in which the column survived `stats_cleaner` |
| `confusion/<run_id>.csv` | Confusion matrix (summed over folds, divided by repeats) |
| `figures/` | AUROC summary plot and top-15 feature importance |
| `run_log.csv`, `config.json`, `cv_folds.csv` | Run status, configuration and library versions, fold assignment |
| `checkpoints/` | One file per fold. Re-running the notebook skips finished folds (git-ignored) |

## Baseline Results (untuned, 5-fold × 3 CV on the training part)

Values are mean ± std over 15 folds. TabPFN has not run yet (see Next Steps).

### DM vs not DM

| Model | Features | AUROC | AUPRC | Sens. at 80% spec. |
|---|---|---|---|---|
| CDRS | CDRS | 0.768 ± 0.031 | 0.168 | 0.55 |
| LR | F1 non-lab | 0.768 ± 0.040 | 0.194 | 0.58 |
| LR | F2 combined | 0.793 ± 0.038 | 0.276 | 0.62 |
| XGBoost | F2 combined | 0.800 ± 0.034 | 0.258 | 0.64 |
| **RF (class_weight)** | **F2 combined** | **0.816 ± 0.029** | 0.223 | **0.64** |

### Prediabetes or DM vs NGR

| Model | Features | AUROC | AUPRC | Sens. at 80% spec. |
|---|---|---|---|---|
| CDRS | CDRS | 0.684 ± 0.025 | 0.428 | 0.42 |
| LR | F1 non-lab | 0.690 ± 0.023 | 0.449 | 0.45 |
| **LR** | **F2 combined** | **0.721 ± 0.015** | **0.504** | **0.50** |
| RF (class_weight) | F2 combined | 0.714 ± 0.019 | 0.482 | 0.49 |
| XGBoost | F2 combined | 0.696 ± 0.009 | 0.476 | 0.48 |

### Three classes

| Model | Features | Imbalance | AUROC (OvR) | Macro-F1 | Recall NGR / Pre / DM |
|---|---|---|---|---|---|
| LR | F2 combined | none | 0.719 ± 0.017 | 0.418 | 0.96 / 0.09 / 0.18 |
| LR | F2 combined | class_weight | 0.687 ± 0.018 | 0.421 | 0.63 / 0.34 / 0.51 |
| RF | F2 combined | class_weight | 0.713 ± 0.016 | 0.422 | 0.88 / 0.25 / 0.09 |
| XGBoost | F2 combined | class_weight | 0.707 ± 0.016 | 0.428 | 0.91 / 0.19 / 0.15 |

The full table for all runs is in `results/cv_summary.csv`.

### Interpretation

**DM is detectable without glucose, prediabetes largely is not.** Every model separates DM from the rest reasonably well, but none can single out prediabetes (AUROC of prediabetes against the rest is at most 0.64). The EDA explains why. Prediabetes shows the same direction of change as DM in age, waist, blood pressure and liver enzymes, but the shift from NGR is small. By definition, prediabetic glucose lies in a narrow band just above normal, so many prediabetic people are physiologically close to the NGR group. The label changes at a glucose threshold, while the non-glycemic profile changes gradually.

**Routine labs add a small but consistent amount (RQ1).** Adding the labs improves AUROC by about 0.03 for LR and 0.04 to 0.06 for the tree models, and F2 beats F1 in 73–100% of the paired folds. The tree models gain more, which suggests that part of what the labs carry is non-linear or works through interactions that a linear model cannot use. For DM, AUPRC rises from about 0.19 to 0.22–0.28, so the labs mainly help rank the highest-risk people. The labs that rank highest in RF and XGBoost are GGT, FIB-4, ALT, AST and white-cell counts. This fits the known link between fatty liver, low-grade inflammation and insulin resistance.

**Without labs, machine learning does not beat the clinical score.** Non-lab LR and CDRS reach the same AUROC for DM (0.768), and differ by 0.006 for any abnormality. CDRS uses almost the same information (age, sex, BMI, waist, blood pressure, family history) and needs no training. The paired advantage of the best F2 model over CDRS for DM is about 0.05, while LR with labs is ahead by 0.025 ± 0.031. These gaps are one to two standard deviations, so they have to be confirmed on the locked test set before they can be claimed.

**No model clearly wins (RQ3).** For DM, RF is ahead of LR by 0.014 ± 0.020 and XGBoost by 0.006 ± 0.026, both inside the fold-to-fold noise. For any abnormality and for three classes, LR is ahead by 0.01 to 0.03. With a weak and mostly monotonic signal (older, larger waist, higher GGT means higher risk) a linear model loses little, while default trees are not regularised. Tuning may change the order, but large gains are unlikely when the ceiling is set by the information in the features.

**Class weighting moves the threshold, not the ranking.** For LR it barely changes AUROC but raises DM sensitivity from 0.12 to 0.62 at a specificity of 0.78. Since ranking stays the same, choosing a threshold on the validation folds is a simpler fix than resampling.

**LR coefficients are not interpretable yet.** The largest LR coefficients belong to white-cell indices (monocyte, eosinophil and neutrophil counts, lymphocyte percentage) and vary strongly between folds (std 0.14 to 0.24). These variables are tied to each other (differential percentages sum to 100% and counts are percentages times the white-cell count), and their correlations stay below the 0.95 pruning threshold, so the model can split weight between them almost arbitrarily. The tree-based rankings are more stable and more clinically plausible.

## Limitations

- Single centre (Shanghai), collected in 2012, with no published inclusion criteria, so the population may not resemble a general screening population.
- All results come from cross-validation on the training part. The temporal test part has not been touched, and no external validation set exists. Its lower prevalence will likely lower AUPRC and may lower AUROC compared with CV.
- CDRS points were implemented from the published scoring table and still need to be checked line by line against Zhou et al. (2013) before reporting.
- Hyperparameters are defaults, so the model comparison is preliminary.

## Next Steps

1. Reduce the collinear white-cell features (keep either percentages or counts) and recompute importance with permutation importance or SHAP.
2. Run TabPFN. The model weights are gated on Hugging Face, so accept the licence on the model page, log in with `hf auth login` (or set `HF_TOKEN`), install `tabpfn`, and re-run `02-modeling.ipynb`, preferably on a GPU. Finished folds are loaded from the checkpoints.
3. Tune RF and XGBoost with nested CV, and select thresholds on the validation folds.
4. Add a statistical comparison of paired fold results (for example a corrected resampled t-test) for RQ1 and RQ3.
5. Evaluate the final models once on the locked temporal test part, with bootstrap confidence intervals and calibration plots.
6. Update `figures/proposed-framework.*`, which still shows the earlier 80:20 random split and plain median imputation.

## Contribution

As of Oct 2026, no published study has used this dataset (0 citations on OpenAlex), and the data descriptor reports no ML baseline. This work contributes:

1. **The first ML baseline** on this dataset, compared against an established clinical risk score (CDRS) and validated on a later time block.
2. **Documented, reproducible label leakage.** The target is fully recomputable from FPG/PG2h, and two undocumented or under-documented columns (`VAR00001`, `ThalussemiaAND`) are derived from it, with an ablation that quantifies how much they inflate performance.
3. **A leakage-free setup for undiagnosed diabetes screening without glucose tests.** Known diabetics are excluded, and all glycemic, derived and leaking columns are removed.
4. **An estimate of how much routine non-glycemic labs add** beyond questionnaire and anthropometric data.

Every claim above can be checked directly from the raw CSV with the notebooks.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order: `eda-diabetes.ipynb` (optional), `01-preprocessing.ipynb`, `02-modeling.ipynb`. The modeling notebook takes about 10 minutes on a laptop CPU without TabPFN, and resumes from `results/checkpoints/` if interrupted.

## Status

- [x] EDA: missing values, target, leakage, distributions, outliers, redundancy
- [x] Preprocessing: fixed rules (`prepare`), per-fold statistics (`stats_cleaner`), leakage ablation, locked temporal split
- [x] Untuned baselines (CDRS, LR, RF, XGBoost) with repeated CV and class weighting
- [ ] TabPFN baseline
- [ ] Collinearity reduction and SHAP/permutation importance
- [ ] Tuning, threshold selection, statistical comparison
- [ ] Final evaluation on the locked temporal test part
