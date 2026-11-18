# Diabetes Classification

Exploratory analysis and classification on a public Chinese diabetes dataset. The goal is to detect undiagnosed prediabetes/diabetes using only non-glycemic features (anthropometrics, blood pressure, lipids, liver/kidney function, blood count, lifestyle).

**Current result in one line.** Without any glucose test, undiagnosed DM can be ranked moderately well (CV AUROC about 0.79 with routine labs, 0.75 for the clinical CDRS score), while prediabetes cannot be separated from normal glucose regulation in a useful way.

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
- processed/: output of `01-preprocessing.ipynb` (clean dataset, feature sets, preprocessing log, leakage audit).

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
| `01-preprocessing.ipynb` | Rule-based cleaning, leakage removal, feature sets, 80/20 split |
| `02-modeling.ipynb` | Untuned baselines with repeated stratified CV, checkpointing, result export |
| `data/` | Raw CSV and data dictionary (unchanged) |
| `data/processed/` | Clean dataset, feature sets, preprocessing log, leakage audit |
| `results/` | All CV outputs (see below) |
| `figures/` | Proposed framework diagram |

## Preprocessing (`01-preprocessing.ipynb`)

Only rules that do not learn from the data are applied here. Imputation and scaling learn statistics from the data, so they live inside the modeling pipeline and are fitted on the training folds only.

1. **Subjects.** People with a known diabetes diagnosis (`DMhis = 1`) are excluded, leaving 5,873 people.
2. **Value fixes.** Blank `HypertenHis`/`DyslipideHis` mean "no" and are set to 0. Gender is recoded to `Male` (1 = male). The "9 = unknown" code in the disease-history items, `WeightLoss`, `FatherDM` and `MotherDM` becomes missing, because the model would otherwise treat 9 as a larger value than 1.
3. **Implausible values become missing.** These are creatinine below 35 µmol/L (327 people, together with the eGFR computed from it), height below 120 cm, systolic not above diastolic pressure, and LDL above total cholesterol.
4. **Leakage and unused columns.** The 17 leaking columns from the EDA are dropped, plus other glycemic tests, study markers and complications. Three more columns are dropped as a precaution. `FINS` and `FCP` (fasting insulin and C-peptide) are markers of glucose metabolism rather than routine labs. `DMhighrisk` has no published definition, and Chinese high-risk criteria usually include a history of impaired glucose regulation.
5. **Redundant, sparse and near-constant columns.** Cut-off versions of a numeric column (BMI classes, waist flags, GFR stages and so on) are dropped in favour of the numeric value. Columns more than 50% missing are dropped, as are binary columns whose minority class has fewer than 20 people (for example `MI` with 2 cases).
6. **Feature sets.** F1 (non-lab, 25 features) holds what can be collected without drawing blood, namely demographics, anthropometrics, blood pressure, lifestyle and history. F2 (combined, 71 features) adds 46 routine non-glycemic labs (liver, lipids, kidney, urine, complete blood count).
7. **Leakage audit.** Each final feature is scored on its own against the abnormal-vs-NGR label, both by value and by missingness. No feature exceeds an AUROC of 0.65, and missingness carries no class signal.
8. **Split.** A stratified 80/20 split gives 4,698 training and 1,175 test people. **The test set is locked** and has not been used yet.

Every change is listed in `data/processed/preprocessing_log.csv`.

## Experimental Setup (`02-modeling.ipynb`)

| | |
|---|---|
| Tasks | `y_dm` DM vs not DM (prevalence 5.2%) · `y_abn` prediabetes or DM vs NGR (24.3%) · `y3` three classes |
| Feature sets | F1 non-lab · F2 combined · CDRS components |
| Models | Logistic Regression, Random Forest, XGBoost, TabPFN, all with **default hyperparameters** |
| Clinical baseline | Chinese Diabetes Risk Score (CDRS) with the standard cut-off of 25, not fitted to the data, binary tasks only |
| Imbalance | `none` and `class_weight` (balanced weights, XGBoost via sample weights) |
| Validation | Repeated stratified 5-fold × 3 on the training set only. Folds are identical across all runs, so comparisons are paired by fold |
| Preprocessing in pipeline | Median imputation (all models except TabPFN), standard scaling (LR) |

The primary metrics are AUROC and AUPRC, because they do not depend on a threshold. Threshold-based metrics use 0.5 (argmax for three classes), which is not meaningful for heavily imbalanced data without class weights. `sens_at_spec80` reports sensitivity when specificity is held at 80%, as a rough picture of screening use.

### Result files (`results/`)

| File | Content |
|---|---|
| `cv_summary.csv` | Mean and std of every metric per run |
| `cv_fold_metrics.csv` | Every metric per run per fold (use this for paired tests) |
| `rq1_lab_vs_nonlab.csv` | Paired F2 − F1 difference per model |
| `rq3_model_comparison.csv` | Paired difference of each model against LR |
| `oof/<run_id>.csv` | Out-of-fold predicted probabilities per person |
| `importance/<run_id>.csv` | Feature importance averaged over folds |
| `confusion/<run_id>.csv` | Confusion matrix (summed over folds, divided by repeats) |
| `figures/` | AUROC summary plot and top-15 feature importance |
| `run_log.csv`, `config.json`, `cv_folds.csv` | Run status, configuration and library versions, fold assignment |
| `checkpoints/` | One file per fold. Re-running the notebook skips finished folds (git-ignored) |

## Baseline Results (untuned, 5-fold × 3 CV on the training set)

Values are mean ± std over 15 folds.

### DM vs not DM

| Model | Features | AUROC | AUPRC | Sens. at 80% spec. |
|---|---|---|---|---|
| CDRS | CDRS | 0.754 ± 0.035 | 0.147 | 0.53 |
| LR | F1 non-lab | 0.764 ± 0.031 | 0.162 | 0.56 |
| **LR** | **F2 combined** | **0.786 ± 0.022** | **0.260** | **0.61** |
| RF (class_weight) | F2 combined | 0.776 ± 0.015 | 0.194 | 0.56 |
| XGBoost | F2 combined | 0.776 ± 0.020 | 0.235 | 0.60 |

### Prediabetes or DM vs NGR

| Model | Features | AUROC | AUPRC | Sens. at 80% spec. |
|---|---|---|---|---|
| CDRS | CDRS | 0.676 ± 0.018 | 0.390 | 0.40 |
| LR | F1 non-lab | 0.679 ± 0.020 | 0.400 | 0.44 |
| **LR** | **F2 combined** | **0.700 ± 0.020** | **0.448** | **0.46** |
| RF (class_weight) | F2 combined | 0.698 ± 0.019 | 0.431 | 0.46 |
| XGBoost | F2 combined | 0.676 ± 0.017 | 0.427 | 0.43 |

### Three classes

| Model | Features | Imbalance | AUROC (OvR) | Macro-F1 | Recall NGR / Pre / DM |
|---|---|---|---|---|---|
| LR | F2 combined | none | 0.708 ± 0.016 | 0.390 | 0.98 / 0.05 / 0.15 |
| LR | F2 combined | class_weight | 0.679 ± 0.015 | 0.413 | 0.62 / 0.37 / 0.49 |
| XGBoost | F2 combined | class_weight | 0.689 ± 0.014 | 0.421 | 0.92 / 0.15 / 0.15 |
| RF | F2 combined | class_weight | 0.698 ± 0.021 | 0.296 | 1.00 / 0.00 / 0.01 |

The full table for all 40 runs is in `results/cv_summary.csv`. TabPFN has not run yet (see Next Steps).

### Interpretation

**DM is detectable without glucose, prediabetes largely is not.** Every model separates DM from the rest reasonably well, but none can single out prediabetes (AUROC of prediabetes against the rest stays near 0.63). The EDA explains why. Prediabetes shows the same direction of change as DM in age, waist, blood pressure and liver enzymes, but the shift from NGR is small. By definition, prediabetic glucose lies in a narrow band just above normal, so many prediabetic people are physiologically close to the NGR group. The label is defined by a glucose threshold, and the non-glycemic profile changes gradually rather than at that threshold.

**Routine labs add a small but consistent amount (RQ1).** Adding the 46 labs improves AUROC by about 0.02 for LR and 0.03 to 0.05 for the tree models, and F2 beats F1 in 67–100% of the paired folds. The gain is real in direction but modest in size. The larger gain for tree models suggests they benefit more from the extra variables, and the larger rise in AUPRC for DM (0.16 to 0.26) means the labs mainly help rank the highest-risk people. The most important labs across RF and XGBoost are GGT, FIB-4, AST, neutrophils, triglycerides and urine albumin. This fits the known link between fatty liver, low-grade inflammation, early kidney damage and insulin resistance.

**Without labs, machine learning does not beat the clinical score.** Non-lab LR is only about 0.01 AUROC above CDRS, well inside the fold-to-fold variation. CDRS uses almost the same information (age, sex, BMI, waist, blood pressure, family history) and needs no training. The paired advantage of LR with labs over CDRS for DM is 0.032 ± 0.029, roughly one standard deviation, so it has to be confirmed on the locked test set before it can be claimed.

**The simplest model is the best one so far (RQ3).** LR ranks first in every task. With a weak, mostly monotonic signal (older, larger waist, higher GGT means higher risk) a linear model has little to lose, while default RF and XGBoost overfit (fully grown trees, no regularisation tuning). Tuning may narrow the gap, but large improvements are unlikely when the ceiling is set by the information in the features.

**Class weighting trades accuracy on NGR for sensitivity but does not change ranking.** For LR it barely moves AUROC but shifts the default threshold, raising DM sensitivity from 0.10 to 0.61 at a specificity of 0.78. For default RF it has almost no effect, most likely because fully grown trees already fit the training data almost perfectly, so reweighting changes little. Since AUROC is unchanged, choosing a threshold on the validation folds is a simpler fix than resampling.

**LR coefficients are not interpretable yet.** The largest LR coefficients belong to blood-count indices (MCH, MCV, lymphocyte percentage, monocyte count) and vary strongly between folds. Several of these variables are mathematically tied (differential percentages sum to 100%, MCH/MCV/MCHC derive from the same red cell measurements, waist/hip/BMI overlap), so the model can split weight between them almost arbitrarily. The tree-based importance rankings are more stable and more clinically plausible.

## Limitations

- Single centre (Shanghai), collected in 2012, with no published inclusion criteria, so the population may not resemble a general screening population.
- All results come from cross-validation on the training set. The test set has not been touched, and no external validation set exists.
- CDRS points were implemented from the published scoring table and still need to be checked line by line against Zhou et al. (2013) before reporting.
- Hyperparameters are defaults, so the model comparison is preliminary.

## Next Steps

1. Reduce collinear blood-count features (keep either percentages or counts) and recompute importance with permutation importance or SHAP.
2. Run TabPFN. The model weights are gated on Hugging Face, so accept the licence on the model page, log in with `hf auth login` (or set `HF_TOKEN`), install `tabpfn`, and re-run `02-modeling.ipynb`, preferably on a GPU. Finished folds are loaded from the checkpoints.
3. Tune RF and XGBoost with nested CV, and select thresholds on the validation folds.
4. Add a statistical comparison of paired fold results (for example a corrected resampled t-test) for RQ1 and RQ3.
5. Evaluate the final models once on the locked test set, with bootstrap confidence intervals and calibration plots.

## Contribution

As of Oct 2026, no published study has used this dataset (0 citations on OpenAlex), and the data descriptor reports no ML baseline. This work contributes:

1. **The first ML baseline** on this dataset, compared against an established clinical risk score (CDRS).
2. **Documented, reproducible label leakage.** The target is fully recomputable from FPG/PG2h, and two undocumented or under-documented columns (`VAR00001`, `ThalussemiaAND`) leak it.
3. **A leakage-free setup for undiagnosed diabetes screening without glucose tests.** Known diabetics are excluded, and all glycemic, derived and leaking columns are removed.
4. **An estimate of how much routine non-glycemic labs add** beyond questionnaire and anthropometric data.

Every claim above can be checked directly from the raw CSV with the notebooks.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order: `eda-diabetes.ipynb` (optional), `01-preprocessing.ipynb`, `02-modeling.ipynb`.

## Status

- [x] EDA: missing values, target, leakage, distributions, outliers, redundancy
- [x] Preprocessing: subject selection, value fixes, leakage removal, redundancy pruning, locked 80/20 split
- [x] Untuned baselines (CDRS, LR, RF, XGBoost) with repeated CV and class weighting
- [ ] TabPFN baseline
- [ ] Collinearity reduction and SHAP/permutation importance
- [ ] Tuning, threshold selection, statistical comparison
- [ ] Final evaluation on the locked test set
