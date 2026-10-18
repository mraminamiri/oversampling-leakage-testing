# Method

## Question and scope

This pilot asks which checks identify a deliberate violation of Train/Test isolation at the sampling stage. It studies one fault family using three samplers. It does not attempt to reproduce a published model or estimate leakage prevalence in other projects.

## Data and split

The supplied `diabetes.csv` has 768 records, eight numeric predictors, and target `Outcome`: 500 records in class 0 and 268 in class 1. Class 1 is the positive class in precision, recall, and F1.

The original row index is preserved as a source identifier. A stratified split with `test_size=0.20` and `random_state=42` creates 614 Train records (400/214) and 154 Test records (100/54). Source IDs identify observations; they are never predictor columns.

This split is reused throughout. There is no cross-validation or tuning in this pilot.

## Preprocessing

1. Replace zero with missing values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`.
2. Fit median imputation on original Train.
3. Fit standard scaling on the imputed original Train.
4. Transform other data using those fitted parameters.

The zero-to-missing rule is fixed in advance and learns no data-dependent parameter. Imputation and scaling are fitted only on original Train in both correct and faulty conditions. Scaling provides the feature space used for neighbor-based sampling and exact-feature comparisons.

## Seven model conditions

All classifiers are default `LogisticRegression()` models. ROS, SMOTE, and ADASYN use `sampling_strategy="minority"` and seed 42. SMOTE uses `k_neighbors=5`; ADASYN uses `n_neighbors=5`.

| Condition | Sampling and classifier fit |
|---|---|
| Baseline | Train-fitted preprocessing, then classifier on original Train. |
| Correct ROS / SMOTE / ADASYN | Preprocess original Train, resample Train only, then fit the classifier. |
| Buggy ROS / SMOTE / ADASYN | Fit preprocessing on original Train, transform all records, resample all records and labels, then fit on original Train plus only the added rows. |

The implementation checks that the sampler output begins with the original input before taking its added rows. Original Test rows are not directly appended, but their copies or synthetic descendants can enter classifier training.

The faulty protocol keeps the evaluation records fixed. It is not literal resampling before the split. Classifier training sizes are 614 for baseline; 800, 800, and 801 for correct ROS, SMOTE, and ADASYN; and 846, 846, and 838 for their faulty versions in the pinned environment.

## Three checks

**Source Isolation.** Intersect recorded sampler-input source IDs with Test IDs. Any overlap violates the sampler-input rule.

**Exact Duplicate.** Count added feature rows that are exactly numerically equal to at least one transformed Test row. A row is counted once even if several Test rows match it. This check compares values without an approximate-distance tolerance; it cannot establish common source identity.

**Generated Provenance.** Intersect the direct parents of added rows with Test IDs. ROS has one source parent per copy. SMOTE and ADASYN have two interpolation parents per synthetic row. This check concerns retained added rows, not all sampler accesses or indirect influences.

The checks receive recorded inputs and generated-row evidence. They do not inspect the scenario's `leaky` label when deciding whether to flag a violation. The known label is used afterward to describe outcomes as detected, missed, false positive, or pass.

## How provenance is obtained

- ROS exposes original input positions through `sample_indices_`.
- `TracedSMOTE` records the actual parent choices through an internal generation hook. Tests compare its output with ordinary SMOTE and reconstruct synthetic rows from parents and interpolation weights.
- ADASYN runs normally. A deterministic replay reconstructs parent positions and interpolation weights; the reconstructed rows are checked against the actual added rows. Difficulty-neighbor accesses are recorded separately.

Sampler-input IDs are partly supplied by the experiment functions. They are not collected by a separate generic runtime observer. These checks assume that source mappings and trace records are correct. A missing trace is not currently a general three-state “inconclusive” mechanism.

The matrix runner rejects missing or misaligned parent records, unknown parent IDs, and invalid input-ID records with an error. This validates trace structure; it does not prove that otherwise well-formed metadata describes the actual sampler call.

## Controls and results

The detection matrix has three correct negative controls and three faulty positive controls. All correct cases pass all three checks. Source Isolation and Generated Provenance flag each faulty case. Exact Duplicate flags faulty ROS but not faulty SMOTE or ADASYN.

In the fixed faulty runs, 54 ROS copies, 78 SMOTE rows, and 88 ADASYN rows have at least one direct Test parent. Every faulty sampler receives all 154 Test IDs. ADASYN's difficulty search also accesses Test records; direct interpolation parents alone do not describe that entire influence.

Two additional fixtures test the meaning of these checks. Filtering Test-origin ROS copies removes retained direct Test parents without undoing the earlier forbidden sampler access. An independently identified Test record with identical features creates an Exact Duplicate warning without a shared source. The latter is a constructed logical control, not a claim about a particular Pima patient.

## Model evaluation

All seven models predict the same original Test records. Class metrics use threshold 0.5. Probability-based metrics use the probability of class 1. Average Precision (AP) is calculated by `average_precision_score`, not trapezoidal integration of the precision-recall curve.

Performance is descriptive. Faulty scores are contaminated estimates, and the changes in training size and class balance prevent attributing score differences solely to leakage. A similar F1 score does not demonstrate that a pipeline used data correctly.

## Limits and next step

This is a small, white-box experiment. It does not establish general detection rates, clinical usefulness, or independence of all audit metadata. Internal SMOTE hooks and ADASYN replay are version-sensitive.

The next step is independent recording of actual sampler inputs, with explicit handling of missing or invalid evidence. Additional fault types, valid controls, seeds, and datasets are proposed work rather than completed experiments.
