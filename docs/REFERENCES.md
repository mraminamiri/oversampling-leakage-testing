# References and data note

These sources support the dataset background, sampling methods, and evaluation rules. The three check names in this repository are project labels; the code is not the official implementation of any cited paper.

## Dataset

- [OpenML dataset 37: diabetes](https://www.openml.org/search?type=data&id=37), with [machine-readable metadata](https://www.openml.org/api/v1/json/data/37). The record describes the Pima Indians Diabetes Database, gives 768 records and eight predictors, and identifies its source as UCI and its original owner as the National Institute of Diabetes and Digestive and Kidney Diseases.

The supplied file is `diabetes.csv`, using the column names `Pregnancies` through `Age` and binary `Outcome`. Its SHA-256 is:

```text
b78029447fae2743b3218bb2b76ef0d04afe8d7e55ce2faf4d1ec82d8f8ae8ac
```

The exact download page for this CSV was not recorded. The hash identifies the file used in this experiment; it does not establish its download history or a redistribution license. The OpenML ARFF uses different names and class labels, so this CSV is not presented as a byte-identical OpenML export.

## Sampling methods

- Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357. [Paper](https://www.jair.org/index.php/jair/article/view/10302), [DOI](https://doi.org/10.1613/jair.953).
- He, H., Bai, Y., Garcia, E. A., and Li, S. (2008). *ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning*. IJCNN, 1322–1328. [DOI](https://doi.org/10.1109/IJCNN.2008.4633969).
- imbalanced-learn API documentation: [RandomOverSampler](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html), [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html), and [ADASYN](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html).

The implementation uses imbalanced-learn 0.14.1. Online `stable` documentation can describe a newer version; the pinned source and output checks determine the behavior reproduced here.

## Leakage and metrics

- imbalanced-learn, [Common pitfalls: data leakage](https://imbalanced-learn.org/stable/common_pitfalls.html#data-leakage). Explains why using evaluation information during resampling violates separation. Its standard resample-before-split example motivates the rule; this project's fixed-Test mutation is a different controlled construction.
- scikit-learn, [Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Defines the AP metric used in the performance report and distinguishes it from trapezoidal area under a precision-recall curve.

The background literature does not establish novelty for this pilot. The contribution at this stage is the controlled comparison of the checks and explicit examples of their limitations.
