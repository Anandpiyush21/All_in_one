# Methodology

This note records the design decisions behind `autoclf` and the reasoning for
each. It is the companion to the code; the README covers usage.

## 1. Problem statement

Given an arbitrary tabular CSV with a categorical target, produce an *honest*
estimate of how well a classifier would perform on data it has never seen, and
a report that a reader can audit. The emphasis is on the word *honest*: a tool
that reports optimistic numbers is worse than no tool, because the optimism is
invisible in the output.

## 2. Where accuracy estimates go wrong

Four failure modes dominate in practice, and the pipeline is arranged to
exclude each of them by construction rather than by discipline.

### 2.1 Preprocessing leakage

Scaling constants, imputation values, encoder vocabularies and feature rankings
are all *learned* from data. If they are learned once from the whole dataset
and the data is then split into folds, each fold's validation rows have already
influenced the transformation applied to the training rows. The resulting score
is biased upwards — the effect is small for centring and scaling, and large for
supervised feature selection on wide data, where selecting `k` features from
`p ≫ k` on the full sample can manufacture apparent signal from pure noise.

**Remedy.** The four stages `prep → scale → select → clf` form one
`sklearn.pipeline.Pipeline`. `cross_validate` clones and refits the entire
object on every training fold, so no statistic crosses the fold boundary. This
is asserted by `tests/test_pipeline.py::test_transformers_learn_from_training_rows_only`.

### 2.2 Inconsistent feature spaces

A subtler variant: fitting the transformers *separately* on the training and
test matrices. Both matrices then have the right shape, and the code runs
without error, but column *k* of one no longer denotes the same quantity as
column *k* of the other, so the test score is meaningless. (This was a defect in
version 1 of this project, and it is the reason a regression test now pins the
behaviour.)

**Remedy.** `fit` on train, `transform` on hold-out — enforced by using a single
fitted pipeline object for both.

### 2.3 Accuracy on an imbalanced target

The bundled dataset has a 4.9 : 1 class ratio. A classifier that answers "No"
unconditionally scores 83% accuracy and is worthless. Accuracy is reported
because readers expect it, but it is never the ranking criterion.

**Remedy.** Models are ranked by **macro F1**, which weights both classes
equally. **Balanced accuracy** and **Matthews' correlation coefficient** are
reported alongside; MCC is near zero for exactly the degenerate classifier
described above, which makes the failure legible at a glance. `class_weight="balanced"`
is applied by default wherever the estimator supports it.

### 2.4 Reading a difference that is not there

With five folds, a 0.004 gap in mean macro F1 between two models is well inside
fold-to-fold noise, but a table of means invites the reader to declare a winner.

**Remedy.** Every model sees *identical* splits, so per-fold scores are paired;
the report adds a paired *t*-test of the best model against each of the others
and prints the *p*-value. The report also states in the text that five folds
give the test little power, so "not significant" means *not distinguishable
here*, not *equivalent*.

## 3. Pipeline stages

| Stage | Component | Notes |
|---|---|---|
| `prep` | `ColumnTransformer` | median imputation + missingness indicator for numeric columns; most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`, `min_frequency=0.01`) for categorical ones |
| `scale` | one of 8 scalers | see `--list-options` |
| `select` | one of 9 selectors | filter, wrapper, embedded and projection families |
| `clf` | one of 11 classifiers | linear, kernel, instance-based, tree ensemble, boosting, neural |

The missingness indicator matters on survey data: a blank answer is itself
informative (a question may not have been asked of that respondent), so
discarding the pattern throws away signal.

`min_frequency=0.01` folds rare categories into an "infrequent" bucket, which
keeps the one-hot expansion bounded and prevents a category seen twice in
training from becoming a high-variance dummy.

## 4. Evaluation protocol

1. A stratified hold-out set (20% by default) is split off **once** and is not
   touched again until the final model has been fitted.
2. On the remaining 80%, stratified *k*-fold cross-validation produces six
   metrics per fold.
3. The pipeline is refitted on the full 80%. With `--tune`, a grid search runs
   here, nested inside the training data, so the hyper-parameter choice is also
   made without sight of the hold-out set.
4. The refitted pipeline is scored once on the hold-out set.
5. Permutation importance is computed on the hold-out set: each raw input column
   is shuffled in turn and the loss in macro F1 recorded, repeated five times.

Steps 2 and 4 answer different questions. Cross-validation estimates the
*expected* performance of the procedure; the hold-out score is a single draw
from data that influenced nothing. A large gap between them signals trouble,
which is why the report plots them side by side.

**Why permutation importance rather than tree importance.** Impurity-based
importance is computed on training data and is biased towards high-cardinality
features. Permutation importance is measured on unseen data, applies to any
estimator, and is expressed in the units of the metric one actually cares
about. Its weakness — correlated features share credit, so both can look
unimportant — is the reason the report prints the standard deviation next to
each value.

## 5. Reproducibility

Every run creates `runs/<name>/` containing `config.json`, `results.json` (with
Python, platform and scikit-learn versions), `summary.csv`, `run.log`,
`report.pdf`, `plots.pdf` and the individual figures. A single seed threads
through the split, the folds, the estimators and the permutations, so
`python main.py --config runs/<name>/config.json` reproduces the run exactly —
a property the test suite checks.

## 6. Known limitations

- **Grouped and temporal data.** `group_kfold` is registered but refused at
  runtime, because the tool does not collect a grouping column. Data with
  repeated measurements per subject needs one, or the folds will leak. Likewise
  `time_series` is available but the tool does not verify that the rows are
  ordered in time.
- **Wide data.** One-hot encoding of many high-cardinality categoricals can
  produce more columns than rows. The loader drops columns above 50 distinct
  values, which is a heuristic, not a solution; target or ordinal encoding
  inside the pipeline would be the principled fix.
- **No calibration.** Probabilities are used for ROC-AUC but are never
  calibrated. For decision-support use, `CalibratedClassifierCV` should wrap the
  final estimator.
- **Single hold-out.** One 80/20 split is one draw. Repeated nested
  cross-validation would give a variance estimate for the hold-out number too,
  at several times the compute cost.
- **No resampling.** Class imbalance is handled by re-weighting only; SMOTE and
  friends would need `imbalanced-learn` and, critically, must be applied inside
  the fold to avoid synthesising training points from validation rows.

## References

1. G. Varoquaux and O. Colliot, "Evaluating machine learning models and their
   diagnostic value", *Machine Learning for Brain Disorders*, 2023.
2. S. Kaufman, S. Rosset and C. Perlich, "Leakage in data mining: formulation,
   detection, and avoidance", *ACM TKDD* 6(4), 2012.
3. D. Chicco and G. Jurman, "The advantages of the Matthews correlation
   coefficient (MCC) over F1 score and accuracy in binary classification
   evaluation", *BMC Genomics* 21(6), 2020.
4. L. Breiman, "Random Forests", *Machine Learning* 45(1), 2001.
5. T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical
   Learning*, 2nd ed., Springer, 2009 — Ch. 7 on model assessment and the
   "wrong way to do cross-validation".
6. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python", *JMLR* 12,
   2011.
