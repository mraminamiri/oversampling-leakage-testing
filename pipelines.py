"""Correct and deliberately faulty oversampling pipelines for the Pima pilot.

The split stays fixed. Faulty samplers use all original records, then only
their added rows enter classifier training. Preprocessing remains Train-only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED = 42
TEST_SIZE = 0.20
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_MISSING_COLS = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
]


def load_data(csv_path=None):
    """Read the original CSV and preserve row positions as source IDs."""
    path = Path(csv_path) if csv_path is not None else Path(__file__).with_name("diabetes.csv")
    data = pd.read_csv(path).reset_index(drop=True)
    if data.columns.tolist() != FEATURES + ["Outcome"]:
        raise ValueError("Unexpected dataset columns or column order")
    X = data[FEATURES].copy()
    y = data["Outcome"].copy()
    return X, y


def original_split(X, y, seed=SEED):
    """Make the fixed original 80/20 stratified split BEFORE sampling."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )


def replace_invalid_zeros(X):
    """Treat zero in five measurement columns as missing; leave other zeros."""
    X = X.copy()
    X[ZERO_MISSING_COLS] = X[ZERO_MISSING_COLS].replace(0, np.nan)
    return X


def make_preprocessor():
    """Return a NEW, unfitted preprocessor for each experiment."""
    return Pipeline([
        ("zero_to_nan", FunctionTransformer(replace_invalid_zeros, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def make_reference_pipeline(method, seed=SEED):
    """Create a fresh Baseline, Correct ROS, SMOTE, or ADASYN pipeline.

    The sampler, if present, runs only during ``fit`` on training data.
    LogisticRegression uses default hyperparameters.
    """
    steps = make_preprocessor().steps
    classifier = ("classifier", LogisticRegression())

    if method == "baseline":
        return Pipeline(steps + [classifier])
    if method == "ros":
        sampler = RandomOverSampler(sampling_strategy="minority", random_state=seed)
    elif method == "smote":
        sampler = SMOTE(sampling_strategy="minority", random_state=seed, k_neighbors=5)
    elif method == "adasyn":
        sampler = ADASYN(sampling_strategy="minority", random_state=seed, n_neighbors=5)
    else:
        raise ValueError("method must be 'baseline', 'ros', 'smote', or 'adasyn'")

    return ImbPipeline(steps + [("sampling", sampler), classifier])


def make_buggy_ros(X_all, y_all, X_train_raw, y_train, seed=SEED):
    """Let ROS use all records, then add its copies to original Train.

    Return a fitted prediction pipeline and audit metadata. Do not refit the
    returned pipeline: sampling is performed here, outside that pipeline.
    Input IDs in the audit are supplied here, not by an independent logger.
    """
    from copy import deepcopy

    # Invariants: IDs refer to ORIGINAL records; no identity remapping.
    if not X_all.index.equals(y_all.index):
        raise ValueError("X_all and y_all must use the same original IDs")
    if not X_train_raw.index.equals(y_train.index):
        raise ValueError("X_train_raw and y_train indexes must match")
    if not X_all.index.is_unique:
        raise ValueError("Original source IDs must be unique")
    if not X_train_raw.index.isin(X_all.index).all():
        raise ValueError("Training IDs must be a subset of original IDs")

    # Identical Train-fitted preprocessing to the reference pipelines.
    prep = make_preprocessor()
    X_train_pre = prep.fit_transform(X_train_raw)
    X_all_pre = prep.transform(X_all)

    # Deliberate bug: sampler sees ALL original records and labels.
    sampler = RandomOverSampler(
        sampling_strategy="minority", random_state=seed
    )
    X_resampled, y_resampled = sampler.fit_resample(X_all_pre, y_all)
    X_resampled = np.asarray(X_resampled)
    y_resampled = np.asarray(y_resampled)

    n_original = len(X_all)
    np.testing.assert_array_equal(X_resampled[:n_original], X_all_pre)
    np.testing.assert_array_equal(y_resampled[:n_original], y_all.to_numpy())

    # Only added copies enter Train; original Test rows stay out of this slice.
    X_added = X_resampled[n_original:]
    y_added = y_resampled[n_original:]
    X_classifier_train = np.vstack([X_train_pre, X_added])
    y_classifier_train = np.concatenate([y_train.to_numpy(), y_added])

    clf = LogisticRegression()  # strictly default model parameters
    clf.fit(X_classifier_train, y_classifier_train)

    # Wrap already-fitted preprocessor and classifier for prediction.
    model = Pipeline(deepcopy(prep.steps) + [("classifier", clf)])

    original_source_ids = X_all.index.to_numpy(copy=True)
    appended_source_ids = original_source_ids[
        sampler.sample_indices_[n_original:]
    ]
    classifier_training_source_ids = np.concatenate([
        X_train_raw.index.to_numpy(copy=True), appended_source_ids
    ])
    audit = {
        "sampler_input_source_ids": original_source_ids,
        "appended_source_ids": appended_source_ids,
        "classifier_training_source_ids": classifier_training_source_ids,
        "classifier_training_labels": y_classifier_train.copy(),
        "sampler": sampler,
    }
    return model, audit



class TracedSMOTE(SMOTE):
    """Record the two parents chosen for each generated SMOTE row.

    This observes imbalanced-learn's protected _generate_samples hook. It does
    not draw random numbers or change the algorithm. Parent positions are
    relative to the original fit_resample input, NOT to any train/test split.

    This instrumentation was verified with imbalanced-learn 0.14.1; internal
    hooks can change in future library releases.
    """

    def _fit_resample(self, X, y):
        labels = np.asarray(y)
        self._original_class_positions = {
            label: np.flatnonzero(labels == label)
            for label in np.unique(labels)
        }
        self._parent_position_batches = []
        self._interpolation_weight_batches = []
        return super()._fit_resample(X, y)

    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps,
                          y_type=None, y=None):
        # These indices are chosen by SMOTE itself (not by this tracer).
        class_positions = self._original_class_positions[y_type]
        parents = np.column_stack([
            class_positions[rows],
            class_positions[nn_num[rows, cols]],
        ])
        self._parent_position_batches.append(parents)
        self._interpolation_weight_batches.append(np.asarray(steps).reshape(-1).copy())
        return super()._generate_samples(
            X, nn_data, nn_num, rows, cols, steps, y_type, y
        )

    @property
    def synthetic_parent_positions_(self):
        if not hasattr(self, "_parent_position_batches"):
            raise RuntimeError("Fit TracedSMOTE first")
        if not self._parent_position_batches:
            return np.empty((0, 2), dtype=int)
        return np.vstack(self._parent_position_batches)

    @property
    def synthetic_interpolation_weights_(self):
        if not hasattr(self, "_interpolation_weight_batches"):
            raise RuntimeError("Fit TracedSMOTE first")
        if not self._interpolation_weight_batches:
            return np.empty((0,), dtype=float)
        return np.concatenate(self._interpolation_weight_batches)

def make_buggy_smote(X_all, y_all, X_train_raw, y_train, seed=SEED):
    """Let SMOTE use all records and append its synthetic rows to Train.

    Preprocessing fits original Train only. TracedSMOTE records direct parents
    so they can be checked separately from sampler access. Return an already
    fitted prediction model and metadata; do not refit the returned model.
    """
    from copy import deepcopy

    if not X_all.index.equals(y_all.index):
        raise ValueError("X_all and y_all must have the same original IDs")
    if not X_train_raw.index.equals(y_train.index):
        raise ValueError("X_train_raw and y_train indexes must match")
    if not X_all.index.is_unique:
        raise ValueError("Original source IDs must be unique")
    if not X_train_raw.index.isin(X_all.index).all():
        raise ValueError("Training IDs must be a subset of original IDs")

    # Train-only preprocessing: this is NOT the injected fault.
    prep = make_preprocessor()
    X_train_pre = prep.fit_transform(X_train_raw)
    X_all_pre = prep.transform(X_all)

    # Deliberate fault: SMOTE receives all original records, including Test.
    sampler = TracedSMOTE(
        sampling_strategy="minority", random_state=seed, k_neighbors=5
    )
    X_resampled, y_resampled = sampler.fit_resample(X_all_pre, y_all)
    X_resampled = np.asarray(X_resampled)
    y_resampled = np.asarray(y_resampled)

    n_original = len(X_all)
    np.testing.assert_array_equal(X_resampled[:n_original], X_all_pre)
    np.testing.assert_array_equal(y_resampled[:n_original], y_all.to_numpy())

    # Add synthetic rows to original Train only.
    X_added = X_resampled[n_original:]
    y_added = y_resampled[n_original:]
    X_classifier_train = np.vstack([X_train_pre, X_added])
    y_classifier_train = np.concatenate([y_train.to_numpy(), y_added])

    clf = LogisticRegression()
    clf.fit(X_classifier_train, y_classifier_train)
    model = Pipeline(deepcopy(prep.steps) + [("classifier", clf)])

    # Convert positional parents to ORIGINAL dataset source IDs.
    parent_positions = sampler.synthetic_parent_positions_
    weights = sampler.synthetic_interpolation_weights_
    assert parent_positions.shape == (len(X_added), 2)
    assert weights.shape == (len(X_added),)
    parent_source_ids = X_all.index.to_numpy()[parent_positions]

    # Reconstruct every generated row; proves the provenance records are
    # aligned to the actual synthetic features, not merely the sampler input.
    reconstructed = (
        X_all_pre[parent_positions[:, 0]]
        + weights[:, None] * (
            X_all_pre[parent_positions[:, 1]] - X_all_pre[parent_positions[:, 0]]
        )
    )
    np.testing.assert_allclose(reconstructed, X_added, rtol=1e-12, atol=1e-12)

    audit = {
        "sampler_input_source_ids": X_all.index.to_numpy(copy=True),
        "original_training_source_ids": X_train_raw.index.to_numpy(copy=True),
        "synthetic_rows_added": len(X_added),
        "classifier_training_labels": y_classifier_train.copy(),
        "synthetic_parent_lineage_recorded": True,
        "synthetic_parent_source_ids": parent_source_ids.copy(),
        "synthetic_interpolation_weights": weights.copy(),
        "synthetic_features": X_added.copy(),
        "original_preprocessed_features": X_all_pre.copy(),
        "sampler": sampler,
    }
    return model, audit


def trace_adasyn_generation(X_original, y_original, fitted_sampler, synthetic_features):
    """Replay ADASYN 0.14.1's deterministic random choices to audit provenance.

    A normal imblearn.ADASYN runs first. We replay its integer-k
    neighbor searches and RNG using the SAME preprocessed input and seed.
    The replay must reconstruct EVERY synthetic row exactly (numerical tolerance)
    or raise an error rather than silently report incorrect parent identifiers.

    Supports the one-minority-class, integer n_neighbors setting of this pilot.
    The implementation follows imbalanced-learn 0.14.1 and should be reverified
    if the library implementation changes.

    Difficulty-neighbor membership shows which Test records were CONSULTED by
    the first, full-data neighborhood search. The influence of the normalized
    difficulty weights is global; this trace does not claim a unique causal
    attribution of each generated row to individual difficulty neighbors.
    """
    from numbers import Integral
    from sklearn.neighbors import NearestNeighbors
    from sklearn.utils import check_random_state

    X = np.asarray(X_original)
    y = np.asarray(y_original)
    generated = np.asarray(synthetic_features)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("Expected 2D X and 1D y with aligned lengths")
    if not isinstance(fitted_sampler.n_neighbors, Integral):
        raise NotImplementedError("Pilot tracer only supports integer n_neighbors")
    strategy = fitted_sampler.sampling_strategy_
    if len(strategy) != 1:
        raise NotImplementedError("Pilot tracer supports one oversampled class")

    class_label, requested_count = next(iter(strategy.items()))
    k = int(fitted_sampler.n_neighbors)
    class_positions = np.flatnonzero(y == class_label)
    X_class = X[class_positions]

    # Pass 1: ALL-class neighbors determine each minority sample's difficulty.
    neighbors = NearestNeighbors(n_neighbors=k + 1)
    neighbors.fit(X)
    difficulty_neighbors = neighbors.kneighbors(
        X_class, return_distance=False
    )[:, 1:]
    difficulty_ratios = (
        np.sum(y[difficulty_neighbors] != class_label, axis=1) / k
    )
    if not difficulty_ratios.sum():
        raise ValueError("ADASYN cannot allocate samples: all difficulty ratios are 0")
    allocations = np.rint(
        difficulty_ratios / difficulty_ratios.sum() * requested_count
    ).astype(int)
    n_synthetic = int(allocations.sum())

    # Pass 2: only same-class neighbors become interpolation parents.
    neighbors.fit(X_class)
    minority_neighbors = neighbors.kneighbors(
        X_class, return_distance=False
    )[:, 1:]
    rng = check_random_state(fitted_sampler.random_state)
    base_rows = np.repeat(np.arange(len(class_positions)), allocations)
    picked_columns = rng.choice(k, size=n_synthetic)
    second_rows = minority_neighbors[base_rows, picked_columns]
    interpolation_weights = rng.uniform(size=(n_synthetic, 1))

    parent_positions = np.column_stack([
        class_positions[base_rows], class_positions[second_rows]
    ])
    reconstructed = (
        X[parent_positions[:, 0]]
        + interpolation_weights * (
            X[parent_positions[:, 1]] - X[parent_positions[:, 0]]
        )
    ).astype(generated.dtype)

    if generated.shape != reconstructed.shape:
        raise AssertionError(
            f"ADASYN provenance mismatch: replay {reconstructed.shape}, "
            f"actual {generated.shape}"
        )
    np.testing.assert_allclose(
        reconstructed, generated, rtol=1e-12, atol=1e-12,
        err_msg="ADASYN provenance replay differs from actual generated features"
    )
    return {
        "synthetic_parent_positions": parent_positions,
        "synthetic_interpolation_weights": interpolation_weights.ravel().copy(),
        "difficulty_base_positions": class_positions.copy(),
        "difficulty_neighbor_positions": difficulty_neighbors.copy(),
        "difficulty_ratios_raw": difficulty_ratios.copy(),
        "generated_per_difficulty_base": allocations.copy(),
        "reconstruction_verified": True,
    }


def make_buggy_adasyn(X_all, y_all, X_train_raw, y_train, seed=SEED):
    """Full-data ADASYN mutant using the fixed original Test, as for SMOTE.

    Preprocessing fits only original Train. ADASYN erroneously sees ALL original
    records and labels. Add ONLY its generated rows to original Train, then
    train a default LogisticRegression. Keep Test fixed for evaluation.

    No samples are inserted from the original Test unchanged; however some
    synthetic rows can depend on Test parents and ADASYN's difficulty-neighbor
    search has already consulted Test records.
    """
    from copy import deepcopy

    if not X_all.index.equals(y_all.index):
        raise ValueError("X_all and y_all must have matching original IDs")
    if not X_train_raw.index.equals(y_train.index):
        raise ValueError("X_train_raw and y_train must have matching indices")
    if not X_all.index.is_unique:
        raise ValueError("Original source IDs must be unique")
    if not X_train_raw.index.isin(X_all.index).all():
        raise ValueError("Training IDs must belong to original data")

    prep = make_preprocessor()
    X_train_pre = prep.fit_transform(X_train_raw)
    X_all_pre = prep.transform(X_all)

    # Deliberate fault: sampling receives the original Test records too.
    sampler = ADASYN(
        sampling_strategy="minority", random_state=seed, n_neighbors=5
    )
    X_resampled, y_resampled = sampler.fit_resample(X_all_pre, y_all)
    X_resampled = np.asarray(X_resampled)
    y_resampled = np.asarray(y_resampled)
    n_original = len(X_all)
    np.testing.assert_array_equal(X_resampled[:n_original], X_all_pre)
    np.testing.assert_array_equal(y_resampled[:n_original], y_all.to_numpy())

    X_added = X_resampled[n_original:]
    y_added = y_resampled[n_original:]
    X_classifier_train = np.vstack([X_train_pre, X_added])
    y_classifier_train = np.concatenate([y_train.to_numpy(), y_added])

    classifier = LogisticRegression()
    classifier.fit(X_classifier_train, y_classifier_train)
    model = Pipeline(deepcopy(prep.steps) + [("classifier", classifier)])

    # Replay on the fitted real ADASYN output; fail if provenance is inaccurate.
    trace = trace_adasyn_generation(X_all_pre, y_all, sampler, X_added)
    source_ids = X_all.index.to_numpy(copy=True)
    audit = {
        "sampler_input_source_ids": source_ids.copy(),
        "original_training_source_ids": X_train_raw.index.to_numpy(copy=True),
        "synthetic_rows_added": len(X_added),
        "classifier_training_labels": y_classifier_train.copy(),
        "synthetic_parent_lineage_recorded": trace["reconstruction_verified"],
        "synthetic_parent_source_ids": source_ids[
            trace["synthetic_parent_positions"]
        ],
        "synthetic_interpolation_weights": trace[
            "synthetic_interpolation_weights"
        ].copy(),
        "difficulty_base_source_ids": source_ids[
            trace["difficulty_base_positions"]
        ],
        "difficulty_neighbor_source_ids": source_ids[
            trace["difficulty_neighbor_positions"]
        ],
        "difficulty_ratios_raw": trace["difficulty_ratios_raw"].copy(),
        "generated_per_difficulty_base": trace["generated_per_difficulty_base"].copy(),
        "synthetic_features": X_added.copy(),
        "original_preprocessed_features": X_all_pre.copy(),
        "sampler": sampler,
    }
    return model, audit
