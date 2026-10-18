"""Phase 2 — ROS and SMOTE source-isolation/provenance checks (pilot).

Run from the project folder:
    python -m pytest test_leakage.py -v -s

These are white-box provenance checks, not a general static leakage detector.
"""
from pathlib import Path

import numpy as np
import pytest

from oracles import assert_no_shared_sources
from pipelines import (
    load_data,
    original_split,
    make_reference_pipeline,
    make_buggy_ros,
)

DATA_FILE = Path(__file__).with_name("diabetes.csv")


@pytest.fixture(scope="module")
def data_partitions():
    X_all, y_all = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test = original_split(X_all, y_all)
    return X_all, y_all, X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def buggy_audit(data_partitions):
    X_all, y_all, X_train, _, y_train, _ = data_partitions
    _, audit = make_buggy_ros(X_all, y_all, X_train, y_train)
    return audit


def test_correct_ros_keeps_test_sources_isolated(data_partitions):
    _, _, X_train, X_test, y_train, _ = data_partitions
    model = make_reference_pipeline("ros")
    model.fit(X_train, y_train)

    # sample_indices_ holds positional indices into original TRAIN input.
    positions = model.named_steps["sampling"].sample_indices_
    ros_sources = X_train.index.to_numpy()[positions]
    assert_no_shared_sources(ros_sources, X_test.index.to_numpy())


def test_buggy_ros_sampler_access_is_detected(data_partitions, buggy_audit):
    _, _, _, X_test, _, _ = data_partitions
    test_ids = X_test.index.to_numpy()
    sampler_inputs = buggy_audit["sampler_input_source_ids"]

    with pytest.raises(AssertionError, match="Source leakage detected"):
        assert_no_shared_sources(sampler_inputs, test_ids)

    print("Distinct Test sources in ROS input:",
          np.intersect1d(sampler_inputs, test_ids).size)


def test_buggy_ros_training_contamination_is_detected(data_partitions, buggy_audit):
    _, _, _, X_test, _, _ = data_partitions
    test_ids = X_test.index.to_numpy()
    training_sources = buggy_audit["classifier_training_source_ids"]

    with pytest.raises(AssertionError, match="Source leakage detected"):
        assert_no_shared_sources(training_sources, test_ids)

    print("Distinct Test sources in classifier training:",
          np.intersect1d(training_sources, test_ids).size)
    print("Classifier training rows sourced from Test:",
          np.isin(training_sources, test_ids).sum())


@pytest.fixture(scope="module")
def buggy_smote_audit(data_partitions):
    from pipelines import make_buggy_smote
    X_all, y_all, X_train, _, y_train, _ = data_partitions
    _, audit = make_buggy_smote(X_all, y_all, X_train, y_train)
    return audit


def test_correct_smote_keeps_test_sources_isolated(data_partitions):
    """Observe the actual source identities passed to Train-only SMOTE."""
    from unittest.mock import patch

    _, _, X_train, X_test, y_train, _ = data_partitions
    model = make_reference_pipeline("smote")
    sampler = model.named_steps["sampling"]

    with patch.object(sampler, "fit_resample", wraps=sampler.fit_resample) as observed:
        model.fit(X_train, y_train)

    assert observed.call_count == 1
    _, observed_y = observed.call_args.args[:2]
    # This assertion requires source-index preservation at sampler input.
    assert hasattr(observed_y, "index")
    np.testing.assert_array_equal(observed_y.index.to_numpy(), X_train.index.to_numpy())
    assert_no_shared_sources(observed_y.index.to_numpy(), X_test.index.to_numpy())


def test_buggy_smote_sampler_access_is_detected(data_partitions, buggy_smote_audit):
    """A full-data sampler must be rejected even without exact duplicates."""
    _, _, _, X_test, _, _ = data_partitions
    test_ids = X_test.index.to_numpy()
    sampler_inputs = buggy_smote_audit["sampler_input_source_ids"]

    with pytest.raises(AssertionError, match="Source leakage detected"):
        assert_no_shared_sources(sampler_inputs, test_ids)

    shared = np.intersect1d(sampler_inputs, test_ids)
    assert shared.size == len(test_ids)
    assert buggy_smote_audit["synthetic_rows_added"] == 232
    assert buggy_smote_audit["synthetic_parent_lineage_recorded"] is True
    print("Distinct Test sources in SMOTE input:", len(shared))
    print("Synthetic rows generated:", buggy_smote_audit["synthetic_rows_added"])


def _reconstruct_smote_rows(audit):
    """Map original source IDs back to feature positions before indexing."""
    pairs = audit["synthetic_parent_source_ids"]
    source_ids = audit["sampler_input_source_ids"]
    assert len(np.unique(source_ids)) == len(source_ids)
    id_to_position = {source_id: position for position, source_id in enumerate(source_ids)}
    try:
        positions = np.array([
            [id_to_position[parent] for parent in pair] for pair in pairs
        ], dtype=int).reshape(-1, 2)
    except KeyError as error:
        raise AssertionError("SMOTE parent ID is missing from sampler inputs") from error

    originals = audit["original_preprocessed_features"]
    weights = audit["synthetic_interpolation_weights"]
    return originals[positions[:, 0]] + weights[:, None] * (
        originals[positions[:, 1]] - originals[positions[:, 0]]
    )


def test_buggy_smote_parent_trace_reconstructs_every_synthetic_row(buggy_smote_audit):
    """Actual parent pairs + weights must reproduce every SMOTE output row."""
    audit = buggy_smote_audit
    pairs = audit["synthetic_parent_source_ids"]
    weights = audit["synthetic_interpolation_weights"]
    generated = audit["synthetic_features"]

    assert pairs.shape == (audit["synthetic_rows_added"], 2)
    assert weights.shape == (audit["synthetic_rows_added"],)
    reconstructed = _reconstruct_smote_rows(audit)
    np.testing.assert_allclose(reconstructed, generated, rtol=1e-12, atol=1e-12)
    print("Synthetic rows with verified two-parent provenance:", len(pairs))


def test_smote_parent_trace_with_noncontiguous_source_ids(data_partitions):
    """Source IDs identify rows; they are not positions into a feature array."""
    from pipelines import make_buggy_smote

    X_all, y_all, X_train, _, y_train, _ = data_partitions
    X_all, y_all = X_all.copy(), y_all.copy()
    X_train, y_train = X_train.copy(), y_train.copy()
    for values in (X_all, y_all, X_train, y_train):
        values.index = values.index * 10 + 1000

    _, audit = make_buggy_smote(X_all, y_all, X_train, y_train)
    assert audit["synthetic_parent_source_ids"].min() >= len(X_all)
    np.testing.assert_allclose(
        _reconstruct_smote_rows(audit), audit["synthetic_features"],
        rtol=1e-12, atol=1e-12,
    )


def test_smote_parent_trace_rejects_unknown_source_id(buggy_smote_audit):
    audit = buggy_smote_audit.copy()
    audit["synthetic_parent_source_ids"] = audit["synthetic_parent_source_ids"].copy()
    audit["synthetic_parent_source_ids"][0, 0] = 999999
    with pytest.raises(AssertionError, match="parent ID is missing"):
        _reconstruct_smote_rows(audit)


def test_buggy_smote_direct_test_parentage_is_detected(data_partitions, buggy_smote_audit):
    """Detect held-out Test sources used as either interpolation parent."""
    _, _, _, X_test, _, _ = data_partitions
    pairs = buggy_smote_audit["synthetic_parent_source_ids"]
    test_ids = X_test.index.to_numpy()

    uses_test_base = np.isin(pairs[:, 0], test_ids)
    uses_test_neighbor = np.isin(pairs[:, 1], test_ids)
    uses_test_either = uses_test_base | uses_test_neighbor
    direct_test_sources = np.intersect1d(pairs.ravel(), test_ids)

    # The oracle rejects any Test parent; pytest sees successful detection.
    with pytest.raises(AssertionError, match="Source leakage detected"):
        assert_no_shared_sources(pairs.ravel(), test_ids)

    assert uses_test_either.sum() > 0
    print("Synthetic rows with a Test base parent:", uses_test_base.sum())
    print("Synthetic rows with a Test neighbor parent:", uses_test_neighbor.sum())
    print("Synthetic rows with either Test parent:", uses_test_either.sum())
    print("Distinct Test sources directly used as parents:", len(direct_test_sources))


def test_exact_duplicate_check_misses_smote_parent_leakage(data_partitions, buggy_smote_audit):
    """Show why a naive EXACT-DUPLICATE oracle misses interpolated rows.

    This checks exact equality, not near-duplicates, source identity or
    upstream effects of Test data on the SMOTE neighborhood structure.
    """
    _, _, _, X_test, _, _ = data_partitions
    audit = buggy_smote_audit
    generated = audit["synthetic_features"]
    original = audit["original_preprocessed_features"]
    pairs = audit["synthetic_parent_source_ids"]
    test_ids = X_test.index.to_numpy()

    original_rows = {tuple(row) for row in original}
    exact_duplicates = sum(tuple(row) in original_rows for row in generated)
    direct_test_parentage = np.isin(pairs, test_ids).any(axis=1).sum()

    assert exact_duplicates == 0
    assert direct_test_parentage > 0
    print("Exact duplicate synthetic rows:", exact_duplicates)
    print("Synthetic rows with Test parents:", direct_test_parentage)
