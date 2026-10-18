"""Dynamic leakage-oracle stress tests for the Pima oversampling pilot.

Purpose
-------
Probe *limitations* of three existing runtime oracles rather than adding
another oversampling method or changing the LogisticRegression protocol.

Case H (real Pima): ROS illegally sees ALL original records, but we deliberately
exclude generated copies that originated from Test before classifier training.
This isolates the distinction between sampler access and direct data lineage.

Case N (minimal synthetic negative control): two different source records have
identical feature values. Train-only ROS legitimately copies the Train record.
An exact-feature duplicate detector therefore flags the independent Test record
although neither its source ID nor its contents were passed to the sampler.

Run (from project folder):
    python -m pytest test_dynamic_edge_cases.py -v -s

These cases demonstrate a *limited pilot*, not general detection rates.
"""

from pathlib import Path

import numpy as np
import pytest
from imblearn.over_sampling import RandomOverSampler

from pipelines import SEED, load_data, make_preprocessor, original_split
from test_leakage import assert_no_shared_sources
from test_ros_vs_smote import exact_test_duplicate_count

DATA_FILE = Path(__file__).with_name("diabetes.csv")


@pytest.fixture(scope="module")
def hidden_ros_access():
    """Run real ROS on all Pima records, then remove Test-originated copies.

    This remains a bad sampling procedure: sampler input includes all held-out
    records, and full-data class counts affect the oversampling process.
    Nevertheless, no *kept generated* row has a Test source ID.
    """
    X_all, y_all = load_data(DATA_FILE)
    X_train, X_test, y_train, _ = original_split(X_all, y_all)

    prep = make_preprocessor()
    prep.fit(X_train)  # Train-only fit, just as in our previous pilot
    X_all_pre = prep.transform(X_all)
    X_test_pre = prep.transform(X_test)

    sampler = RandomOverSampler(sampling_strategy="minority", random_state=SEED)
    X_resampled, _ = sampler.fit_resample(X_all_pre, y_all)

    n_original = len(X_all)
    appended_positions = sampler.sample_indices_[n_original:]
    original_ids = X_all.index.to_numpy()
    generated_ids = original_ids[appended_positions]

    # Explicitly discard copies sourced from Test; don't pretend the access
    # was safe simply because the direct contamination was screened out.
    from_train = np.isin(generated_ids, X_train.index.to_numpy())
    kept_ids = generated_ids[from_train]
    kept_generated = np.asarray(X_resampled)[n_original:][from_train]

    assert len(generated_ids) > len(kept_ids) > 0
    assert np.intersect1d(kept_ids, X_test.index.to_numpy()).size == 0

    return {
        "test_ids": X_test.index.to_numpy(),
        "sampler_input_ids": original_ids,
        "kept_generated_ids": kept_ids,
        "kept_generated_features": kept_generated,
        "test_features": np.asarray(X_test_pre),
        "total_new_rows": len(generated_ids),
        "discarded_test_copies": int((~from_train).sum()),
    }


def test_source_isolation_still_detects_hidden_ros_access(hidden_ros_access):
    """Original Source Isolation oracle detects forbidden sampler input."""
    c = hidden_ros_access
    with pytest.raises(AssertionError, match="Source leakage detected"):
        assert_no_shared_sources(c["sampler_input_ids"], c["test_ids"])
    print(
        "Source Isolation: DETECTED —",
        np.intersect1d(c["sampler_input_ids"], c["test_ids"]).size,
        "Test records entered ROS",
    )


def test_generated_lineage_oracle_misses_sampler_only_access(hidden_ros_access):
    """Output-parent oracle cannot see forbidden access if copies are filtered."""
    c = hidden_ros_access
    assert_no_shared_sources(c["kept_generated_ids"], c["test_ids"])
    assert c["discarded_test_copies"] > 0
    print(
        "Generated Provenance: MISSED sampler access —",
        len(c["kept_generated_ids"]),
        "Train-originated copies kept;",
        c["discarded_test_copies"],
        "Test-originated copies excluded",
    )


def test_exact_duplicate_oracle_misses_sampler_only_access(hidden_ros_access):
    """A test-copy equality check misses the same hidden input violation."""
    c = hidden_ros_access
    n_matches = exact_test_duplicate_count(
        c["kept_generated_features"], c["test_features"]
    )
    assert n_matches == 0
    print("Exact Duplicate: MISSED sampler access —", n_matches, "matching copies")


def test_exact_duplicate_false_alarm_with_independent_identical_features():
    """Synthetic clean case: matching measurements are not the same source.

    Record 101 (Train) and record 999 (Test) are *declared independent* but
    have coincidentally identical feature values. The test is a logical
    counterexample; it is not an empirical claim about Pima duplicates.
    """
    train_features = np.array([
        [1.0, 2.0],  # minority Train record, source 101
        [5.0, 6.0],  # majority Train record, source 102
        [7.0, 8.0],  # majority Train record, source 103
    ])
    train_labels = np.array([1, 0, 0])
    train_ids = np.array([101, 102, 103])

    # Separately identified held-out observation with identical measurements.
    independent_test_features = np.array([[1.0, 2.0]])
    independent_test_ids = np.array([999])

    sampler = RandomOverSampler(sampling_strategy="minority", random_state=SEED)
    resampled, _ = sampler.fit_resample(train_features, train_labels)
    added_rows = np.asarray(resampled)[len(train_features):]
    added_sources = train_ids[sampler.sample_indices_[len(train_features):]]

    # Data-provenance invariant passes: Test was never passed to the sampler.
    assert_no_shared_sources(train_ids, independent_test_ids)
    assert_no_shared_sources(added_sources, independent_test_ids)

    # Yet comparing feature bytes produces a spurious warning.
    count = exact_test_duplicate_count(added_rows, independent_test_features)
    assert count == 1
    print(
        "Exact Duplicate: FALSE POSITIVE in synthetic clean control —",
        count,
        "feature match but different independent source IDs",
    )
