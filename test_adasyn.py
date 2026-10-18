"""ADASYN white-box leakage pilot; extends the existing ROS/SMOTE tests.

Run all tests:
    python -m pytest test_leakage.py test_ros_vs_smote.py test_adasyn.py -v -s

All assertions distinguish a detection oracle from a pytest test outcome:
pytest PASSED on a leaky case means the oracle correctly REJECTED that case.
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from imblearn.over_sampling import ADASYN

from pipelines import (
    SEED, load_data, original_split, make_reference_pipeline,
    make_buggy_adasyn, trace_adasyn_generation,
)
from test_leakage import assert_no_shared_sources
from test_ros_vs_smote import exact_test_duplicate_count

DATA_FILE = Path(__file__).with_name("diabetes.csv")


@pytest.fixture(scope="module")
def adasyn_cases():
    X_all, y_all = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test = original_split(X_all, y_all)

    reference = make_reference_pipeline("adasyn")
    sampler = reference.named_steps["sampling"]
    # Capture ACTUAL source IDs provided to the correct sampler by the pipeline.
    with patch.object(sampler, "fit_resample", wraps=sampler.fit_resample) as observed:
        reference.fit(X_train, y_train)
    assert observed.call_count == 1
    _, sampler_y = observed.call_args.args[:2]
    np.testing.assert_array_equal(sampler_y.index.to_numpy(), X_train.index.to_numpy())

    # Re-run UNMODIFIED ADASYN on the exact reference preprocessed Train
    # to independently verify its generated rows and source lineage.
    X_train_pre = reference[:-2].transform(X_train)
    reference_replay_sampler = ADASYN(
        sampling_strategy="minority", random_state=SEED, n_neighbors=5
    )
    ref_resampled, _ = reference_replay_sampler.fit_resample(X_train_pre, y_train)
    ref_new = np.asarray(ref_resampled)[len(X_train):]
    ref_trace = trace_adasyn_generation(
        X_train_pre, y_train, reference_replay_sampler, ref_new
    )

    mutant, mutant_audit = make_buggy_adasyn(
        X_all, y_all, X_train, y_train
    )
    return {
        "X_all": X_all, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "reference": reference, "mutant": mutant,
        "ref_trace": ref_trace, "ref_synthetic": ref_new,
        "audit": mutant_audit,
    }


def test_correct_adasyn_isolates_test_at_both_sampling_and_generation(adasyn_cases):
    c = adasyn_cases
    test_ids = c["X_test"].index.to_numpy()
    train_ids = c["X_train"].index.to_numpy()
    parents = train_ids[c["ref_trace"]["synthetic_parent_positions"]]
    difficulty_ids = train_ids[c["ref_trace"]["difficulty_neighbor_positions"]]
    assert_no_shared_sources(train_ids, test_ids)
    assert_no_shared_sources(parents.ravel(), test_ids)
    assert_no_shared_sources(difficulty_ids.ravel(), test_ids)
    assert len(c["ref_synthetic"]) == 187
    print("Correct ADASYN: no held-out Test sources consulted; 187 synthetic rows")


def test_buggy_adasyn_sampler_input_is_rejected(adasyn_cases):
    c = adasyn_cases
    test_ids = c["X_test"].index.to_numpy()
    sampler_ids = c["audit"]["sampler_input_source_ids"]
    with pytest.raises(AssertionError, match="Source leakage detected"):
        assert_no_shared_sources(sampler_ids, test_ids)
    assert np.intersect1d(sampler_ids, test_ids).size == 154
    print("Buggy ADASYN: all 154 Test source IDs found in sampling input")


def test_buggy_adasyn_parent_reconstruction_matches_real_outputs(adasyn_cases):
    c = adasyn_cases
    a = c["audit"]
    pairs = a["synthetic_parent_source_ids"]
    original_source_ids = c["X_all"].index
    # Map source IDs to array positions; never silently assume ID==position.
    positions = original_source_ids.get_indexer(pairs.ravel()).reshape(-1, 2)
    assert np.all(positions >= 0)
    features = a["original_preprocessed_features"]
    weights = a["synthetic_interpolation_weights"]
    expected = features[positions[:, 0]] + weights[:, None] * (
        features[positions[:, 1]] - features[positions[:, 0]]
    )
    np.testing.assert_allclose(expected, a["synthetic_features"], rtol=1e-12, atol=1e-12)
    assert len(pairs) == a["synthetic_rows_added"] == 224
    assert len(a["classifier_training_labels"]) == 838
    print("ADASYN provenance: all 224 synthetic rows reconstructed exactly")


def test_buggy_adasyn_direct_test_parent_is_detected(adasyn_cases):
    a = adasyn_cases["audit"]
    test_ids = adasyn_cases["X_test"].index.to_numpy()
    pairs = a["synthetic_parent_source_ids"]
    affected = np.isin(pairs, test_ids).any(axis=1)
    with pytest.raises(AssertionError, match="Source leakage detected"):
        assert_no_shared_sources(pairs.ravel(), test_ids)
    assert int(affected.sum()) == 88
    print("Buggy ADASYN: 88 synthetic training rows used Test parents")


def test_buggy_adasyn_difficulty_search_uses_test_records(adasyn_cases):
    a = adasyn_cases["audit"]
    test_ids = adasyn_cases["X_test"].index.to_numpy()
    difficulty_test = np.intersect1d(a["difficulty_neighbor_source_ids"], test_ids)
    direct_test = np.intersect1d(a["synthetic_parent_source_ids"], test_ids)
    neighbor_only = np.setdiff1d(difficulty_test, direct_test)
    assert difficulty_test.size == 93
    assert neighbor_only.size == 57
    with pytest.raises(AssertionError, match="Source leakage detected"):
        assert_no_shared_sources(a["difficulty_neighbor_source_ids"].ravel(), test_ids)
    print("Buggy ADASYN: 93 distinct Test sources in difficulty-neighbor search")
    print("  57 of these Test sources were NOT direct interpolation parents")
    print("  Neighbor consultation is observed; row-level causal effect is not claimed")


def test_same_exact_duplicate_oracle_misses_adasyn(adasyn_cases):
    a = adasyn_cases["audit"]
    X_all = adasyn_cases["X_all"]
    X_test = adasyn_cases["X_test"]
    test_positions = X_all.index.get_indexer(X_test.index)
    original_test_pre = a["original_preprocessed_features"][test_positions]
    exact_hits = exact_test_duplicate_count(
        a["synthetic_features"], original_test_pre
    )
    assert exact_hits == 0
    assert np.isin(
        a["synthetic_parent_source_ids"], X_test.index.to_numpy()
    ).any()
    print("Same exact-duplicate oracle: MISSED ADASYN (0 copies; 88 leaky synthetic rows)")
