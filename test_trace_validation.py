"""Incomplete trace data must not be reported as a successful check."""

from dataclasses import replace

import numpy as np
import pytest

from detection_matrix import Scenario, run_oracles


@pytest.fixture
def clean_case():
    return Scenario(
        name="clean example", leaky=False,
        sampler_input_ids=np.array([100, 200]),
        generated_features=np.array([[1.0, 2.0], [3.0, 4.0]]),
        generated_parent_ids=np.array([100, 200]),
        test_features=np.array([[9.0, 8.0]]),
        test_ids=np.array([300]),
    )


@pytest.mark.parametrize("parents", [
    np.empty((0, 2), dtype=int), np.array([100]), np.empty((2, 0), dtype=int), None,
])
def test_missing_or_misaligned_parents_are_rejected(clean_case, parents):
    with pytest.raises(ValueError, match="Parent traces"):
        run_oracles(replace(clean_case, generated_parent_ids=parents))


def test_unknown_parent_ids_are_rejected(clean_case):
    with pytest.raises(ValueError, match="parent ID is missing"):
        run_oracles(replace(clean_case, generated_parent_ids=np.array([100, 999])))


@pytest.mark.parametrize("ids", [None, np.array([]), np.array([100, 100])])
def test_invalid_input_identity_records_are_rejected(clean_case, ids):
    with pytest.raises(ValueError, match="original integer IDs"):
        run_oracles(replace(clean_case, sampler_input_ids=ids))


def test_test_features_must_align_with_test_ids(clean_case):
    with pytest.raises(ValueError, match="matching lengths"):
        run_oracles(replace(clean_case, test_ids=np.array([300, 400])))


def test_valid_empty_generation_is_not_a_missing_trace(clean_case):
    case = replace(
        clean_case, generated_features=np.empty((0, 2)),
        generated_parent_ids=np.empty((0,), dtype=int),
    )
    flags, evidence = run_oracles(case)
    assert not any(flags.values())
    assert evidence["added_rows"] == 0


def test_detection_uses_evidence_not_scenario_label(clean_case):
    clean_flags, _ = run_oracles(clean_case)
    relabeled_flags, _ = run_oracles(replace(clean_case, name="Buggy example", leaky=True))
    assert clean_flags == relabeled_flags
