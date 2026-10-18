"""Value and identity edge cases for the reusable oracle functions."""

import numpy as np
import pytest

from oracles import assert_no_shared_sources, exact_test_duplicate_count


def test_duplicate_count_counts_generated_rows_not_matching_pairs():
    added = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
    test = np.array([[1.0, 2.0], [1.0, 2.0]])
    assert exact_test_duplicate_count(added, test) == 2


def test_exact_comparison_does_not_accept_near_duplicates():
    added = np.array([[1.0, np.nextafter(2.0, 3.0)]])
    assert exact_test_duplicate_count(added, [[1.0, 2.0]]) == 0


def test_equal_numeric_values_match_across_int_and_float_dtypes():
    assert exact_test_duplicate_count(
        np.array([[1, 2]], dtype=np.int64),
        np.array([[1.0, 2.0]], dtype=np.float64),
    ) == 1


def test_signed_zeros_match():
    assert exact_test_duplicate_count([[-0.0, 1.0]], [[0.0, 1.0]]) == 1


def test_identical_bytes_with_different_numeric_values_do_not_match():
    added = np.array([[1.0]], dtype=np.float64)
    test = added.view(np.int64)
    assert added.tobytes() == test.tobytes()
    assert exact_test_duplicate_count(added, test) == 0


def test_large_integers_are_not_rounded_into_float_matches():
    assert exact_test_duplicate_count(
        np.array([[2**53 + 1]], dtype=np.int64),
        np.array([[float(2**53)]], dtype=np.float64),
    ) == 0


def test_empty_row_sets_have_no_duplicates():
    assert exact_test_duplicate_count(np.empty((0, 2)), [[1, 2]]) == 0
    assert exact_test_duplicate_count([[1, 2]], np.empty((0, 2))) == 0


@pytest.mark.parametrize("added,test", [
    ([1, 2], [[1, 2]]),
    ([[1, 2]], [[1]]),
    (np.empty((1, 0)), np.empty((1, 0))),
])
def test_duplicate_check_rejects_invalid_feature_shapes(added, test):
    with pytest.raises(ValueError, match="2D arrays"):
        exact_test_duplicate_count(added, test)


@pytest.mark.parametrize("invalid", [
    [[np.nan]], [[np.inf]], [[-np.inf]], [["1"]], [[1 + 2j]],
])
def test_duplicate_check_rejects_missing_nonfinite_or_nonnumeric_values(invalid):
    with pytest.raises(ValueError, match="finite real numeric"):
        exact_test_duplicate_count(invalid, [[1]])
    with pytest.raises(ValueError, match="finite real numeric"):
        exact_test_duplicate_count([[1]], invalid)


def test_source_check_accepts_repeated_noncontiguous_training_ids():
    assert_no_shared_sources([900, 100, 900], [300, 700])
    assert_no_shared_sources([], [300, 700])


def test_source_check_reports_distinct_forbidden_ids():
    with pytest.raises(AssertionError, match="1 distinct Test source IDs"):
        assert_no_shared_sources([100, 300, 300], [300, 700])


@pytest.mark.parametrize("invalid", [
    [[1, 2]], [-1], [True], [1.0], ["1"], None,
])
def test_source_check_rejects_invalid_id_records(invalid):
    with pytest.raises(ValueError, match="Source IDs"):
        assert_no_shared_sources(invalid, [300])
    with pytest.raises(ValueError, match="Source IDs"):
        assert_no_shared_sources([100], invalid)
