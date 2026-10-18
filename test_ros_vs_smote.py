"""Cross-apply the SAME ROS-style test oracles to ROS and SMOTE.

This is a controlled demonstration of TEST TRANSFER, not an evaluation
of generalization to other datasets or unseen pipeline implementations.

Run together with the existing 8 tests:
    python -m pytest test_leakage.py test_ros_vs_smote.py -v -s
"""

from pathlib import Path

import numpy as np
import pytest

from pipelines import load_data, original_split, make_buggy_ros, make_buggy_smote
from oracles import assert_no_shared_sources, exact_test_duplicate_count

DATA_FILE = Path(__file__).with_name("diabetes.csv")


def assert_no_exact_test_copies(generated_rows, test_rows):
    """ROS-style exact-copy oracle: reject copies of held-out test records."""
    n_duplicates = exact_test_duplicate_count(generated_rows, test_rows)
    assert n_duplicates == 0, (
        f"Exact Test duplicates in generated training rows: {n_duplicates}"
    )


@pytest.fixture(scope="module")
def examples():
    X_all, y_all = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test = original_split(X_all, y_all)
    ros_model, ros = make_buggy_ros(X_all, y_all, X_train, y_train)
    _, smote = make_buggy_smote(X_all, y_all, X_train, y_train)

    # The appended ROS rows are copies of their original source records.
    # Transform them and Test with the EXACT SAME fitted ROS preprocessor.
    ros_copies_pre = ros_model[:-1].transform(
        X_all.loc[ros["appended_source_ids"]]
    )
    ros_test_pre = ros_model[:-1].transform(X_test)

    # TracedSMOTE records the exact generated rows and original transformed
    # features; fetch the original Test positions (not new synthetic indices).
    smote_test_positions = X_all.index.get_indexer(X_test.index)
    assert np.all(smote_test_positions >= 0)
    smote_test_pre = smote["original_preprocessed_features"][smote_test_positions]

    return {
        "test_ids": X_test.index.to_numpy(),
        "ros_audit": ros,
        "smote_audit": smote,
        "ros_added": ros_copies_pre,
        "ros_test": ros_test_pre,
        "smote_added": smote["synthetic_features"],
        "smote_test": smote_test_pre,
    }


def test_original_ros_source_oracle_on_both_samplers(examples):
    """Reuse the EXACT SAME existing ROS oracle on both sampler inputs."""
    test_ids = examples["test_ids"]
    print("\nSAME source-isolation oracle — no changes between samplers:")
    for name, key in [("ROS", "ros_audit"), ("SMOTE", "smote_audit")]:
        input_ids = examples[key]["sampler_input_source_ids"]
        n_shared = np.intersect1d(input_ids, test_ids).size
        with pytest.raises(AssertionError, match="Source leakage detected"):
            assert_no_shared_sources(input_ids, test_ids)
        assert n_shared == len(test_ids)
        print(f"  Buggy {name}: DETECTED ({n_shared} Test source IDs in sampler input)")


def test_same_exact_copy_oracle_on_ros_and_smote(examples):
    """The same naive exact-copy detector catches ROS but misses SMOTE."""
    ros_count = exact_test_duplicate_count(examples["ros_added"], examples["ros_test"])
    smote_count = exact_test_duplicate_count(
        examples["smote_added"], examples["smote_test"]
    )

    # ROS: exact copies of Test appear among appended classifier rows.
    with pytest.raises(AssertionError, match="Exact Test duplicates"):
        assert_no_exact_test_copies(examples["ros_added"], examples["ros_test"])

    # SMOTE: this naive oracle PASSES, even though Test records were used.
    assert_no_exact_test_copies(examples["smote_added"], examples["smote_test"])
    assert ros_count > 0
    assert smote_count == 0
    assert np.isin(examples["smote_audit"]["synthetic_parent_source_ids"],
                 examples["test_ids"]).any()

    print("\nSAME exact-copy oracle — no changes between samplers:")
    print(f"  Buggy ROS:   DETECTED ({ros_count} generated copies of Test records)")
    print(f"  Buggy SMOTE: MISSED   ({smote_count} exact copies, despite parent leakage)")


def test_original_ros_source_oracle_on_generated_row_lineage(examples):
    """Same source-overlap assertion; SMOTE requires two-parent lineage."""
    test_ids = examples["test_ids"]
    ros_sources = examples["ros_audit"]["appended_source_ids"]
    smote_parents = examples["smote_audit"]["synthetic_parent_source_ids"]
    ros_rows_from_test = int(np.isin(ros_sources, test_ids).sum())
    smote_rows_using_test = int(np.isin(smote_parents, test_ids).any(axis=1).sum())

    with pytest.raises(AssertionError, match="Source leakage detected"):
        assert_no_shared_sources(ros_sources, test_ids)
    with pytest.raises(AssertionError, match="Source leakage detected"):
        assert_no_shared_sources(smote_parents.ravel(), test_ids)

    print("\nSAME source-overlap oracle on generated-row provenance:")
    print(f"  Buggy ROS:   DETECTED ({ros_rows_from_test} added rows copied from Test)")
    print(f"  Buggy SMOTE: DETECTED ({smote_rows_using_test} synthetic rows with a Test parent)")
    print("  (Different provenance extraction is required for SMOTE.)")
