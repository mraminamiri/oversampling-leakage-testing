"""Compare three checks on six controlled sampling scenarios.

Run with ``python detection_matrix.py``. The checks use trace evidence;
scenario labels are used afterward to describe detection outcomes.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE

from pipelines import (
    SEED,
    TracedSMOTE,
    load_data,
    make_buggy_ros,
    make_buggy_smote,
    make_buggy_adasyn,
    trace_adasyn_generation,
    make_reference_pipeline,
    original_split,
)
from oracles import assert_no_shared_sources, exact_test_duplicate_count

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "diabetes.csv"
OUTPUT_DIR = ROOT / "results"


@dataclass
class Scenario:
    name: str
    leaky: bool
    sampler_input_ids: np.ndarray
    generated_features: np.ndarray
    generated_parent_ids: np.ndarray  # one ROS source or two SMOTE parents
    test_features: np.ndarray
    test_ids: np.ndarray
    difficulty_neighbor_ids: np.ndarray | None = None  # ADASYN-only audit


def make_scenarios():
    X_all, y_all = load_data(DATA_FILE)
    X_train, X_test, y_train, _ = original_split(X_all, y_all)
    test_ids = X_test.index.to_numpy()
    train_ids = X_train.index.to_numpy()
    n_train = len(X_train)

    # NEGATIVE CONTROL A: Correct ROS. The actual fitted sampler exposes
    # positions of all output rows; positions n_train: are added copies.
    correct_ros = make_reference_pipeline("ros")
    correct_ros.fit(X_train, y_train)
    ros_indices = correct_ros.named_steps["sampling"].sample_indices_
    ros_added_ids = train_ids[ros_indices[n_train:]]
    ros_generated = correct_ros[:3].transform(X_train.iloc[ros_indices[n_train:]])
    ros_test = correct_ros[:3].transform(X_test)

    correct_ros_case = Scenario(
        name="Correct ROS", leaky=False,
        sampler_input_ids=train_ids,
        generated_features=np.asarray(ros_generated),
        generated_parent_ids=ros_added_ids,
        test_features=np.asarray(ros_test), test_ids=test_ids,
    )

    # NEGATIVE CONTROL B: Correct SMOTE. Replay the SAME default SMOTE
    # with instrumentation, on the SAME train-fitted preprocessing.
    correct_smote = make_reference_pipeline("smote")
    correct_smote.fit(X_train, y_train)
    train_pre = correct_smote[:3].transform(X_train)
    smote_test = correct_smote[:3].transform(X_test)
    traced = TracedSMOTE(sampling_strategy="minority", random_state=SEED, k_neighbors=5)
    traced_resampled, _ = traced.fit_resample(train_pre, y_train)
    regular_resampled, _ = SMOTE(
        sampling_strategy="minority", random_state=SEED, k_neighbors=5
    ).fit_resample(train_pre, y_train)
    # Ensure tracing has NOT changed SMOTE's synthetic data.
    np.testing.assert_allclose(traced_resampled, regular_resampled, atol=1e-12, rtol=1e-12)
    smote_added = np.asarray(traced_resampled)[n_train:]
    smote_parent_ids = train_ids[traced.synthetic_parent_positions_]

    correct_smote_case = Scenario(
        name="Correct SMOTE", leaky=False,
        sampler_input_ids=train_ids,
        generated_features=smote_added,
        generated_parent_ids=smote_parent_ids,
        test_features=np.asarray(smote_test), test_ids=test_ids,
    )

    # POSITIVE CONTROL A: Leaky full-data ROS, as in the reviewed notebook.
    buggy_ros_model, buggy_ros_audit = make_buggy_ros(
        X_all, y_all, X_train, y_train
    )
    buggy_ros_added = buggy_ros_model[:3].transform(
        X_all.loc[buggy_ros_audit["appended_source_ids"]]
    )
    buggy_ros_test = buggy_ros_model[:3].transform(X_test)
    buggy_ros_case = Scenario(
        name="Buggy ROS", leaky=True,
        sampler_input_ids=buggy_ros_audit["sampler_input_source_ids"],
        generated_features=np.asarray(buggy_ros_added),
        generated_parent_ids=buggy_ros_audit["appended_source_ids"],
        test_features=np.asarray(buggy_ros_test), test_ids=test_ids,
    )

    # POSITIVE CONTROL B: Leaky full-data SMOTE, same fixed Test.
    _, buggy_smote_audit = make_buggy_smote(X_all, y_all, X_train, y_train)
    positions = X_all.index.get_indexer(X_test.index)
    if np.any(positions < 0):
        raise RuntimeError("Some Test sources are absent from the original data")
    buggy_smote_case = Scenario(
        name="Buggy SMOTE", leaky=True,
        sampler_input_ids=buggy_smote_audit["sampler_input_source_ids"],
        generated_features=buggy_smote_audit["synthetic_features"],
        generated_parent_ids=buggy_smote_audit["synthetic_parent_source_ids"],
        test_features=buggy_smote_audit["original_preprocessed_features"][positions],
        test_ids=test_ids,
    )

    # NEGATIVE CONTROL C: Correct ADASYN, fit ONLY on original Train.
    correct_adasyn = make_reference_pipeline("adasyn")
    correct_adasyn.fit(X_train, y_train)
    train_pre_adasyn = correct_adasyn[:-2].transform(X_train)
    adasyn_test_pre = correct_adasyn[:-2].transform(X_test)
    adasyn_sampler = ADASYN(
        sampling_strategy="minority", random_state=SEED, n_neighbors=5
    )
    adasyn_resampled, _ = adasyn_sampler.fit_resample(train_pre_adasyn, y_train)
    adasyn_added = np.asarray(adasyn_resampled)[n_train:]
    adasyn_trace = trace_adasyn_generation(
        train_pre_adasyn, y_train, adasyn_sampler, adasyn_added
    )
    correct_adasyn_case = Scenario(
        name="Correct ADASYN", leaky=False,
        sampler_input_ids=train_ids,
        generated_features=adasyn_added,
        generated_parent_ids=train_ids[adasyn_trace["synthetic_parent_positions"]],
        test_features=np.asarray(adasyn_test_pre), test_ids=test_ids,
        difficulty_neighbor_ids=train_ids[adasyn_trace["difficulty_neighbor_positions"]],
    )

    # POSITIVE CONTROL C: Real ADASYN sees ALL original records, fixed Test.
    _, buggy_adasyn_audit = make_buggy_adasyn(
        X_all, y_all, X_train, y_train
    )
    buggy_adasyn_test_positions = X_all.index.get_indexer(X_test.index)
    if np.any(buggy_adasyn_test_positions < 0):
        raise RuntimeError("Test source IDs missing from original dataset")
    buggy_adasyn_case = Scenario(
        name="Buggy ADASYN", leaky=True,
        sampler_input_ids=buggy_adasyn_audit["sampler_input_source_ids"],
        generated_features=buggy_adasyn_audit["synthetic_features"],
        generated_parent_ids=buggy_adasyn_audit["synthetic_parent_source_ids"],
        test_features=buggy_adasyn_audit["original_preprocessed_features"][buggy_adasyn_test_positions],
        test_ids=test_ids,
        difficulty_neighbor_ids=buggy_adasyn_audit["difficulty_neighbor_source_ids"],
    )

    return [
        correct_ros_case, correct_smote_case, correct_adasyn_case,
        buggy_ros_case, buggy_smote_case, buggy_adasyn_case,
    ]


def source_overlap_flags(ids, test_ids):
    """Convert a source-isolation violation to a flag."""
    try:
        assert_no_shared_sources(np.asarray(ids).ravel(), test_ids)
    except AssertionError as error:
        if "Source leakage detected" not in str(error):
            raise
        return True
    return False


def run_oracles(case):
    """Same three oracles, applied uniformly to EVERY scenario."""
    input_ids = np.asarray(case.sampler_input_ids)
    test_ids = np.asarray(case.test_ids)
    generated = np.asarray(case.generated_features)
    test_features = np.asarray(case.test_features)
    parents = np.asarray(case.generated_parent_ids)
    for ids in (input_ids, test_ids):
        if (ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or len(np.unique(ids)) != len(ids)):
            raise ValueError("Sampler and Test traces need unique original integer IDs")
    if generated.ndim != 2 or test_features.ndim != 2:
        raise ValueError("Recorded features must be two-dimensional")
    if len(test_features) != len(test_ids):
        raise ValueError("Test features and Test IDs must have matching lengths")
    if parents.ndim == 1:
        parents = parents[:, None]
    if (parents.ndim != 2 or parents.shape[1] == 0
            or parents.shape[0] != len(generated)):
        raise ValueError("Parent traces must contain parents for every added row")
    if parents.size and (parents.dtype.kind not in "iu" or np.any(parents < 0)):
        raise ValueError("Parent IDs must be nonnegative integers")
    if not np.isin(parents, input_ids).all():
        raise ValueError("A parent ID is missing from the recorded sampler input")

    input_hits = int(np.intersect1d(case.sampler_input_ids, case.test_ids).size)
    generated_with_test_parent = int(
        np.isin(parents, test_ids).any(axis=1).sum()
    )
    exact_copies = exact_test_duplicate_count(
        case.generated_features, case.test_features
    )
    detections = {
        "Source isolation": source_overlap_flags(case.sampler_input_ids, case.test_ids),
        "Exact duplicate": exact_copies > 0,
        "Generated provenance": source_overlap_flags(parents.ravel(), test_ids),
    }
    return detections, {
        "scenario": case.name,
        "sampler_input_test_ids": input_hits,
        "added_rows": len(case.generated_features),
        "generated_rows_with_test_parent": generated_with_test_parent,
        "exact_test_copies": exact_copies,
        # Missing for ROS/SMOTE means NOT APPLICABLE, not zero.
        "adasyn_distinct_test_ids_in_difficulty_neighbors": (
            int(np.intersect1d(case.difficulty_neighbor_ids, case.test_ids).size)
            if case.difficulty_neighbor_ids is not None else None
        ),
    }


def report_status(detected, leaky):
    if leaky:
        return "DETECTED" if detected else "MISSED"
    return "FALSE POSITIVE" if detected else "PASS"


def main():
    scenarios = make_scenarios()
    matrix_rows = []
    evidence_rows = []
    long_rows = []
    oracle_names = ["Source isolation", "Exact duplicate", "Generated provenance"]

    for case in scenarios:
        detections, evidence = run_oracles(case)
        evidence_rows.append(evidence)
        matrix_row = {"Scenario": case.name, "Ground truth": "LEAKY" if case.leaky else "CLEAN"}
        for oracle_name in oracle_names:
            flagged = detections[oracle_name]
            matrix_row[oracle_name] = report_status(flagged, case.leaky)
            long_rows.append({
                "Scenario": case.name, "Oracle": oracle_name,
                "Leaky": case.leaky, "Detected": flagged,
            })
        matrix_rows.append(matrix_row)

    matrix = pd.DataFrame(matrix_rows).set_index("Scenario")
    evidence = pd.DataFrame(evidence_rows).set_index("scenario")
    long_data = pd.DataFrame(long_rows)
    metric_rows = []
    for oracle_name in oracle_names:
        subset = long_data.loc[long_data["Oracle"] == oracle_name]
        leaky, flagged = subset["Leaky"], subset["Detected"]
        tp = int((leaky & flagged).sum())
        fn = int((leaky & ~flagged).sum())
        fp = int((~leaky & flagged).sum())
        tn = int((~leaky & ~flagged).sum())
        metric_rows.append({
            "Oracle": oracle_name, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "Detection rate": tp / (tp + fn),
            "False-positive rate": fp / (fp + tn),
        })
    metrics = pd.DataFrame(metric_rows).set_index("Oracle")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(OUTPUT_DIR / "detection_matrix.csv")
    metrics.to_csv(OUTPUT_DIR / "oracle_metrics.csv")
    evidence.to_csv(OUTPUT_DIR / "leakage_evidence.csv")

    print("\n=== LEAKAGE DETECTION MATRIX ===")
    print(matrix.to_string())
    print("\n=== ORACLE EFFECTIVENESS (3 leaky + 3 clean cases) ===")
    print(metrics.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n=== OBSERVED EVIDENCE ===")
    print(evidence.to_string())
    print(f"\nCSV reports saved in: {OUTPUT_DIR}")

    # Guard against silently presenting different conditions as the pilot.
    expected = {
        "Correct ROS": ["PASS", "PASS", "PASS"],
        "Correct SMOTE": ["PASS", "PASS", "PASS"],
        "Correct ADASYN": ["PASS", "PASS", "PASS"],
        "Buggy ROS": ["DETECTED", "DETECTED", "DETECTED"],
        "Buggy SMOTE": ["DETECTED", "MISSED", "DETECTED"],
        "Buggy ADASYN": ["DETECTED", "MISSED", "DETECTED"],
    }
    for name, expected_row in expected.items():
        actual = matrix.loc[name, oracle_names].tolist()
        if actual != expected_row:
            raise AssertionError(f"Pilot results changed for {name}: {actual}")
    return matrix, metrics, evidence


if __name__ == "__main__":
    main()
