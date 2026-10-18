"""Small checks used by the controlled leakage experiments.

Source checks need trustworthy original-row identities. Exact duplicate
checks compare values only and cannot establish that two rows share a source.
"""

import numpy as np


def _source_ids(values):
    ids = np.asarray(values)
    if ids.ndim != 1:
        raise ValueError("Source IDs must be a one-dimensional array")
    if ids.size and (ids.dtype.kind not in "iu" or np.any(ids < 0)):
        raise ValueError("Source IDs must be nonnegative integers")
    return ids


def assert_no_shared_sources(train_or_sampler_source_ids, test_source_ids):
    """Reject Test IDs in training or sampler input.

    Repeated IDs are allowed: ROS copies and generated-row parents can repeat.
    An empty input is allowed, for example when no generated rows are retained.
    """
    sources = _source_ids(train_or_sampler_source_ids)
    test_ids = _source_ids(test_source_ids)
    overlap = np.intersect1d(sources, test_ids)
    if len(overlap):
        raise AssertionError(
            f"Source leakage detected: {len(overlap)} distinct Test source IDs "
            "were used in training or sampling."
        )


def exact_test_duplicate_count(generated_rows, test_rows):
    """Count added rows exactly equal to any original Test feature row.

    Inputs must be finite real numeric 2D arrays with the same feature count,
    transformed with the same fitted preprocessor. Missing or infinite values
    raise ValueError; they are not silently treated as nonmatches.

    Comparison uses numeric values, without tolerance. Equal int/float values
    and signed zeros match. Each generated row is counted at most once, even
    if multiple Test rows match it. Matching values do not prove common origin.
    """
    generated = np.asarray(generated_rows)
    test = np.asarray(test_rows)
    if (generated.ndim != 2 or test.ndim != 2
            or generated.shape[1] != test.shape[1] or generated.shape[1] == 0):
        raise ValueError("Features must be 2D arrays with matching, nonzero column counts")
    for rows in (generated, test):
        if rows.dtype.kind not in "iuf" or not np.isfinite(rows).all():
            raise ValueError("Features must contain only finite real numeric values")

    # Python numeric values compare by value across int/float dtypes. Row bytes
    # would incorrectly distinguish signed zero and equal mixed-dtype rows.
    test_values = {tuple(row) for row in test.tolist()}
    return sum(tuple(row) in test_values for row in generated.tolist())
