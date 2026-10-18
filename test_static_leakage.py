"""Unit tests for our small AST-only static oversampling-leakage oracle.

These tests do NOT run ROS/SMOTE/ADASYN; the snippets are parsed as text.
Test outcomes on deliberately buggy programs validate that a warning is emitted.
"""
from pathlib import Path

import pytest

from static_leakage import scan_file, scan_source

SAMPLERS = ("RandomOverSampler", "SMOTE", "ADASYN")


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_s01_detects_resampling_before_splitting(sampler):
    code = f'''
from imblearn.over_sampling import {sampler}
from sklearn.model_selection import train_test_split
sampler = {sampler}(random_state=42)
X_bal, y_bal = sampler.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal)
'''
    findings = scan_source(code)
    assert len(findings) == 1
    assert findings[0].rule == "S01_RESAMPLE_BEFORE_SPLIT"
    assert findings[0].sampler == sampler


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_s02_detects_full_data_resampling_after_existing_split(sampler):
    code = f'''
from imblearn.over_sampling import {sampler}
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
prep = make_preprocessor()
X_pre = prep.transform(X)
sampler = {sampler}(random_state=42)
X_new, y_new = sampler.fit_resample(X_pre, y)
'''
    findings = scan_source(code)
    assert len(findings) == 1
    assert findings[0].rule == "S02_FULL_DATA_RESAMPLING"
    assert findings[0].sampler == sampler


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_correct_split_then_train_only_has_no_findings(sampler):
    code = f'''
from imblearn.over_sampling import {sampler}
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
sampler = {sampler}(random_state=42)
X_bal, y_bal = sampler.fit_resample(X_train, y_train)
model.fit(X_bal, y_bal)
'''
    assert scan_source(code) == []


def test_real_current_project_flags_three_buggy_functions():
    findings = scan_file(Path(__file__).with_name("pipelines.py"))
    by_function = {finding.function: finding for finding in findings}
    assert set(by_function) == {
        "make_buggy_ros", "make_buggy_smote", "make_buggy_adasyn"
    }
    assert {finding.rule for finding in findings} == {"S02_FULL_DATA_RESAMPLING"}
    assert {finding.sampler for finding in findings} == set(SAMPLERS)
    assert all(finding.line > 0 for finding in findings)
    assert not any(finding.function == "make_reference_pipeline" for finding in findings)


def test_aliases_and_transforms_are_followed():
    code = '''
from imblearn.over_sampling import SMOTE as Oversampler
from sklearn.model_selection import train_test_split as tts
xtr, xte, ytr, yte = tts(X, y)
scanner = Oversampler()
sampling_input = preprocessor.transform(X)
x_new, y_new = scanner.fit_resample(sampling_input, y)
'''
    findings = scan_source(code)
    assert len(findings) == 1
    assert findings[0].rule == "S02_FULL_DATA_RESAMPLING"


def test_comments_and_docstrings_do_not_create_finding():
    code = '''
"""sampler.fit_resample(X_all, y_all)"""
# sampler.fit_resample(X_all, y_all)
def example():
    sentence = 'fit_resample(X_all, y_all)'
    return sentence
'''
    assert scan_source(code) == []


def test_parser_does_not_execute_untrusted_program():
    # Executing this Python source would raise. AST parsing must not.
    code = '''
raise RuntimeError("MUST NOT RUN")
from imblearn.over_sampling import ADASYN
sampler = ADASYN()
X_res, y_res = sampler.fit_resample(X_all, y_all)
'''
    findings = scan_source(code)
    assert len(findings) == 1
    assert findings[0].rule == "S02_FULL_DATA_RESAMPLING"


def test_separate_internal_training_resplit_can_trigger_cautious_warning():
    # Demonstrate a limitation: not every flagged resplit is test leakage.
    code = '''
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
sampler = SMOTE()
X_res, y_res = sampler.fit_resample(X_train, y_train)
X_subtrain, X_val, y_subtrain, y_val = train_test_split(X_res, y_res)
'''
    findings = scan_source(code)
    assert {f.rule for f in findings} == {"S01_RESAMPLE_BEFORE_SPLIT"}


def test_direct_inline_sampler_constructor_is_detected():
    code = """
from imblearn.over_sampling import SMOTE
X_bal, y_bal = SMOTE().fit_resample(X_all, y_all)
"""
    findings = scan_source(code)
    assert len(findings) == 1
    assert findings[0].rule == "S02_FULL_DATA_RESAMPLING"
    assert findings[0].sampler == "SMOTE"


@pytest.mark.parametrize("sampler", SAMPLERS)
@pytest.mark.parametrize("arguments", ("X=X, y=y", "X, y=y"))
def test_keyword_sampler_inputs_detect_resampling_before_split(sampler, arguments):
    code = f'''
from imblearn.over_sampling import {sampler}
from sklearn.model_selection import train_test_split
sampler = {sampler}()
X_bal, y_bal = sampler.fit_resample({arguments})
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)
'''
    findings = scan_source(code)
    assert len(findings) == 1
    assert findings[0].rule == "S01_RESAMPLE_BEFORE_SPLIT"
    assert findings[0].sampler == sampler


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_keyword_sampler_inputs_detect_full_data(sampler):
    code = f'''
from imblearn.over_sampling import {sampler}
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
sampler = {sampler}()
X_new, y_new = sampler.fit_resample(y=y, X=X)
'''
    findings = scan_source(code)
    assert len(findings) == 1
    assert findings[0].rule == "S02_FULL_DATA_RESAMPLING"
    assert findings[0].sampler == sampler


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_keyword_train_inputs_with_named_split_options_are_clean(sampler):
    code = f'''
from imblearn.over_sampling import {sampler}
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
sampler = {sampler}()
X_bal, y_bal = sampler.fit_resample(X=X_train, y=y_train)
'''
    assert scan_source(code) == []


def test_split_options_are_not_treated_as_data_arrays():
    code = '''
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
X_bal, y_bal = SMOTE().fit_resample(X=X_train, y=y_train)
X_a, X_b = train_test_split(X_other, test_size=0.2, stratify=y_bal)
'''
    # The checker follows arrays passed to the split, not its stratify option.
    # This does not establish whether every aspect of the example is sound.
    assert scan_source(code) == []


def test_test_labels_in_keyword_y_are_checked():
    code = '''
from imblearn.over_sampling import SMOTE
X_bal, y_bal = SMOTE().fit_resample(X_train, y=y_test)
'''
    findings = scan_source(code)
    assert len(findings) == 1
    assert findings[0].rule == "S02_FULL_DATA_RESAMPLING"
