"""Optional AST-based checker for a few oversampling scope patterns.

Supported: straight-line Python within one function, ROS/SMOTE/ADASYN
constructors, variable aliases and simple transformations, and calls to
train_test_split/fit_resample. No user code is executed. This is a heuristic,
not a proof of data leakage or its absence. It is separate from the three
runtime checks used in the pilot study.

fit_resample inputs may be positional or use X= and y=. train_test_split
accepts data arrays positionally; named options such as stratify= do not add
data inputs. Dynamic **kwargs, branches, loops and helper calls are not traced.

Two rules:
    S01_RESAMPLE_BEFORE_SPLIT: the results of fit_resample are then split.
    S02_FULL_DATA_RESAMPLING: fit_resample gets a variable marked as full/test
        data by a preceding split or its name (possibly transformed). This
        may be a false positive if the function is intentionally training
        the final model on all available data.

Usage:
    python static_leakage.py pipelines.py
    python static_leakage.py pipelines.py --csv results/static_findings.csv
    python static_leakage.py pipelines.py --fail-on-finding  # CI mode
"""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from pathlib import Path

SAMPLERS = frozenset({"RandomOverSampler", "SMOTE", "ADASYN"})
SPLITTER = "train_test_split"


@dataclass(frozen=True)
class StaticFinding:
    file: str
    function: str
    line: int
    rule: str
    sampler: str
    message: str


def _simple_name(expr: ast.expr) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return ""


def _parts(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for elt in target.elts for name in _parts(elt)]
    return []


def _sampler_inputs(call: ast.Call) -> list[ast.expr]:
    inputs = list(call.args[:2])
    for position, name in enumerate(("X", "y")):
        if position >= len(call.args):
            inputs.extend(keyword.value for keyword in call.keywords if keyword.arg == name)
    return inputs


def _name_hint(name: str) -> frozenset[str]:
    # Explicit naming convention. This is not a whole-program proof.
    low = name.lower()
    if low in ("x_all", "y_all", "x_full", "y_full", "full_data", "all_data"):
        return frozenset({"full"})
    if low.startswith(("x_all_", "y_all_", "x_full_", "y_full_")):
        return frozenset({"full"})
    if low.startswith(("x_train", "y_train", "train_")):
        return frozenset({"train"})
    if low.startswith(("x_test", "y_test", "test_", "holdout_")):
        return frozenset({"test"})
    return frozenset()


class _ScopeScanner:
    def __init__(self, name: str, path: str, constructors: dict[str, str],
                 split_names: set[str]):
        self.name, self.path = name, path
        self.constructors, self.split_names = constructors, split_names
        self.tags: dict[str, frozenset[str]] = {}
        self.instances: dict[str, str] = {}
        self.resample_events: dict[str, tuple[int, str]] = {}
        self.findings: list[StaticFinding] = []
        self._seen: set[tuple[int, str]] = set()

    def _tag(self, expression: ast.AST) -> frozenset[str]:
        if isinstance(expression, ast.Name):
            return self.tags.get(expression.id, _name_hint(expression.id))
        if isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
            out = frozenset()
            for x in expression.elts:
                out = out | self._tag(x)
            return out
        if isinstance(expression, ast.Subscript):
            return self._tag(expression.value)
        if isinstance(expression, ast.Attribute):
            return self._tag(expression.value)
        if isinstance(expression, ast.Call):
            # Propagate lineage through straightforward transformations such as
            # prep.transform(X_all), np.asarray(X_all), and X_all.copy().
            if isinstance(expression.func, ast.Attribute):
                method = expression.func.attr
                if method in {"copy", "to_numpy", "values", "reset_index"}:
                    return self._tag(expression.func.value)
                if method in {"transform", "fit_transform", "asarray", "array",
                              "drop", "dropna", "astype", "reshape"}:
                    if expression.args:
                        return self._tag(expression.args[0])
                    return self._tag(expression.func.value)
            if expression.args:
                # Unknown wrapper functions are NOT followed: report nothing.
                return frozenset()
        return frozenset()

    def _event_for(self, expression: ast.AST) -> tuple[int, str] | None:
        if isinstance(expression, ast.Name):
            return self.resample_events.get(expression.id)
        if isinstance(expression, ast.Call):
            # Limited propagation through known linear, row-preserving wrappers.
            if isinstance(expression.func, ast.Attribute):
                if expression.func.attr in {"transform", "asarray", "array", "copy", "to_numpy"}:
                    target = expression.args[0] if expression.args else expression.func.value
                    return self._event_for(target)
        if isinstance(expression, ast.Subscript):
            return self._event_for(expression.value)
        return None

    def _constructor(self, expression: ast.AST) -> str | None:
        if not isinstance(expression, ast.Call):
            return None
        func_name = _simple_name(expression.func)
        return self.constructors.get(func_name)

    def _sampler_call(self, expression: ast.AST) -> str | None:
        if (not isinstance(expression, ast.Call)
                or not isinstance(expression.func, ast.Attribute)
                or expression.func.attr != "fit_resample"):
            return None
        owner = expression.func.value
        if isinstance(owner, ast.Name):
            return self.instances.get(owner.id)
        # Also accept SMOTE().fit_resample(...) without a named sampler.
        if isinstance(owner, ast.Call):
            return self._constructor(owner)
        return None

    def _split_call(self, expression: ast.AST) -> bool:
        return (isinstance(expression, ast.Call)
                and _simple_name(expression.func) in self.split_names)

    def _emit(self, line: int, rule: str, sampler: str, message: str):
        if (line, rule) not in self._seen:
            self.findings.append(StaticFinding(self.path, self.name, line,
                                                rule, sampler, message))
            self._seen.add((line, rule))

    def _visit_assignment(self, targets: list[ast.expr], value: ast.expr):
        names = [n for target in targets for n in _parts(target)]
        if not names:
            return
        sampler_kind = self._constructor(value)
        if sampler_kind:
            for name in names:
                self.instances[name] = sampler_kind
            return
        if isinstance(value, ast.Name) and value.id in self.instances:
            for name in names:
                self.instances[name] = self.instances[value.id]
            return

        if self._split_call(value):
            # Dataset supplied to a split is full relative to THAT split.
            for arg in value.args[:2]:
                if isinstance(arg, ast.Name):
                    prev = self.tags.get(arg.id, _name_hint(arg.id))
                    if "train" not in prev and "test" not in prev:
                        self.tags[arg.id] = prev | {"full"}
            in_tags = frozenset().union(*(self._tag(x) for x in value.args[:2]))
            if "resampled" in in_tags:
                for arg in value.args[:2]:
                    event = self._event_for(arg)
                    if event:
                        self._emit(event[0], "S01_RESAMPLE_BEFORE_SPLIT", event[1],
                                   f"Resampled values are fed to train_test_split at line {value.lineno}.")
            # Python unpacking convention: X_train, X_test, y_train, y_test.
            for idx, name in enumerate(names):
                self.tags[name] = frozenset({"train" if idx % 2 == 0 else "test"})
            return

        detected = self._sampler_call(value)
        if detected:
            self._on_sampler(value, detected)
            for name in names:
                self.tags[name] = frozenset({"resampled"})
                self.resample_events[name] = (value.lineno, detected)
            return

        derived = self._tag(value)
        old_event = self._event_for(value)
        for name in names:
            self.tags[name] = derived
            if old_event:
                self.resample_events[name] = old_event
            else:
                self.resample_events.pop(name, None)

    def _on_sampler(self, value: ast.Call, sampler_name: str):
        source_tags = frozenset().union(*(self._tag(x) for x in _sampler_inputs(value)))
        if "full" in source_tags or "test" in source_tags:
            self._emit(
                value.lineno, "S02_FULL_DATA_RESAMPLING", sampler_name,
                "fit_resample receives full/test-labeled input (directly or via "
                "a tracked transformation); verify whether held-out rows exist."
            )

    def scan(self, statements: list[ast.stmt]) -> list[StaticFinding]:
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                self._visit_assignment(stmt.targets, stmt.value)
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                self._visit_assignment([stmt.target], stmt.value)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                sampler_name = self._sampler_call(stmt.value)
                if sampler_name:
                    self._on_sampler(stmt.value, sampler_name)
            # Deliberately do not infer through branches, loops, callbacks,
            # or imported helper functions; avoid claiming a clean bill of health.
        return self.findings


def scan_source(source: str, filename: str = "<source>") -> list[StaticFinding]:
    """Parse Python source; return suspicious calls without executing code."""
    tree = ast.parse(source, filename=filename)
    constructors = {name: name for name in SAMPLERS}
    split_names = {SPLITTER}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in SAMPLERS and node.module and "over_sampling" in node.module:
                    constructors[alias.asname or alias.name] = alias.name
                if alias.name == SPLITTER and node.module == "sklearn.model_selection":
                    split_names.add(alias.asname or alias.name)

    # Recognize transparent, locally declared SMOTE subclasses such as
    # 'class TracedSMOTE(SMOTE): ...' without executing their class bodies.
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_kind = constructors.get(_simple_name(base))
                if base_kind:
                    constructors[node.name] = base_kind
                    break

    scopes = [("<module>", [s for s in tree.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))])]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append((node.name, node.body))
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scopes.append((f"{node.name}.{method.name}", method.body))

    findings = []
    for name, statements in scopes:
        findings.extend(_ScopeScanner(name, filename, constructors, split_names).scan(statements))
    return sorted(findings, key=lambda f: (f.line, f.rule))


def scan_file(path: str | Path) -> list[StaticFinding]:
    path = Path(path)
    return scan_source(path.read_text(encoding="utf-8"), str(path))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Python files to inspect (not execute)")
    parser.add_argument("--csv", help="Optional CSV report path")
    parser.add_argument("--fail-on-finding", action="store_true", help="Exit 1 on potential leaks")
    args = parser.parse_args(argv)
    try:
        results = [finding for path in args.files for finding in scan_file(path)]
    except (OSError, SyntaxError) as exc:
        parser.error(str(exc))
    if results:
        for f in results:
            print(f"{f.file}:{f.line}: {f.rule} [{f.sampler}] ({f.function})")
            print(f"    {f.message}")
    else:
        print("No supported leakage pattern detected (NOT proof of safety).")
    print(f"Total potential findings: {len(results)}")
    if args.csv:
        target = Path(args.csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle)
            w.writerow(["file", "function", "line", "rule", "sampler", "message"])
            for f in results:
                w.writerow([f.file, f.function, f.line, f.rule, f.sampler, f.message])
        print(f"Wrote: {target}")
    return int(bool(results) and args.fail_on_finding)


if __name__ == "__main__":
    raise SystemExit(main())
