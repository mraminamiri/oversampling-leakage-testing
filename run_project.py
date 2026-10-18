"""Run the tests and regenerate the pilot's tables and figures."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXPECTED_SHA256 = "b78029447fae2743b3218bb2b76ef0d04afe8d7e55ce2faf4d1ec82d8f8ae8ac"


def run_step(name, arguments):
    print(f"\nRunning {name}...", flush=True)
    completed = subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True)
    log_path = ROOT / "results" / f"{name}_local.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="", flush=True)
    if completed.returncode:
        raise RuntimeError(f"{name} failed; see {log_path.name}")
    return {"exit_code": completed.returncode, "log": log_path.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="Regenerate outputs without pytest")
    parser.add_argument("--static", action="store_true", help="Also run the optional static checker")
    arguments = parser.parse_args()
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    # A failed rerun must not leave a previous success record behind.
    (output_dir / "run_metadata.json").unlink(missing_ok=True)
    dataset_hash = hashlib.sha256((ROOT / "diabetes.csv").read_bytes()).hexdigest()
    if dataset_hash != EXPECTED_SHA256:
        raise ValueError("diabetes.csv differs from the CSV used in this pilot")

    steps = {}
    if not arguments.skip_tests:
        steps["pytest"] = run_step("pytest", [sys.executable, "-m", "pytest", "-q"])
    for name in ["detection_matrix", "evaluate_models", "plot_results"]:
        steps[name] = run_step(name, [sys.executable, f"{name}.py"])
    if arguments.static:
        steps["static_analysis"] = run_step("static_analysis", [
            sys.executable, "static_leakage.py", "pipelines.py",
            "--csv", str(output_dir / "static_findings.csv"),
        ])

    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in [
            "numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
            "pytest", "matplotlib",
        ]},
        "dataset_sha256": dataset_hash,
        "seed": 42,
        "original_train_records": 614,
        "original_test_records": 154,
        "tests_skipped": arguments.skip_tests,
        "steps": steps,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\nDone. Tables are in results/ and figures are in figures/.")


if __name__ == "__main__":
    main()
