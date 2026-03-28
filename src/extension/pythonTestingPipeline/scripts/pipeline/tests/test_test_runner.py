"""Unit tests for the pipeline test runner module."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.test_runner import (
    _discover_missing_dependencies,
    run_tests,
    validate_generated_test_file,
)


def _make_test_layout(tmp_path: Path) -> tuple[Path, Path]:
    codebase = tmp_path / "sample_app"
    tests_dir = codebase / "tests"
    tests_dir.mkdir(parents=True)
    test_file = tests_dir / "test_generated_sample.py"
    test_file.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    return codebase, test_file


def test_run_tests_uses_path_valid_for_codebase_cwd_and_fresh_coverage(tmp_path):
    codebase, test_file = _make_test_layout(tmp_path)
    stale_coverage = codebase / "coverage.json"
    stale_coverage.write_text("stale", encoding="utf-8")

    captured = {}
    fresh_coverage = codebase / "coverage.123456789.json"

    def fake_run(cmd, capture_output, text, timeout, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        fresh_coverage.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="1 passed in 0.10s", stderr="")

    with patch("pipeline.test_runner.subprocess.run", side_effect=fake_run), patch(
        "pipeline.test_runner.parse_coverage_json",
        return_value={"percentage": 88.8, "uncovered_areas_text": "missing", "detailed_reports": {"sample.py": {}}},
    ) as mock_parse, patch("pipeline.test_runner.time.time", return_value=100.0), patch(
        "pipeline.test_runner.time.time_ns", return_value=123456789
    ):
        result = run_tests(test_file, codebase)

    assert captured["cwd"] == codebase.resolve()
    assert captured["cmd"][3] == str(Path("tests") / test_file.name)
    assert f"--cov-report=json:{fresh_coverage}" in captured["cmd"]
    mock_parse.assert_called_once_with(fresh_coverage, codebase.resolve())
    assert result["total_tests"] == 1
    assert result["passed"] == 1
    assert result["coverage_percentage"] == 88.8
    assert stale_coverage.read_text(encoding="utf-8") == "{}"


def test_run_tests_fails_fast_when_pytest_reports_missing_target(tmp_path):
    codebase, test_file = _make_test_layout(tmp_path)
    stale_coverage = codebase / "coverage.json"
    stale_coverage.write_text("stale", encoding="utf-8")

    missing_output = "ERROR: file or directory not found: tests/test_generated_sample.py"

    with patch(
        "pipeline.test_runner.subprocess.run",
        return_value=subprocess.CompletedProcess([], 4, stdout="", stderr=missing_output),
    ), patch("pipeline.test_runner.parse_coverage_json") as mock_parse, patch(
        "pipeline.test_runner.time.time", return_value=100.0
    ), patch("pipeline.test_runner.time.time_ns", return_value=123456789):
        result = run_tests(test_file, codebase)

    mock_parse.assert_not_called()
    assert result["exit_code"] == 4
    assert result["coverage_percentage"] == 0.0
    assert result["total_tests"] == 0
    assert "file or directory not found" in result["output"].lower()
    assert stale_coverage.read_text(encoding="utf-8") == "stale"


def test_run_tests_ignores_stale_run_specific_coverage_file(tmp_path):
    codebase, test_file = _make_test_layout(tmp_path)
    stale_coverage = codebase / "coverage.json"
    stale_coverage.write_text("stale", encoding="utf-8")
    fresh_coverage = codebase / "coverage.123456789.json"

    def fake_run(cmd, capture_output, text, timeout, cwd):
        fresh_coverage.write_text("{}", encoding="utf-8")
        os.utime(fresh_coverage, (50.0, 50.0))
        return subprocess.CompletedProcess(cmd, 0, stdout="1 passed in 0.10s", stderr="")

    with patch("pipeline.test_runner.subprocess.run", side_effect=fake_run), patch(
        "pipeline.test_runner.parse_coverage_json"
    ) as mock_parse, patch("pipeline.test_runner.time.time", return_value=100.0), patch(
        "pipeline.test_runner.time.time_ns", return_value=123456789
    ):
        result = run_tests(test_file, codebase)

    mock_parse.assert_not_called()
    assert result["coverage_percentage"] == 0.0
    assert result["coverage_details"] == {}
    assert stale_coverage.read_text(encoding="utf-8") == "stale"


def test_validate_generated_test_file_runs_py_compile_then_collect_only(tmp_path):
    codebase, test_file = _make_test_layout(tmp_path)
    commands = []

    def fake_run(cmd, capture_output, text, timeout, cwd):
        commands.append((cmd, cwd))
        if cmd[2] == "py_compile":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            cmd, 0, stdout="1 test collected in 0.01s", stderr=""
        )

    with patch("pipeline.test_runner.subprocess.run", side_effect=fake_run):
        result = validate_generated_test_file(test_file, codebase)

    assert result["passed"] is True
    assert result["collected_tests"] == 1
    assert commands[0][0][2] == "py_compile"
    assert commands[1][0][3] == str(Path("tests") / test_file.name)


def test_validate_generated_test_file_fails_on_collect_only_error(tmp_path):
    codebase, test_file = _make_test_layout(tmp_path)

    def fake_run(cmd, capture_output, text, timeout, cwd):
        if cmd[2] == "py_compile":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            cmd,
            2,
            stdout="",
            stderr="ImportError while importing test module",
        )

    with patch("pipeline.test_runner.subprocess.run", side_effect=fake_run):
        result = validate_generated_test_file(test_file, codebase)

    assert result["passed"] is False
    assert result["stage"] == "collect_only"
    assert "collect-only failed" in result["message"]


def test_discover_missing_dependencies_uses_import_probes_for_generated_suite(tmp_path):
    codebase = tmp_path / "sample_app"
    codebase.mkdir()
    (codebase / "build.py").write_text(
        "import PyInstaller.__main__\n", encoding="utf-8"
    )

    def fake_probe(module_name, codebase_path, cwd, timeout=30):
        if module_name == "build":
            return (
                False,
                "ModuleNotFoundError: No module named 'PyInstaller'",
                "PyInstaller",
            )
        return True, "", None

    with patch("pipeline.test_runner._probe_module_import", side_effect=fake_probe):
        missing, diagnostics = _discover_missing_dependencies(
            ["pytest"],
            codebase,
            test_code="import pytest\nimport build\n",
            project_root=codebase,
        )

    assert "PyInstaller" in missing
    assert any("build" in diagnostic for diagnostic in diagnostics)


def test_validate_generated_test_file_rejects_risky_top_level_imports(tmp_path):
    codebase = tmp_path / "sample_app"
    tests_dir = codebase / "tests"
    tests_dir.mkdir(parents=True)
    (codebase / "build.py").write_text(
        "import PyInstaller.__main__\n", encoding="utf-8"
    )
    test_file = tests_dir / "test_generated_sample.py"
    test_file.write_text(
        "import build\n\n"
        "def test_placeholder():\n"
        "    assert True\n",
        encoding="utf-8",
    )

    def fake_run(cmd, capture_output, text, timeout, cwd):
        if cmd[2] == "py_compile":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError("collect-only should not run for risky top-level imports")

    with patch("pipeline.test_runner.subprocess.run", side_effect=fake_run), patch(
        "pipeline.test_runner._probe_module_import",
        return_value=(
            False,
            "ModuleNotFoundError: No module named 'PyInstaller'",
            "PyInstaller",
        ),
    ):
        result = validate_generated_test_file(test_file, codebase)

    assert result["passed"] is False
    assert result["stage"] == "top_level_imports"
    assert "PyInstaller" in result["output"]
