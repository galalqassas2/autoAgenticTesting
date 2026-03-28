"""Test execution and dependency management for the Python Testing Pipeline."""

import ast
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llm_config import create_llm_client
from pipeline.prompts import (
    DEPENDENCY_ANALYSIS_SYSTEM_PROMPT,
    DEPENDENCY_FIX_SYSTEM_PROMPT,
)

__all__ = [
    "extract_dependencies",
    "install_dependencies",
    "run_tests",
    "parse_pytest_output",
    "parse_coverage_json",
    "validate_generated_test_file",
]

_STDLIB_MODULES = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
_PACKAGE_TO_IMPORT_CANDIDATES = {
    "beautifulsoup4": ["bs4"],
    "opencv-python": ["cv2"],
    "pillow": ["PIL"],
    "pyinstaller": ["PyInstaller"],
    "pytest-asyncio": ["pytest_asyncio"],
    "pytest-cov": ["pytest_cov"],
    "pytest-timeout": ["pytest_timeout"],
    "python-dotenv": ["dotenv"],
    "pyyaml": ["yaml"],
    "scikit-learn": ["sklearn"],
}
_IMPORT_TO_PACKAGE = {
    "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
    "dotenv": "python-dotenv",
    "pil": "Pillow",
    "pyinstaller": "PyInstaller",
    "pytest_asyncio": "pytest-asyncio",
    "pytest_cov": "pytest-cov",
    "pytest_timeout": "pytest-timeout",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}


def _build_pytest_target(test_file: Path, run_cwd: Path) -> str:
    """Return a pytest target path that is valid from the chosen working dir."""
    test_file = test_file.resolve()
    run_cwd = run_cwd.resolve()

    try:
        return str(test_file.relative_to(run_cwd))
    except ValueError:
        return str(test_file)


def _build_coverage_json_path(codebase_path: Path) -> Path:
    """Create a unique coverage output path for a single pytest run."""
    return codebase_path / f"coverage.{time.time_ns()}.json"


def _strip_version_spec(package: str) -> str:
    """Normalize a requirement string down to its package name."""
    return (
        package.split("==")[0]
        .split(">=")[0]
        .split("<=")[0]
        .split(">")[0]
        .split("<")[0]
        .strip()
    )


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    """Return values with duplicates removed while keeping the original order."""
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _build_codebase_modules(codebase_path: Path) -> set[str]:
    """Collect importable root module names from the codebase."""
    modules = set()
    if not codebase_path.exists():
        return modules
    for py_file in codebase_path.rglob("*.py"):
        parts = {part.lower() for part in py_file.parts}
        if "__pycache__" in parts or "tests" in parts:
            continue
        modules.add(py_file.stem)
    return modules


def _module_path_for_name(module_name: str, codebase_path: Path) -> Optional[Path]:
    """Resolve a simple module name to a file inside the codebase."""
    direct = codebase_path / f"{module_name}.py"
    if direct.exists():
        return direct
    package_init = codebase_path / module_name / "__init__.py"
    if package_init.exists():
        return package_init
    return None


def _package_to_import_candidates(package: str) -> List[str]:
    """Map a package name to the most likely import roots."""
    stripped = _strip_version_spec(package)
    normalized = stripped.lower().replace("_", "-")
    candidates = list(_PACKAGE_TO_IMPORT_CANDIDATES.get(normalized, ()))
    candidates.extend([stripped, normalized.replace("-", "_")])
    return _dedupe_preserve_order(candidates)


def _package_for_import(module_name: str) -> str:
    """Map an import root back to a likely installable package name."""
    normalized = module_name.lower().replace("_", "-")
    return _IMPORT_TO_PACKAGE.get(normalized, module_name)


def _extract_import_roots_from_code(test_code: str) -> List[str]:
    """Extract imported root modules from generated test code."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return []

    modules: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module.split(".")[0])
    return _dedupe_preserve_order(modules)


def _probe_module_import(
    module_name: str,
    codebase_path: Path,
    cwd: Path,
    timeout: int = 30,
) -> Tuple[bool, str, Optional[str]]:
    """Attempt to import a module in an isolated subprocess."""
    probe = f"""
import importlib
import sys
import traceback

sys.path.insert(0, {json.dumps(str(codebase_path))})

try:
    importlib.import_module({json.dumps(module_name)})
except Exception:
    traceback.print_exc()
    raise
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    missing_module = None
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", output)
    if match:
        missing_module = match.group(1).split(".")[0]
    return result.returncode == 0, output, missing_module


def _is_main_guard(node: ast.If) -> bool:
    """Check whether an if-statement is `if __name__ == "__main__":`."""
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _node_has_import_side_effect(node: ast.stmt) -> bool:
    """Detect obvious module-level side effects in an imported source module."""
    if isinstance(node, ast.Expr):
        return not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        )
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        value = getattr(node, "value", None)
        return value is not None and any(
            isinstance(child, ast.Call) for child in ast.walk(value)
        )
    if isinstance(node, ast.If):
        return not _is_main_guard(node)
    if isinstance(
        node,
        (
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Pass,
        ),
    ):
        return False
    return True


def _analyze_top_level_import_risk(module_name: str, codebase_path: Path) -> List[str]:
    """Inspect a codebase module for import-time dependency or side-effect risks."""
    module_path = _module_path_for_name(module_name, codebase_path)
    if module_path is None:
        return []

    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return [
            f"Top-level import of '{module_name}' is unsafe because {module_path.name} does not parse cleanly"
        ]

    codebase_modules = _build_codebase_modules(codebase_path)
    reasons = []
    external_imports = []
    for node in tree.body:
        if _node_has_import_side_effect(node):
            reasons.append(
                f"Top-level import of '{module_name}' is unsafe because {module_path.name} executes module-level calls or state changes on import"
            )
            break

        if isinstance(node, ast.Import):
            external_imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            external_imports.append(node.module.split(".")[0])

    for dependency in _dedupe_preserve_order(external_imports):
        if dependency in _STDLIB_MODULES or dependency in codebase_modules:
            continue
        success, _, missing_module = _probe_module_import(
            dependency, codebase_path, codebase_path
        )
        if not success:
            missing_name = missing_module or dependency
            reasons.append(
                f"Top-level import of '{module_name}' is unsafe because dependency '{missing_name}' is not importable"
            )

    return reasons


def _find_unsafe_top_level_imports(test_file: Path, codebase_path: Path) -> List[str]:
    """Reject generated suites that import risky source modules at file import time."""
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []

    codebase_modules = _build_codebase_modules(codebase_path)
    risky_modules: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            risky_modules.extend(
                alias.name.split(".")[0]
                for alias in node.names
                if alias.name.split(".")[0] in codebase_modules
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in codebase_modules:
                risky_modules.append(root)

    issues = []
    for module_name in _dedupe_preserve_order(risky_modules):
        issues.extend(_analyze_top_level_import_risk(module_name, codebase_path))
    return issues


def _discover_missing_dependencies(
    packages: List[str],
    cwd: Path,
    test_code: str = "",
    project_root: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """Probe actual imports to determine which dependencies are still missing."""
    root = (project_root or cwd).resolve()
    modules_to_probe: List[str] = []
    for package in packages:
        modules_to_probe.extend(_package_to_import_candidates(package))
    if test_code:
        modules_to_probe.extend(_extract_import_roots_from_code(test_code))
    modules_to_probe = _dedupe_preserve_order(modules_to_probe)

    codebase_modules = _build_codebase_modules(root)
    missing_packages: List[str] = []
    diagnostics: List[str] = []

    for module_name in modules_to_probe:
        root_name = module_name.split(".")[0]
        if root_name in _STDLIB_MODULES:
            continue

        success, output, missing_module = _probe_module_import(module_name, root, cwd)
        if success:
            continue

        if root_name in codebase_modules:
            if (
                missing_module
                and missing_module not in _STDLIB_MODULES
                and missing_module not in codebase_modules
            ):
                missing_packages.append(_package_for_import(missing_module))
                diagnostics.append(
                    f"Import probe for '{root_name}' failed because '{missing_module}' is not importable"
                )
            else:
                last_line = output.splitlines()[-1] if output else "unknown import error"
                diagnostics.append(
                    f"Import probe for '{root_name}' failed: {last_line}"
                )
            continue

        missing_packages.append(_package_for_import(root_name))
        if (
            missing_module
            and missing_module not in _STDLIB_MODULES
            and missing_module not in codebase_modules
        ):
            missing_packages.append(_package_for_import(missing_module))
        diagnostics.append(f"Module '{module_name}' is not importable")

    filtered_missing = []
    for package in _dedupe_preserve_order(missing_packages):
        normalized = _strip_version_spec(package).lower().replace("_", "-")
        if normalized in _STDLIB_MODULES:
            continue
        filtered_missing.append(package)
    return filtered_missing, diagnostics


def _parse_collected_test_count(output: str) -> Optional[int]:
    """Extract collected-test count from pytest --collect-only output."""
    patterns = (
        r"(\d+)\s+tests?\s+collected",
        r"collected\s+(\d+)\s+items?",
    )
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return int(match.group(1))
    if "no tests collected" in output.lower():
        return 0
    return None


def validate_generated_test_file(test_file: Path, codebase_path: Path) -> Dict[str, object]:
    """
    Run fast semantic checks before the full pytest+coverage execution.

    This keeps clearly invalid suites out of the main loop and gives the
    implementation agent focused repair feedback.
    """
    codebase_path = codebase_path.resolve()
    test_file = test_file.resolve()
    run_cwd = codebase_path

    if not test_file.exists():
        return {
            "passed": False,
            "stage": "missing_file",
            "message": f"Generated test file does not exist: {test_file}",
            "output": "",
            "collected_tests": 0,
        }

    try:
        compile_result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(test_file)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=run_cwd,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "stage": "py_compile",
            "message": "py_compile timed out while validating generated tests",
            "output": "",
            "collected_tests": 0,
        }

    compile_output = compile_result.stdout + "\n" + compile_result.stderr
    if compile_result.returncode != 0:
        return {
            "passed": False,
            "stage": "py_compile",
            "message": "py_compile failed for generated tests",
            "output": compile_output.strip(),
            "collected_tests": 0,
        }

    unsafe_top_level_imports = _find_unsafe_top_level_imports(test_file, codebase_path)
    if unsafe_top_level_imports:
        return {
            "passed": False,
            "stage": "top_level_imports",
            "message": "Generated tests import risky source modules at file import time",
            "output": "\n".join(unsafe_top_level_imports),
            "collected_tests": 0,
        }

    pytest_target = _build_pytest_target(test_file, run_cwd)
    try:
        collect_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                pytest_target,
                "--collect-only",
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=run_cwd,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "stage": "collect_only",
            "message": "pytest --collect-only timed out while validating generated tests",
            "output": "",
            "collected_tests": 0,
        }

    collect_output = (collect_result.stdout + "\n" + collect_result.stderr).strip()
    collected_tests = _parse_collected_test_count(collect_output)

    if collect_result.returncode != 0:
        return {
            "passed": False,
            "stage": "collect_only",
            "message": "pytest --collect-only failed for generated tests",
            "output": collect_output,
            "collected_tests": collected_tests or 0,
        }

    if collected_tests == 0:
        return {
            "passed": False,
            "stage": "collect_only",
            "message": "pytest collected 0 tests from the generated suite",
            "output": collect_output,
            "collected_tests": 0,
        }

    return {
        "passed": True,
        "stage": "collect_only",
        "message": f"Semantic validation passed; collected {collected_tests or 'unknown'} test(s)",
        "output": collect_output,
        "collected_tests": collected_tests,
    }


def analyze_dependencies_with_llm(test_code: str) -> Optional[List[str]]:
    """
    Uses LLM to analyze test code and determine exact PyPI packages.
    Returns None if analysis fails, triggering fallback to regex.
    """
    print("   ðŸ¤– Asking LLM to identify dependencies...")
    try:
        llm_client = create_llm_client(use_mock_on_failure=True)

        # Truncate test code if it's too long to avoid token limits
        # 10k chars is usually enough to see imports and usage
        code_sample = test_code[:10000]
        if len(test_code) > 10000:
            code_sample += "\n... (truncated)"

        response, _ = llm_client.call(
            DEPENDENCY_ANALYSIS_SYSTEM_PROMPT,
            f"Identify PyPI packages for this code:\n\n{code_sample}",
        )

        # Parse JSON response
        if "```" in response:
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if json_match:
                response = json_match.group(1).strip()

        data = json.loads(response)
        packages = data.get("packages", [])

        # Basic validation
        if isinstance(packages, list) and all(isinstance(p, str) for p in packages):
            return packages

    except Exception as e:
        print(f"   âš ï¸  LLM dependency analysis failed: {e}")

    return None


def extract_dependencies(test_code: str) -> List[str]:
    """Extracts required packages using LLM with regex fallback."""

    # 1. Try LLM first
    llm_packages = analyze_dependencies_with_llm(test_code)
    if llm_packages is not None:
        print(f"   âœ¨ LLM identified packages: {', '.join(llm_packages)}")
        return llm_packages

    # 2. Fallback to regex if LLM fails
    print("   âš ï¸  Falling back to regex dependency extraction...")

    # Common import to package name mappings
    import_to_package = {
        "fastapi": "fastapi",
        "starlette": "starlette",
        "httpx": "httpx",
        "pytest": "pytest",
        "pytest_asyncio": "pytest-asyncio",
        "jinja2": "jinja2",
        "pydantic": "pydantic",
        "sqlalchemy": "sqlalchemy",
        "flask": "flask",
        "django": "django",
        "requests": "requests",
        "aiohttp": "aiohttp",
        "numpy": "numpy",
        "pandas": "pandas",
        "cv2": "opencv-python",
        "bs4": "beautifulsoup4",
        "yaml": "PyYAML",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
    }

    # Find all imports
    import_pattern = r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    imports = set()
    for line in test_code.split("\n"):
        match = re.match(import_pattern, line.strip())
        if match:
            module = match.group(1)
            if module in import_to_package:
                imports.add(import_to_package[module])

    # Always include pytest essentials
    imports.add("pytest")
    imports.add("pytest-cov")
    imports.add("pytest-timeout")

    return list(imports)


def install_dependencies_with_retry(
    packages: List[str],
    cwd: Path,
    test_code: str = "",
    project_root: Optional[Path] = None,
) -> Tuple[str, int]:
    """Install only dependencies that are still not importable."""
    attempt = 0
    max_retries = 3
    current_packages = _dedupe_preserve_order(packages)
    last_output = ""
    last_return_code = 0

    while attempt <= max_retries:
        if not current_packages:
            return "No packages to install", 0

        missing_packages, diagnostics = _discover_missing_dependencies(
            current_packages,
            cwd,
            test_code=test_code,
            project_root=project_root,
        )
        if not missing_packages:
            if diagnostics:
                return "\n".join(diagnostics), 1
            print(
                f"\nAll dependencies already importable: {', '.join(current_packages)}"
            )
            return "All dependencies already importable", 0

        print(
            f"\nInstalling dependencies (Attempt {attempt + 1}/{max_retries + 1}): {', '.join(missing_packages)}"
        )

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", *missing_packages],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired:
            return "Dependency installation timed out", 1
        except Exception as e:
            return f"Error installing dependencies: {e}", 1

        last_output = (result.stdout + "\n" + result.stderr).strip()
        last_return_code = result.returncode

        if result.returncode == 0:
            remaining_packages, remaining_diagnostics = _discover_missing_dependencies(
                current_packages,
                cwd,
                test_code=test_code,
                project_root=project_root,
            )
            if not remaining_packages:
                print("   Dependencies installed successfully")
                return last_output, 0
            current_packages = remaining_packages
            last_output = "\n".join(
                part
                for part in [last_output, "\n".join(remaining_diagnostics)]
                if part
            )
            last_return_code = 1
            print("   Installation completed, but some imports are still unavailable")
            attempt += 1
            continue

        print(f"   Installation failed: {result.stderr.strip()}")
        if attempt >= max_retries:
            break

        llm_client = create_llm_client(use_mock_on_failure=True)
        user_prompt = f"""Dependency installation failed.

Packages attempted: {missing_packages}

Error message:
{result.stderr}

Suggest a fix."""

        response, _ = llm_client.call(DEPENDENCY_FIX_SYSTEM_PROMPT, user_prompt)
        if "```" in response:
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if json_match:
                response = json_match.group(1).strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            break

        new_packages = _dedupe_preserve_order(data.get("packages", []))
        if not new_packages:
            break

        current_packages = new_packages
        attempt += 1

    return last_output, last_return_code


def install_dependencies(
    packages: List[str],
    cwd: Path,
    test_code: str = "",
    project_root: Optional[Path] = None,
) -> Tuple[str, int]:
    """Wrapper for install_dependencies_with_retry."""
    return install_dependencies_with_retry(
        packages,
        cwd,
        test_code=test_code,
        project_root=project_root,
    )


def parse_pytest_output(output: str) -> Dict[str, int]:
    """Parses pytest output to extract test counts."""
    total = passed = failed = 0

    # Match patterns like "5 passed", "3 failed", "10 passed, 2 failed"
    passed_match = re.search(r"(\d+) passed", output)
    failed_match = re.search(r"(\d+) failed", output)
    error_match = re.search(r"(\d+) error", output)

    if passed_match:
        passed = int(passed_match.group(1))
    if failed_match:
        failed = int(failed_match.group(1))
    if error_match:
        failed += int(error_match.group(1))

    total = passed + failed
    return {"total": total, "passed": passed, "failed": failed}


def parse_coverage_json(coverage_json_path: Path, source_root: Path) -> dict:
    """
    Parses coverage.json to extract detailed coverage information.

    Returns a dict with:
      - percentage: overall coverage percentage
      - uncovered_areas_text: formatted string for LLM
      - detailed_reports: dict of FileCoverageReport (as dicts)
    """
    try:
        from dataclasses import asdict

        from pipeline.coverage import (
            analyze_coverage,
            format_uncovered_areas,
            get_overall_percentage,
        )

        if coverage_json_path.exists():
            reports = analyze_coverage(coverage_json_path, source_root)
            return {
                "percentage": get_overall_percentage(reports),
                "uncovered_areas_text": format_uncovered_areas(reports),
                "detailed_reports": {
                    fp: asdict(r) for fp, r in reports.items()
                },
            }
    except Exception as e:
        print(f"   âš ï¸ Could not parse coverage.json: {e}")
    return {"percentage": 0.0, "uncovered_areas_text": "", "detailed_reports": {}}


def run_tests(
    test_file: Path,
    codebase_path: Path,
    run_mutation_tests: bool = False,
) -> dict:
    """Runs the generated PyTest suite with coverage measurement.

    Args:
        test_file: Path to the pytest test file.
        codebase_path: Root path of the source code being tested.
        run_mutation_tests: If True, run mutation testing after the
            pytest suite completes.

    Returns:
        Dict containing test results, coverage data, and optionally
        mutation testing results.
    """
    print("\nRunning tests with coverage...")

    codebase_path = codebase_path.resolve()
    test_file = test_file.resolve()
    run_cwd = codebase_path

    if not test_file.exists():
        return {
            "output": f"Test file does not exist: {test_file}",
            "exit_code": 1,
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "coverage_percentage": 0.0,
            "uncovered_areas_text": "",
            "coverage_details": {},
            "mutation_score": 0.0,
            "mutation_report": None,
            "mutation_feedback": "",
        }

    # Get source directory to measure coverage
    source_dir = str(codebase_path)
    pytest_target = _build_pytest_target(test_file, run_cwd)
    coverage_json_path = _build_coverage_json_path(codebase_path)
    canonical_coverage_path = codebase_path / "coverage.json"

    # Create a .coveragerc file to exclude test files from coverage measurement.
    # This prevents the AI from trying to generate tests for test files.
    coveragerc_path = codebase_path / ".coveragerc"
    coveragerc_content = """[run]
omit =
    */tests/*
    */test/*
    **/test_*.py
    **/*_test.py
    **/conftest.py

[report]
omit =
    */tests/*
    */test/*
    **/test_*.py
    **/*_test.py
    **/conftest.py
"""
    try:
        with open(coveragerc_path, "w", encoding="utf-8") as f:
            f.write(coveragerc_content)
    except Exception as e:
        print(f"   âš ï¸ Could not create .coveragerc: {e}")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        pytest_target,
        "-v",
        "--tb=short",
        "--timeout=30",  # Per-test timeout of 30 seconds
        f"--cov={source_dir}",
        "--cov-branch",
        "--cov-report=term-missing",
        f"--cov-report=json:{coverage_json_path}",
    ]

    try:
        run_started_at = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2-minute overall timeout
            cwd=run_cwd,
        )
        output = result.stdout + "\n" + result.stderr

        if "file or directory not found" in output.lower():
            return {
                "output": output,
                "exit_code": result.returncode or 1,
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "coverage_percentage": 0.0,
                "uncovered_areas_text": "",
                "coverage_details": {},
                "mutation_score": 0.0,
                "mutation_report": None,
                "mutation_feedback": "",
            }

        # Parse test results from output
        test_results = parse_pytest_output(output)

        # Parse coverage from this run only. Ignore stale coverage artifacts.
        coverage_data = {"percentage": 0.0, "uncovered_areas_text": "", "detailed_reports": {}}
        if coverage_json_path.exists() and coverage_json_path.stat().st_mtime >= run_started_at:
            coverage_data = parse_coverage_json(coverage_json_path, codebase_path)
            try:
                shutil.copyfile(coverage_json_path, canonical_coverage_path)
            except Exception as exc:
                print(f"   âš ï¸ Could not refresh canonical coverage.json: {exc}")
        else:
            print("   âš ï¸ Ignoring stale or missing coverage report from this pytest run")

        # Run mutation testing if enabled
        mutation_score = 0.0
        mutation_report = None
        mutation_feedback = ""

        if run_mutation_tests:
            try:
                from pipeline.mutation_testing import (
                    format_mutation_feedback,
                    run_mutation_testing,
                )

                mutation_report = run_mutation_testing(
                    codebase_path=codebase_path,
                    test_file=test_file,
                    min_file_coverage=95.0,
                    timeout=600,
                )
                mutation_score = mutation_report.mutation_score
                mutation_feedback = format_mutation_feedback(mutation_report)
            except Exception as exc:
                print(f"   Mutation testing failed: {exc}")

        return {
            "output": output,
            "exit_code": result.returncode,
            "total_tests": test_results["total"],
            "passed": test_results["passed"],
            "failed": test_results["failed"],
            "coverage_percentage": coverage_data["percentage"],
            "uncovered_areas_text": coverage_data["uncovered_areas_text"],
            "coverage_details": coverage_data["detailed_reports"],
            "mutation_score": mutation_score,
            "mutation_report": mutation_report,
            "mutation_feedback": mutation_feedback,
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "Test execution timed out",
            "exit_code": 1,
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "coverage_percentage": 0.0,
            "uncovered_areas_text": "",
            "coverage_details": {},
            "mutation_score": 0.0,
            "mutation_report": None,
            "mutation_feedback": "",
        }
    except Exception as e:
        return {
            "output": f"Error running tests: {e}",
            "exit_code": 1,
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "coverage_percentage": 0.0,
            "uncovered_areas_text": "",
            "coverage_details": {},
            "mutation_score": 0.0,
            "mutation_report": None,
            "mutation_feedback": "",
        }

