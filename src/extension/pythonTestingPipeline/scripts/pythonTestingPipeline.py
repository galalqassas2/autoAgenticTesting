#!/usr/bin/env python3
"""
Python Automated Testing Pipeline

Usage:
    python pythonTestingPipeline.py <codebase_path> [--auto-approve] \\
        [--no-run-tests] [--output-dir <dir>] [--model <model>]

Note:
    Generated tests are run by default unless --no-run-tests is supplied.
    The --coverage flag is accepted for compatibility, but coverage is already
    collected automatically when tests are executed.

Example:
    python pythonTestingPipeline.py ./my_project --auto-approve
"""

import argparse
import json
import re
import sys
import time as time_module
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# Import LLM configuration and client
from llm_config import create_llm_client
from pipeline.agents import (
    EvaluationAgent,
    IdentificationAgent,
    ImplementationAgent,
)
from pipeline.governance import FailureReason, governance_log

# ==================== Type Definitions ====================
from pipeline.models import (
    TestEvaluationOutput,
    TestScenariosOutput,
)

# ==================== System Prompts ====================
from pipeline.test_runner import (
    extract_dependencies,
    install_dependencies,
    run_tests,
    validate_generated_test_file,
)

# ==================== Pipeline Implementation ====================


class PythonTestingPipeline:
    """Orchestrates the three-agent testing pipeline."""

    def __init__(self, model: Optional[str] = None):
        # Use LLM client from llm_config for auto API key rotation and model fallback
        self.llm_client = create_llm_client(use_mock_on_failure=False)
        self.prompt_history = []  # Track all prompts for later analysis

        # Initialize agents
        self.identification_agent = IdentificationAgent(
            self.llm_client, self.prompt_history
        )
        self.implementation_agent = ImplementationAgent(
            self.llm_client, self.prompt_history
        )
        self.evaluation_agent = EvaluationAgent(self.llm_client, self.prompt_history)

        # Print current configuration
        if self.llm_client.current_api_key:
            print(f"Using model: {self.llm_client.current_model}")

    @property
    def model(self) -> str:
        """Get current model from LLM client."""
        return self.llm_client.current_model

    def save_prompts(self, output_dir: Path, run_id: str = None) -> Path:
        """Saves all prompts from this run to a JSON file."""
        run_id = run_id or time_module.strftime("%Y%m%d_%H%M%S")

        prompts_file = output_dir / f"prompts_{run_id}.json"

        prompt_data = {
            "run_id": run_id,
            "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S"),
            "model": getattr(self.llm_client, "last_used_model", None) or self.model,
            "total_prompts": len(self.prompt_history),
            "prompts": self.prompt_history,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(prompts_file, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, indent=2, default=str)

        print(f"   Prompts saved: {prompts_file}")
        return prompts_file

    def generate_report(
        self, results: dict, output_dir: Path, run_id: str = None
    ) -> Path:
        """Generates a concise markdown report with LLM summary."""
        run_id = run_id or time_module.strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"report_{run_id}.md"

        # Prepare data for LLM summary
        eval_data = results.get("evaluation", {})
        summary_data = eval_data.get("execution_summary", {})
        coverage = eval_data.get("code_coverage_percentage", 0)
        security = eval_data.get("security_issues", [])
        scenarios = results.get("approved_scenarios", {}).get("test_scenarios", [])

        # Generate LLM summary
        summary_prompt = f"""Summarize this test run in 2-3 sentences:
- Tests: {summary_data.get("passed", 0)}/{summary_data.get("total_tests", 0)} passed
- Coverage: {coverage:.1f}%
- Security issues: {len(security)}
- Scenarios tested: {len(scenarios)}
Be concise and professional."""

        try:
            llm_summary, _ = self.llm_client.call(
                sys_p="You are a concise technical writer. Summarize test results.",
                usr_p=summary_prompt,
            )
        except Exception:
            llm_summary = f"Test run completed with {coverage:.1f}% coverage."

        # Build markdown report
        status_icon = "✅" if results.get("status") == "completed" else "❌"
        report = f"""# Test Pipeline Report

**Status:** {status_icon} {results.get("status", "unknown").upper()}
**Generated:** {time_module.strftime("%Y-%m-%d %H:%M:%S")}
**Model:** {getattr(self.llm_client, "last_used_model", None) or self.model}

## Summary

{llm_summary}

## Results

| Metric | Value |
|--------|-------|
| Total Tests | {summary_data.get("total_tests", 0)} |
| Passed | {summary_data.get("passed", 0)} |
| Failed | {summary_data.get("failed", 0)} |
| Coverage | {coverage:.1f}% |
| Security Issues | {len(security)} |
"""

        # Add security section if issues exist
        if security:
            report += "\n## Security Issues\n\n"
            for issue in security[:5]:  # Limit to top 5
                sev = (
                    issue.get("severity", "unknown")
                    if isinstance(issue, dict)
                    else getattr(issue, "severity", "unknown")
                )
                # Use 'issue' field (model) or fallback to 'description'
                desc = (
                    issue.get("issue", issue.get("description", ""))
                    if isinstance(issue, dict)
                    else getattr(issue, "issue", getattr(issue, "description", ""))
                )
                loc = (
                    issue.get("location", "")
                    if isinstance(issue, dict)
                    else getattr(issue, "location", "")
                )
                report += f"- **[{sev.upper()}]** {desc}"
                if loc:
                    report += f" (`{loc}`)"
                report += "\n"

        # Add timing section
        timing = results.get("timing", {})
        if timing:
            total_secs = timing.get("total_seconds", 0)
            iter_times = timing.get("iteration_times", [])
            report += "\n## Timing\n\n"
            report += "| Phase | Duration |\n|-------|----------|\n"
            report += f"| Total Pipeline | {total_secs:.1f}s |\n"
            if iter_times:
                for i, t in enumerate(iter_times, 1):
                    report += f"| Iteration {i} | {t:.1f}s |\n"

        # Add test file location
        if results.get("test_file"):
            report += f"\n## Output\n\n- Test file: `{results['test_file']}`\n"

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"   Report saved: {report_file}")
        return report_file

    def identify_test_scenarios(self, codebase_path: Path) -> TestScenariosOutput:
        """Agent 1: Identifies test scenarios from codebase using parallel processing."""
        return self.identification_agent.run(codebase_path)

    def interpret_user_input(
        self, user_input: str, scenarios: TestScenariosOutput
    ) -> dict:
        """Uses LLM to interpret user input and determine action."""
        scenario_list = "\n".join(
            f"{i + 1}. [{s.priority}] {s.scenario_description}"
            for i, s in enumerate(scenarios.test_scenarios)
        )
        prompt = f"""Scenarios:\n{scenario_list}\n\nUser said: "{user_input}"

Determine intent. Return JSON:
{{"action": "approve"|"remove"|"refine", "indices": [1,2,...] if removing, "feedback": "..."}}"""
        try:
            response, _ = self.llm_client.call(
                sys_p="Interpret intent for test scenario approval. Return valid JSON.",
                usr_p=prompt,
            )
            match = re.search(r"\{[^{}]*\}", response)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {
            "action": "refine",
            "feedback": user_input,
        }  # Default: treat as feedback

    def refine_scenarios(
        self, scenarios: TestScenariosOutput, feedback: str
    ) -> TestScenariosOutput:
        """Uses LLM to refine scenarios based on feedback."""
        current = "\n".join(
            f"- [{s.priority}] {s.scenario_description}"
            for s in scenarios.test_scenarios
        )
        prompt = f'Scenarios:\n{current}\n\nFeedback: {feedback}\n\nReturn refined JSON: {{"test_scenarios": [{{"priority": "High/Medium/Low", "scenario_description": "..."}}]}}'
        try:
            response, _ = self.llm_client.call(
                sys_p="Refine test scenarios. Return valid JSON only.", usr_p=prompt
            )
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                data = json.loads(match.group())
                from pipeline.models import TestScenario

                new = [
                    TestScenario(
                        priority=s.get("priority", "Medium"),
                        scenario_description=s["scenario_description"],
                    )
                    for s in data.get("test_scenarios", [])
                    if "scenario_description" in s
                ]
                if new:
                    print(f"   Refined to {len(new)} scenarios")
                    return TestScenariosOutput(test_scenarios=new)
        except Exception as e:
            print(f"   Could not refine: {e}")
        return scenarios

    def request_approval(self, scenarios: TestScenariosOutput) -> TestScenariosOutput:
        """Natural language approval flow using LLM interpretation."""

        def display(scens):
            print("\n" + "=" * 60)
            print("TEST SCENARIOS FOR REVIEW")
            print("=" * 60)
            for i, s in enumerate(scens.test_scenarios, 1):
                icon = {"High": "!", "Medium": "~", "Low": "."}.get(s.priority, " ")
                print(f"  {i}. [{icon}] {s.scenario_description}")
            print(
                f"\n  Total: {len(scens.test_scenarios)} | Type anything to proceed/feedback"
            )
            print("-" * 60)

        display(scenarios)
        while True:
            user_input = input("\n>>> ").strip()
            if not user_input:
                continue
            intent = self.interpret_user_input(user_input, scenarios)
            action = intent.get("action", "refine")

            if action == "approve":
                print("Approved!")
                return scenarios
            elif action == "remove":
                indices = {
                    int(i) - 1 for i in intent.get("indices", []) if isinstance(i, int)
                }
                scenarios.test_scenarios = [
                    s
                    for i, s in enumerate(scenarios.test_scenarios)
                    if i not in indices
                ]
                print(f"   Removed {len(indices)} scenario(s)")
                display(scenarios)
            else:  # refine
                scenarios = self.refine_scenarios(
                    scenarios, intent.get("feedback", user_input)
                )
                display(scenarios)

    def generate_test_code(
        self, scenarios: TestScenariosOutput, codebase_path: Path, output_dir: Path
    ) -> tuple[str, Path]:
        """Agent 2: Generates PyTest test code from approved scenarios."""
        return self.implementation_agent.run(scenarios, codebase_path, output_dir)

    def _extract_uncovered_areas(self, test_output: str) -> str:
        """Extracts uncovered lines/areas from pytest-cov output."""
        uncovered_lines = []

        # Try to extract from term-missing format: "file.py   10   2   80%   5-6"
        missing_pattern = r"^(.+\.py)\s+\d+\s+\d+\s+\d+%\s+(.+)$"
        for line in test_output.split("\n"):
            match = re.match(missing_pattern, line.strip())
            if match and match.group(2).strip():
                uncovered_lines.append(f"{match.group(1)}: lines {match.group(2)}")

        # Also check coverage.json for detailed missing lines
        try:
            coverage_json = Path("coverage.json")
            if coverage_json.exists():
                with open(coverage_json, "r") as f:
                    data = json.load(f)
                    for file_path, file_data in data.get("files", {}).items():
                        missing = file_data.get("missing_lines", [])
                        if missing:
                            uncovered_lines.append(f"{file_path}: lines {missing[:20]}")
        except Exception:
            pass

        return (
            "\n".join(uncovered_lines)
            if uncovered_lines
            else "No specific uncovered areas identified"
        )

    def evaluate_results(
        self, test_results: dict, scenarios: TestScenariosOutput, codebase_path: Path
    ) -> TestEvaluationOutput:
        """Agent 3: Evaluates test results, coverage, and security."""
        return self.evaluation_agent.run(test_results, scenarios, codebase_path)

    def generate_additional_tests(
        self,
        codebase_path: Path,
        existing_test_file: Path,
        coverage_percentage: float,
        uncovered_areas: str,
        syntax_errors: str = "",
        validation_errors: str = "",
        security_issues: list = None,
    ) -> tuple[str, Path]:
        """Generates additional tests to improve coverage and fix security issues."""
        return self.implementation_agent.improve_tests(
            codebase_path,
            existing_test_file,
            coverage_percentage,
            uncovered_areas,
            syntax_errors,
            validation_errors,
            security_issues,
        )

    @staticmethod
    def _empty_test_results(message: str) -> dict:
        """Return a minimal failed test result payload."""
        return {
            "output": message,
            "exit_code": 1,
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "coverage_percentage": 0.0,
            "uncovered_areas_text": "",
            "coverage_details": {},
            "mutation_results": None,
        }

    @staticmethod
    def _result_signature(
        test_results: dict, evaluation: Optional[TestEvaluationOutput]
    ) -> tuple:
        """Build a stable signature for final replay comparison."""
        return (
            test_results.get("exit_code", 1),
            test_results.get("total_tests", 0),
            test_results.get("passed", 0),
            test_results.get("failed", 0),
            round(evaluation.code_coverage_percentage if evaluation else 0.0, 1),
        )

    def _final_acceptance_replay(
        self,
        test_file: Path,
        codebase_path: Path,
        scenarios: TestScenariosOutput,
        expected_results: dict,
        expected_evaluation: Optional[TestEvaluationOutput],
        iteration: int,
    ) -> tuple[dict, TestEvaluationOutput, bool]:
        """Replay the saved suite and use that replay as the final source of truth."""
        validation = validate_generated_test_file(test_file, codebase_path)
        if not validation["passed"]:
            message = f"{validation['stage']}: {validation['message']}"
            governance_log.log_validation(
                "final_acceptance", str(test_file), False, message
            )
            governance_log.log_failure(
                FailureReason.TEST_FAILURE,
                f"Final acceptance failed semantic validation: {message}",
                iteration,
            )
            failed_results = self._empty_test_results(
                message
                + (
                    f"\n\n{validation['output']}"
                    if validation.get("output")
                    else ""
                )
            )
            failed_evaluation = self.evaluate_results(
                failed_results, scenarios, codebase_path
            )
            return failed_results, failed_evaluation, False

        replay_results = run_tests(test_file, codebase_path)
        replay_evaluation = self.evaluate_results(
            replay_results, scenarios, codebase_path
        )

        if not expected_results or expected_evaluation is None:
            governance_log.log_validation(
                "final_acceptance",
                str(test_file),
                True,
                "Final acceptance replay succeeded",
            )
            return replay_results, replay_evaluation, True

        expected_signature = self._result_signature(
            expected_results, expected_evaluation
        )
        replay_signature = self._result_signature(replay_results, replay_evaluation)
        if expected_signature == replay_signature:
            governance_log.log_validation(
                "final_acceptance",
                str(test_file),
                True,
                "Final acceptance replay matched the saved suite metrics",
            )
            return replay_results, replay_evaluation, True

        message = (
            "Final acceptance replay mismatch: "
            f"expected {expected_signature}, got {replay_signature}"
        )
        governance_log.log_validation(
            "final_acceptance", str(test_file), False, message
        )
        governance_log.log_failure(FailureReason.TEST_FAILURE, message, iteration)
        return replay_results, replay_evaluation, False

    def run_pipeline(
        self,
        codebase_path: Path,
        output_dir: Optional[Path] = None,
        should_run_tests: bool = True,
        coverage: bool = False,
        auto_approve: bool = False,
    ) -> dict:
        """
        Runs the complete testing pipeline.

        Coverage is collected automatically whenever test execution is enabled.
        """
        print("=" * 60)
        print("🚀 Python Automated Testing Pipeline")
        print("=" * 60)

        output_dir = output_dir or codebase_path / "tests"
        results = {"status": "started"}

        # Start total pipeline timer
        pipeline_start_time = time_module.time()
        iteration_times = []

        try:
            # Step 1: Identify scenarios
            scenarios = self.identify_test_scenarios(codebase_path)
            results["identified_scenarios"] = asdict(scenarios)

            # Step 2: Get approval
            if auto_approve:
                approved_scenarios = scenarios
            else:
                approved_scenarios = self.request_approval(scenarios)
            results["approved_scenarios"] = asdict(approved_scenarios)

            # Step 3: Generate tests
            test_code, test_file = self.generate_test_code(
                approved_scenarios, codebase_path, output_dir
            )
            results["test_file"] = str(test_file)
            results["test_code"] = test_code

            # Step 4: Install dependencies
            if should_run_tests:
                deps = extract_dependencies(test_code)
                if deps:
                    dep_output, dep_exit = install_dependencies(
                        deps,
                        codebase_path,
                        test_code=test_code,
                        project_root=codebase_path,
                    )
                    results["dependencies_installed"] = deps
                    results["dependency_output"] = dep_output

            # Step 5: Run tests with coverage and improvement loop
            if should_run_tests:
                target_coverage = 90.0
                target_mutation_score = 80.0
                max_iterations = 15  # Safety limit to prevent infinite loops
                current_test_file = test_file
                current_test_code = test_code
                iteration = 0

                # Track progress to prevent getting stuck
                best_coverage = 0.0
                best_test_code = None  # Will store snapshot of best test code
                best_test_results = None
                best_evaluation = None
                best_severe_count = float("inf")
                consecutive_no_progress = 0
                previous_coverage = 0.0  # For mutation testing delta trigger
                current_coverage = 0.0
                test_results = {}
                evaluation = None
                has_severe_security = False

                while iteration < max_iterations:
                    iteration += 1
                    iteration_start = time_module.time()
                    print(f"\n--- Iteration {iteration} ---")

                    validation = validate_generated_test_file(
                        current_test_file, codebase_path
                    )
                    if not validation["passed"]:
                        validation_message = (
                            f"{validation['stage']}: {validation['message']}"
                        )
                        print(
                            f"   âš ï¸  Semantic validation failed: {validation_message}"
                        )
                        governance_log.log_validation(
                            "semantic_validator",
                            str(current_test_file),
                            False,
                            validation_message,
                        )
                        governance_log.log_failure(
                            FailureReason.TEST_FAILURE,
                            validation_message,
                            iteration,
                        )

                        consecutive_no_progress += 1
                        if consecutive_no_progress >= 5:
                            print(
                                f"\nâš ï¸  No progress limits for {consecutive_no_progress} iterations. Stopping."
                            )
                            print(f"   Best coverage: {best_coverage:.1f}%")
                            print(f"   Lowest severe issues: {best_severe_count}")
                            break

                        current_test_code, current_test_file = (
                            self.generate_additional_tests(
                                codebase_path,
                                current_test_file,
                                current_coverage,
                                validation.get("output", "")
                                or "Semantic validation failed before coverage could be measured",
                                validation_errors=(
                                    validation_message
                                    + "\n\n"
                                    + validation.get("output", "")[:2000]
                                ),
                            )
                        )

                        new_deps = extract_dependencies(current_test_code)
                        if new_deps:
                            install_dependencies(
                                new_deps,
                                codebase_path,
                                test_code=current_test_code,
                                project_root=codebase_path,
                            )

                        iteration_time = time_module.time() - iteration_start
                        iteration_times.append(iteration_time)
                        print(f"   â±ï¸  Iteration time: {iteration_time:.1f}s")
                        continue

                    governance_log.log_validation(
                        "semantic_validator",
                        str(current_test_file),
                        True,
                        validation["message"],
                    )

                    # Determine if mutation testing should run this iteration
                    from pipeline.mutation_testing import should_enable_mutation_testing

                    run_mutation = should_enable_mutation_testing(
                        current_coverage=previous_coverage,
                        previous_coverage=best_coverage
                        if iteration > 1
                        else 0.0,
                        iteration=iteration,
                    )

                    # Run tests with coverage (and mutation if enabled)
                    test_results = run_tests(
                        current_test_file,
                        codebase_path,
                        run_mutation_tests=run_mutation,
                    )
                    results["test_output"] = test_results["output"]
                    results["exit_code"] = test_results["exit_code"]

                    # Step 6: Evaluate results (includes security analysis)
                    evaluation = self.evaluate_results(
                        test_results, approved_scenarios, codebase_path
                    )
                    results["evaluation"] = asdict(evaluation)

                    current_coverage = evaluation.code_coverage_percentage
                    has_severe_security = evaluation.has_severe_security_issues
                    security_issues = evaluation.security_issues
                    mutation_score = evaluation.mutation_score

                    # Check completion criteria: coverage >= 90% AND no severe security issues
                    # AND mutation score >= 80% (only enforced when mutation testing ran)
                    coverage_met = current_coverage >= target_coverage
                    security_met = not has_severe_security
                    mutation_met = (
                        mutation_score >= target_mutation_score
                        or not run_mutation
                    )

                    if coverage_met and security_met and mutation_met:
                        print("\nAll targets met!")
                        print(
                            f"   Coverage: {current_coverage:.1f}% (>={target_coverage}%)"
                        )
                        if run_mutation:
                            print(
                                f"   Mutation Score: {mutation_score:.1f}% "
                                f"(>={target_mutation_score}%)"
                            )
                        print("   Severe security issues: None")
                        if security_issues:
                            low_med = [
                                si
                                for si in security_issues
                                if si.severity in ("low", "medium")
                            ]
                            if low_med:
                                print(
                                    f"   Minor security issues (low/medium): {len(low_med)}"
                                )
                        iteration_time = time_module.time() - iteration_start
                        iteration_times.append(iteration_time)
                        print(f"   Iteration time: {iteration_time:.1f}s")
                        break

                    # Check for progress
                    current_severe_count = sum(
                        1
                        for si in security_issues
                        if si.severity in ("critical", "high")
                    )

                    progress_made = False
                    if current_coverage > best_coverage:
                        best_coverage = current_coverage
                        # Snapshot current test code (strings are immutable, so safe)
                        best_test_code = current_test_code
                        best_test_results = test_results
                        best_evaluation = evaluation
                        progress_made = True

                    if current_severe_count < best_severe_count:
                        best_severe_count = current_severe_count
                        progress_made = True

                    # Update previous coverage for delta calculation
                    previous_coverage = current_coverage

                    if progress_made:
                        consecutive_no_progress = 0
                    else:
                        consecutive_no_progress += 1

                    if consecutive_no_progress >= 5:
                        print(
                            f"\n⚠️  No progress limits for {consecutive_no_progress} iterations. Stopping."
                        )
                        print(f"   Best coverage: {best_coverage:.1f}%")
                        print(f"   Lowest severe issues: {best_severe_count}")
                        break

                    # Log current status
                    status_parts = []
                    if not coverage_met:
                        status_parts.append(
                            f"coverage {current_coverage:.1f}% < {target_coverage}%"
                        )
                        governance_log.log_failure(
                            FailureReason.COVERAGE_LOW,
                            f"Coverage {current_coverage:.1f}% < {target_coverage}%",
                            iteration,
                        )
                    if not security_met:
                        severe_count = sum(
                            1
                            for si in security_issues
                            if si.severity in ("critical", "high")
                        )
                        status_parts.append(f"{severe_count} severe security issue(s)")
                        governance_log.log_failure(
                            FailureReason.SECURITY_ISSUE,
                            f"{severe_count} severe security issue(s)",
                            iteration,
                        )
                    print(f"   ⚠️  Needs improvement: {', '.join(status_parts)}")

                    # Extract uncovered areas from test output
                    uncovered_areas = test_results.get(
                        "uncovered_areas_text", ""
                    ) or self._extract_uncovered_areas(test_results["output"])

                    # Check for syntax errors in output
                    syntax_errors = ""
                    if "SyntaxError" in test_results["output"]:
                        syntax_match = re.search(
                            r"(SyntaxError:.*?)(?:\n\n|\Z)",
                            test_results["output"],
                            re.DOTALL,
                        )
                        if syntax_match:
                            syntax_errors = syntax_match.group(1)
                        else:
                            syntax_errors = "SyntaxError detected in test file"
                        governance_log.log_failure(
                            FailureReason.SYNTAX_ERROR, syntax_errors, iteration
                        )

                    # Generate additional tests with coverage and security feedback
                    current_test_code, current_test_file = (
                        self.generate_additional_tests(
                            codebase_path,
                            current_test_file,
                            current_coverage,
                            uncovered_areas,
                            syntax_errors=syntax_errors,
                            validation_errors="",
                            security_issues=security_issues
                            if has_severe_security
                            else None,
                        )
                    )

                    # Re-extract and install any new dependencies
                    new_deps = extract_dependencies(current_test_code)
                    if new_deps:
                        install_dependencies(
                            new_deps,
                            codebase_path,
                            test_code=current_test_code,
                            project_root=codebase_path,
                        )

                    # Record iteration time
                    iteration_time = time_module.time() - iteration_start
                    iteration_times.append(iteration_time)
                    print(f"   ⏱️  Iteration time: {iteration_time:.1f}s")

                else:
                    # Reached max iterations without meeting targets
                    print(f"\n⚠️ Max iterations ({max_iterations}) reached")
                    print(f"   Final coverage: {current_coverage:.1f}%")
                    if has_severe_security:
                        print("   ⚠️  Unresolved severe security issues remain")
                    recommendations = (
                        evaluation.actionable_recommendations if evaluation else []
                    )
                    if recommendations:
                        print("   Recommendations:")
                        for rec in recommendations[:5]:
                            print(f"   • {rec}")

                # Restore best test code if final iteration is worse
                if best_test_code and best_coverage > current_coverage:
                    print(
                        f"\n🔄 Restoring test code with best coverage: {best_coverage:.1f}%"
                    )
                    with open(current_test_file, "w", encoding="utf-8") as f:
                        f.write(best_test_code)
                    current_test_code = best_test_code
                    final_validation = validate_generated_test_file(
                        current_test_file, codebase_path
                    )
                    if final_validation["passed"]:
                        print(
                            "   Re-running restored best suite so report and coverage match the saved file"
                        )
                        governance_log.log_validation(
                            "semantic_validator",
                            str(current_test_file),
                            True,
                            final_validation["message"],
                        )
                        test_results = run_tests(current_test_file, codebase_path)
                        results["test_output"] = test_results["output"]
                        results["exit_code"] = test_results["exit_code"]
                        evaluation = self.evaluate_results(
                            test_results, approved_scenarios, codebase_path
                        )
                        results["evaluation"] = asdict(evaluation)
                        current_coverage = evaluation.code_coverage_percentage
                    else:
                        message = (
                            f"{final_validation['stage']}: {final_validation['message']}"
                        )
                        governance_log.log_validation(
                            "semantic_validator",
                            str(current_test_file),
                            False,
                            message,
                        )
                        governance_log.log_failure(
                            FailureReason.TEST_FAILURE,
                            f"Restored best suite failed semantic validation: {message}",
                            iteration,
                        )
                        if best_test_results is not None:
                            test_results = best_test_results
                            results["test_output"] = test_results["output"]
                            results["exit_code"] = test_results["exit_code"]
                        if best_evaluation is not None:
                            evaluation = best_evaluation
                            results["evaluation"] = asdict(evaluation)
                            current_coverage = evaluation.code_coverage_percentage

                results["test_file"] = str(current_test_file)
                results["test_code"] = current_test_code
                test_results, evaluation, acceptance_passed = (
                    self._final_acceptance_replay(
                        current_test_file,
                        codebase_path,
                        approved_scenarios,
                        test_results,
                        evaluation,
                        iteration,
                    )
                )
                results["test_output"] = test_results["output"]
                results["exit_code"] = test_results["exit_code"]
                results["evaluation"] = asdict(evaluation)
                if not acceptance_passed:
                    results["status"] = "failed"

            # Calculate total time
            total_time = time_module.time() - pipeline_start_time
            results["timing"] = {
                "total_seconds": round(total_time, 2),
                "iteration_times": [round(t, 2) for t in iteration_times],
                "iterations_count": len(iteration_times),
            }
            if results.get("status") != "failed":
                results["status"] = "completed"

            # Save all prompts to JSON for later analysis
            run_id = str(int(time_module.time()))
            prompts_file = self.save_prompts(output_dir, run_id)
            results["prompts_file"] = str(prompts_file)
            results["total_prompts"] = len(self.prompt_history)

            if "coverage_details" in (test_results or {}):
                coverage_report_path = output_dir / f"coverage_report_{run_id}.json"
                try:
                    with open(coverage_report_path, "w", encoding="utf-8") as f:
                        import json
                        json.dump(test_results["coverage_details"], f, indent=2)
                    results["coverage_report_file"] = str(coverage_report_path)
                    print(f"Coverage Report saved: {coverage_report_path}")
                except Exception as e:
                    print(f"   ⚠️ Could not save coverage report: {e}")

            # Export governance audit trail (transparency, accountability)
            governance_file = governance_log.export_audit_trail(
                output_dir / f"governance_{run_id}.json"
            )
            results["governance_file"] = str(governance_file)

            # Generate markdown report with LLM summary
            report_file = self.generate_report(results, output_dir, run_id)
            results["report_file"] = str(report_file)

            # Print summary
            print("\n" + "=" * 60)
            print("✅ Pipeline Complete!")
            print("=" * 60)
            print(f"   Test file: {results.get('test_file', test_file)}")
            print(f"   Scenarios: {len(approved_scenarios.test_scenarios)}")

            if "evaluation" in results:
                eval_data = results["evaluation"]
                summary = eval_data["execution_summary"]
                print(f"   Tests: {summary['passed']}/{summary['total_tests']} passed")
                print(f"   Coverage: {eval_data['code_coverage_percentage']:.1f}%")

            print(f"   Prompts used: {len(self.prompt_history)}")
            print(
                f"   Total time: {total_time:.1f}s ({len(iteration_times)} iterations)"
            )
            print(f"   Report: {report_file}")

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            print(f"\n❌ Pipeline failed: {e}")

            # Still save prompts even on failure
            try:
                prompts_file = self.save_prompts(
                    output_dir or codebase_path / "tests", str(int(time_module.time()))
                )
                print(f"   Prompts saved to: {prompts_file}")
            except Exception as save_error:
                print(f"   Could not save prompts: {save_error}")

        return results


# ==================== Main Entry Point ====================


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Python Automated Testing Pipeline")
    parser.add_argument("codebase_path", type=Path, help="Path to Python codebase")
    # The pipeline will run generated tests by default. Use --no-run-tests to disable.
    parser.add_argument(
        "--run-tests",
        dest="run_tests",
        action="store_true",
        help="Run generated tests (deprecated; tests run by default)",
    )
    parser.add_argument(
        "--no-run-tests",
        dest="no_run_tests",
        action="store_true",
        help="Do not run generated tests",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help=(
            "Compatibility flag; coverage is already collected automatically "
            "when tests run"
        ),
    )
    parser.add_argument(
        "--auto-approve", action="store_true", help="Auto-approve scenarios"
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory for tests")
    parser.add_argument("--model", type=str, help="Model to use")
    args = parser.parse_args()

    if not args.codebase_path.exists():
        print(f"Error: Path does not exist: {args.codebase_path}")
        sys.exit(1)

    pipeline = PythonTestingPipeline(model=args.model)
    run_tests_flag = not args.no_run_tests  # Run tests by default unless --no-run-tests

    results = pipeline.run_pipeline(
        codebase_path=args.codebase_path,
        output_dir=args.output_dir,
        should_run_tests=run_tests_flag,
        coverage=args.coverage,
        auto_approve=args.auto_approve,
    )
    sys.exit(0 if results["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
