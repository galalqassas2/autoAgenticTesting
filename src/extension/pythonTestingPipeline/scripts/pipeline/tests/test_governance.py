"""Unit tests for governance status reporting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.governance import FailureReason, GovernanceLog


def test_governance_status_requires_no_failures_and_no_failed_validations():
    log = GovernanceLog()
    log.log_decision("agent", "action", "because", confidence=0.9)

    summary = log.get_audit_trail()["summary"]

    assert summary["failed_validations"] == 0
    assert summary["total_failures"] == 0
    assert summary["status"] == "PASS"


def test_governance_status_requires_review_when_failures_exist():
    log = GovernanceLog()
    log.log_decision("agent", "action", "because", confidence=0.9)
    log.log_failure(FailureReason.COVERAGE_LOW, "coverage too low", iteration=1)

    summary = log.get_audit_trail()["summary"]

    assert summary["total_failures"] == 1
    assert summary["status"] == "REVIEW_NEEDED"
