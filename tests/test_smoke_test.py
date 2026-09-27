"""
Unit test for end-to-end infrastructure smoke test execution.
"""

from python_infra.smoke_test import run_infrastructure_smoke_test


def test_smoke_test_execution():
    """Validates that the infrastructure smoke test runs cleanly to completion."""
    result = run_infrastructure_smoke_test()
    assert result is True
