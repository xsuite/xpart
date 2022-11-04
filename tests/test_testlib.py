import random
import pytest

from xpart.test_helpers import retry


@retry(on=AssertionError, n_times=100)
def test_retry_passes():
    """Below assertion almost always fails, however the test should
    practically always pass (1 in 38k chance of failure)."""
    assert random.random() > 0.9


def test_retry_fails():
    @retry(on=AssertionError, n_times=2)
    def actual_test():
        """Below assertion practically always fails, thus the test is
        expected to fail as well after only 2 runs."""
        assert random.random() > 0.99_999

    with pytest.raises(AssertionError):
        actual_test()
