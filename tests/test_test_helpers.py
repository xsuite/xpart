import random

import pytest

from xpart.test_helpers import flaky_assertions, retry


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


@pytest.mark.parametrize(
    'inner_when, outer_when, error_substring',
    [
        (3, 3, 'inner'),
        (0, 3, 'outer'),
        (3, 0, 'inner'),
        (3, 1, 'inner'),
        (2, 1, 'outer'),
        (0, 0, None),
        (2, 0, None),
    ]
)
def test_retry_with_context_many_errors_fails(
        inner_when,
        outer_when,
        error_substring
):
    class FlakyFailer:
        """A class simulating flaky behaviour.

        The failer shall succeed after `succeed_after` failures with an
        assertion error containing `name` in the message.
        """
        def __init__(self, succeed_after, name):
            self.name = name
            self.when = succeed_after

        def test(self):
            if self.when == 0:
                return

            self.when -= 1
            raise AssertionError(f'Failing {self.name}')

    inner_failer = FlakyFailer(succeed_after=inner_when, name='inner')
    outer_failer = FlakyFailer(succeed_after=outer_when, name='outer')

    @retry(n_times=3)
    def actual_test():
        with flaky_assertions():
            inner_failer.test()

        outer_failer.test()

    if error_substring:  # if `error_substring` is None, assert test failure
        with pytest.raises(AssertionError) as e_info:
            actual_test()

        assert error_substring in e_info.exconly()
    else:  # assert test success
        actual_test()
