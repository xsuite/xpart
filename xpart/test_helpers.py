# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2022.                 #
# ######################################### #

from contextlib import contextmanager
from functools import wraps
from typing import Optional, Tuple, Type, Union


class _RetryableError(Exception):
    def __init__(self, original_error: Exception):
        self.message = ''
        self.original_error = original_error
        self.retries = None

    def __str__(self):
        return self.message


def retry(
        on: Optional[Union[Type[Exception], Tuple[Type[Exception]]]] = None,
        n_times: int = 3,
):
    """
    A decorator to be used on a flaky test. The test will be rerun until no
    exception defined in `on` is thrown, up to `n_times`. If `on` is
    unspecified, then the functionality of `flaky_assertions` is expected to be
    applied: the test will only be retried on AssertionErrors occurring within
    a `with flaky_assertions(): ...` statement.
    """
    if not on:
        on = _RetryableError

    def decorator(test_function):
        @wraps(test_function)
        def wrapper(*args, **kwargs):
            for i in range(n_times):
                try:
                    return test_function(*args, **kwargs)
                except on as e:
                    if i + 1 < n_times:
                        continue

                    if isinstance(e, _RetryableError):
                        # This is maybe not the prettiest, but let's hide
                        # the traceback as it pollutes the pytest output,
                        # and detracts from the traceback of the actual
                        # error:
                        e.__traceback__ = None
                        e.message = (f'Failing on retry number {i + 1}. '
                                     f'Raising the original error now!')

                        raise e.original_error
                    raise e
        return wrapper
    return decorator


@contextmanager
def flaky_assertions():
    try:
        yield
    except AssertionError as e:
        raise _RetryableError(original_error=e)
