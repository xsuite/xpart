from functools import wraps
from typing import Tuple, Type, Union


def retry(on: Union[Type[Exception], Tuple[Type[Exception]]],
          n_times: int = 3):
    """
    A decorator to be used on a flaky test. The test will be rerun until no
    exception defined in `on` is thrown, up to `n_times`.
    """
    def decorator(test_function):
        @wraps(test_function)
        def wrapper(*args, **kwargs):
            for i in range(n_times):
                try:
                    return test_function(*args, **kwargs)
                except on as e:
                    if i + 1 < n_times:
                        continue
                    else:
                        raise e
        return wrapper
    return decorator
