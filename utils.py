import time
import functools
import logging

from logging import Logger


def time_function(logger: Logger = None, verbose: bool = True):

    def decorator(func):
        _logger = logger if logger else logging.getLogger(func.__name__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if verbose:
                _logger.info(
                    f"{func.__name__} took {end_time - start_time:.4f} seconds to execute"
                )
            return result

        return wrapper

    return decorator
