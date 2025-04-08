import time
import functools
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def time_function(logger: logging.Logger = None, verbose: bool = True):

    def decorator(func):
        _logger = logger if logger else logging.getLogger(func.__name__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if verbose:
                _logger.info(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
            return result

        return wrapper

    return decorator


def is_rag(model_name: str) -> bool:
    return model_name.find("-RAG") >= 0
