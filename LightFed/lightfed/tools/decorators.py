from collections import defaultdict
from functools import wraps
import logging
import types
import time


def logging_time_cost(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        logging.debug(f"start func: {func.__name__}")
        t1 = time.time()
        ans = func(*args, **kwargs)
        t2 = time.time()
        logging.debug(f"finish func: {func.__name__}, time cost: {t2 - t1}s")
        return ans

    return decorated


