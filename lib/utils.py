import functools
import joblib

import pandas as pd


def cache(folder, sources):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_file = folder / f'{func.__name__}.jbl'
            if cache_file.exists():
                data_mtime = max(s.stat().st_mtime for s in sources)
                cache_mtime = cache_file.stat().st_mtime
                if data_mtime < cache_mtime:
                    return joblib.load(cache_file)
            result = func(*args, **kwargs)
            joblib.dump(result, cache_file)
            return result
        return wrapper
    return decorator
