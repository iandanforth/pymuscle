from functools import wraps
from time import time


def timing(f):
    """
    From https://stackoverflow.com/a/27737385/1775741
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        dur = te - ts
        print('func:{!r} args:[{!r}, {!r}] took: {:2.4f} sec'.format(
            f.__name__, args, kw, dur)
        )
        return dur, result
    return wrap
