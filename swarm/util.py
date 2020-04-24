#  -*- coding: utf-8 -*-

import functools


def collector(aggregator_func):
    """Combines the output of a generator with *aggregator_func*
        >>> @collector(" ".join)
        ... def string_yielder():
        ...     yield "Hello"
        ...     yield "World!"
        "Hello World!
    Allows the usage of yields in the code without needing to convert to list allowing
    for more expressive and cleaner code instead of ``list.append``
    This is particularly nice when interfacing with code that recieves data in a
    paginated fashion but you'd like to return a ``pd.DataFrame`` or ``np.array``
    This can also be chained.
    """
    def wrapper(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            return aggregator_func(func(*args, **kwargs))

        return inner

    return wrapper
