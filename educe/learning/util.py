"""
Common helper functions for feature extraction.
"""

from functools import wraps


def tuple_feature(combine):
    """
    ::

       (a -> a -> b) ->
       ((current, cache, edu) -> a) ->
       (current, cache, edu, edu) -> b)

    Combine the result of single-edu feature function to make
    a pair feature
    """

    def decorator(wrapped):
        "apply the combiner"

        @wraps(wrapped)
        def inner(current, cache, edu1, edu2):
            "wrapped :: (current, cache, edu) -> String"
            val1 = wrapped(current, cache, edu1)
            val2 = wrapped(current, cache, edu2)
            return combine(val1, val2)
        return inner
    return decorator


def underscore(str1, str2):
    """
    join two strings with an underscore
    """
    return '%s_%s' % (str1, str2)


def space_join(str1, str2):
    """
    join two strings with a space
    """
    return '%s %s' % (str1, str2)
