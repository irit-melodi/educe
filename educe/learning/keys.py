"""
Feature extraction keys.

A key is basically a feature name, its type, some help text.

We also provide a notion of groups that allow us to organise
keys into sections
"""

from __future__ import absolute_import

import re


# pylint: disable=too-few-public-methods
class Substance(object):
    """
    The kind of the variable represented by this key.

       * continuous
       * discrete
       * string (for meta vars; you probably want discrete instead)

    If we ever reach a point where we're happy to switch to Python 3
    wholesale, we should subclass Enum
    """
    CONTINUOUS = 1
    DISCRETE = 2
    STRING = 3
    BASKET = 4


class Key(object):
    """
    Feature name plus a bit of metadata
    """

# pylint: disable=pointless-string-statement
    def __init__(self, substance, name, description):
        """
        You probably want to use the methods `continuous`,
        or `discrete` instead
        """
        self.substance = substance
        "see `Substance`"
        self.name = name
        self.description = description
# pylint: disable=pointless-string-statement

    @classmethod
    def continuous(cls, name, description):
        "A key for fields that have range value (eg. numbers)"
        return cls(Substance.CONTINUOUS, name, description)

    @classmethod
    def discrete(cls, name, description):
        "A key for fields that have a finite set of possible values"
        return cls(Substance.DISCRETE, name, description)

    @classmethod
    def basket(cls, name, description):
        """
        A key for fields that represent a multiset of possible values.
        Baskets should be dictionaries from string to int
        (collections.Counter would be a good bet for collecting these)
        """
        return cls(Substance.BASKET, name, description)


class MagicKey(Key):
    """
    Somewhat fancier variant of Key that is built from a function
    The goal of the magic key is to reduce the amount of boilerplate
    needed to define keys
    """
    def __init__(self, substance, function):
        name = re.sub("^feat_", "", function.__name__)
        description = function.__doc__
        super(MagicKey, self).__init__(substance, name, description)
        self.function = function

    @classmethod
    def continuous_fn(cls, function):
        "A key for fields that have range value (eg. numbers)"
        return cls(Substance.CONTINUOUS, function)

    @classmethod
    def discrete_fn(cls, function):
        "A key for fields that have a finite set of possible values"
        return cls(Substance.DISCRETE, function)

    @classmethod
    def basket_fn(cls, function):
        """
        A key for fields that represent a multiset of possible values.
        Baskets should be dictionaries from string to int
        (collections.Counter would be a good bet for collecting these)
        """
        return cls(Substance.BASKET, function)


class KeyGroup(dict):
    """
    A set of related features.

    Note that a KeyGroup can be used as a dictionary, but instead
    of using Keys as values, you use the key names
    """
    NAME_WIDTH = 35
    DEBUG = True

    def __init__(self, description, keys):
        self.description = description
        self.keys = keys
        self.keynames = [key.name for key in keys]
        super(KeyGroup, self).__init__()

    def __setitem__(self, key, val):
        if self.DEBUG and key not in self.keynames:
            raise KeyError(key)
        else:
            super(KeyGroup, self).__setitem__(key, val)

    def one_hot_values_gen(self, suffix=''):
        """Get a one-hot encoded version of this KeyGroups as a generator

        suffix is added to the feature name
        """
        for key in self.keys:
            subst = key.substance
            kname = key.name
            fval = self[kname]
            if fval is None:
                continue

            if subst is Substance.DISCRETE:
                if fval is False:
                    continue
                feature = u'{}{}={}'.format(kname, suffix, fval)
                yield (feature, 1)
            elif subst is Substance.CONTINUOUS:
                feature = u'{}{}'.format(kname, suffix)
                yield (feature, fval)
            elif subst is Substance.STRING:
                feature = u'{}{}={}'.format(kname, suffix, fval)
                yield (feature, 1)
            elif subst is Substance.BASKET:
                for bkey, bval in fval.items():
                    feature = u'{}{}'.format(bkey, suffix)
                    yield (feature, bval)
            else:
                raise ValueError('Unknown substance for {}'.format(subst))


class MergedKeyGroup(KeyGroup):
    """
    A key group that is formed by fusing several key groups
    into one.

    Note that for now all the keys in a merged group are lumped
    into the same object.

    The help text tries to preserve the internal breakdown into
    the subgroups, however.  It comes with a "level 1" section
    header, eg. ::

        =======================================================
        big block of features
        =======================================================
    """
    def __init__(self, description, groups):
        self.groups = groups
        keys = []
        for group in groups:
            keys.extend(group.keys)
        super(MergedKeyGroup, self).__init__(description, keys)
