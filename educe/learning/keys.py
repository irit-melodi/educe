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

    @classmethod
    def to_orange(cls, substance):
        """
        Orange-compatible string representation
        """
        if substance is cls.CONTINUOUS:
            return "c"
        elif substance is cls.DISCRETE:
            return "d"
        elif substance is cls.STRING:
            return "s"
        elif substance is cls.BASKET:
            return "basket"
        else:
            raise ValueError("Unknown substance " + substance)


class Purpose(object):
    """
    A key can have one of three purposes:

        * feature: feature for learning tasks
        * meta: some sort of indexing feature (eg. id)
        * class: the thing we are trying to learn (normally only one
          key has this purpose)

    If we ever reach a point where we're happy to switch to Python 3
    wholesale, we should subclass Enum
    """
    FEATURE = 1
    META = 2
    CLASS = 3

    @classmethod
    def to_orange(cls, purpose):
        """
        Orange-compatible string representation
        """
        if purpose is cls.FEATURE:
            return ""
        elif purpose is cls.META:
            return "m"
        elif purpose is cls.CLASS:
            return "class"
        else:
            raise ValueError("Unknown purpose" + purpose)
# pylint: enable=too-few-public-methods


class HeaderType(object):
    """
    For output files

    If we ever reach a point where we're happy to switch to Python 3
    wholesale, we should subclass Enum
    """
    OLD_CSV = 1
    NAME = 2
    SUBSTANCE = 3
    PURPOSE = 4


class Key(object):
    """
    Feature name plus a bit of metadata
    """

# pylint: disable=pointless-string-statement
    def __init__(self, substance, purpose, name, description):
        """
        You probably want to use the methods `continuous`,
        or `discrete` instead
        """
        self.substance = substance
        "see `Substance`"
        self.purpose = purpose
        "see `Purpose`"
        self.name = name
        self.description = description
# pylint: disable=pointless-string-statement

    def keycode(self):
        """
        A short code indicating the purpose and substance of the
        feature.

        If you stay within continuous/discrete for non-meta features,
        this will be compatible with those recognised by the Orange
        CSV reader.
        """
        if self.purpose is Purpose.CLASS:
            return "c"
        elif self.purpose is Purpose.META:
            return "m"
        elif self.purpose is Purpose.FEATURE:
            if self.substance is Substance.CONTINUOUS:
                return "C"
            elif self.substance is Substance.DISCRETE:
                return "D"  # not the same as that used in tab format (d)
            elif self.substance is Substance.BASKET:
                return "basket"
            else:
                oops = "Unknown substance {} in key {}".format(self.substance,
                                                               self.name)
                raise Exception(oops)
        else:
            oops = "Unknown purpose {} in key {}".format(self.purpose,
                                                         self.name)
            raise Exception(oops)

    def to_csv(self):
        "CSV-compatible string representation of this key"
        return self.keycode() + "#" + self.name

    @classmethod
    def continuous(cls, name, description, purpose=None):
        "A key for fields that have range value (eg. numbers)"
        purpose = purpose or Purpose.FEATURE
        return cls(Substance.CONTINUOUS, purpose, name, description)

    @classmethod
    def meta(cls, name, description, substance=None):
        """
        A key for fields that are used for indexing only.

        Slightly deprecated. You should really use `discrete`
        instead and specify that purpose=Purpose.META. But it's not a
        big deal.
        """
        substance = substance or Substance.STRING
        return cls(substance, Purpose.META, name, description)

    @classmethod
    def discrete(cls, name, description, purpose=None):
        "A key for fields that have a finite set of possible values"
        purpose = purpose or Purpose.FEATURE
        return cls(Substance.DISCRETE, purpose, name, description)

    @classmethod
    def basket(cls, name, description, purpose=None):
        """
        A key for fields that represent a multiset of possible values.
        Baskets should be dictionaries from string to int
        (collections.Counter would be a good bet for collecting these)
        """
        purpose = purpose or Purpose.FEATURE
        return cls(Substance.BASKET, purpose, name, description)


class MagicKey(Key):
    """
    Somewhat fancier variant of Key that is built from a function
    The goal of the magic key is to reduce the amount of boilerplate
    needed to define keys
    """
    def __init__(self, substance, purpose, function):
        name = re.sub("^feat_", "", function.__name__)
        description = function.__doc__
        super(MagicKey, self).__init__(substance, purpose, name, description)
        self.function = function

    @classmethod
    def continuous_fn(cls, function, purpose=None):
        "A key for fields that have range value (eg. numbers)"
        purpose = purpose or Purpose.FEATURE
        return cls(Substance.CONTINUOUS, purpose, function)

    @classmethod
    def meta_fn(cls, function, substance=None):
        """
        A key for fields that are used for indexing only.

        Slightly deprecated. You should really use `discrete`
        instead and specify that purpose=Purpose.META. But it's not a
        big deal.
        """
        substance = substance or Substance.STRING
        return cls(substance, Purpose.META, function)

    @classmethod
    def discrete_fn(cls, function, purpose=None):
        "A key for fields that have a finite set of possible values"
        purpose = purpose or Purpose.FEATURE
        return cls(Substance.DISCRETE, purpose, function)

    @classmethod
    def basket_fn(cls, function, purpose=None):
        """
        A key for fields that represent a multiset of possible values.
        Baskets should be dictionaries from string to int
        (collections.Counter would be a good bet for collecting these)
        """
        purpose = purpose or Purpose.FEATURE
        return cls(Substance.BASKET, purpose, function)


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
            ks = key.substance
            fn = key.name
            fv = self[fn]
            if fv is None:
                continue

            if ks is Substance.DISCRETE:
                if fv is False:
                    continue
                feature = u'{}{}={}'.format(fn, suffix, fv)
                yield (feature, 1)
            elif ks is Substance.CONTINUOUS:
                feature = u'{}{}'.format(fn, suffix)
                yield (feature, fv)
            elif ks is Substance.STRING:
                feature = u'{}{}={}'.format(fn, suffix, fv)
                yield (feature, 1)
            elif ks is Substance.BASKET:
                for k, v in fv.items():
                    feature = u'{}{}'.format(k, suffix)
                    yield (feature, v)
            else:
                raise ValueError('Unknown substance for {}'.format(ks))


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


# pylint: disable=too-many-public-methods, pointless-string-statement
class ClassKeyGroup(KeyGroup):
    """
    A key group which contains a single class key and another
    key group. The help text does not mention the class key.

    To set the class value, use `set_class`
    (the usual dictionary mechanism also works if you don't
    mind having to remember the name of the key)
    """
    def __init__(self, group, classname="CLASS"):
        keys = [Key.discrete(classname, "what we are trying to learn",
                             purpose=Purpose.CLASS)]
        self.classname = classname
        self.group = group
        self.value = None
        super(ClassKeyGroup, self).__init__("", keys)

    def set_class(self, value):
        """
        Set the value to be associated with the distinguished
        class key
        """
        self[self.classname] = value
# pylint: enable=too-many-public-methods, pointless-string-statement
