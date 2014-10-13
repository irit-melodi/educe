"""
Feature extraction keys.

A key is basically a feature name, its type, some help text.

We also provide a notion of groups that allow us to organise
keys into sections
"""

from __future__ import absolute_import
import csv
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


class Purpose(object):
    """
    A key can have one of there purposes:

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
# pylint: enable=too-few-public-methods


class Key(object):
    """
    Feature name plus a bit of metadata
    """

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

    def to_csv(self):
        "Return a string that could be used as a CSV field"
        if self.purpose is Purpose.CLASS:
            code = "c"
        elif self.purpose is Purpose.META:
            code = "m"
        elif self.purpose is Purpose.FEATURE:
            if self.substance is Substance.CONTINUOUS:
                code = "C"
            elif self.substance is Substance.DISCRETE:
                code = "d"
            else:
                oops = "Unknown substance {0} in key {1}".format(self.substance,
                                                                 self.name)
                raise Exception(oops)
        else:
            oops = "Unknown purpose {0} in key {1}".format(self.purpose,
                                                           self.name)
            raise Exception(oops)
        return code + "#" + self.name

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

    def csv_headers(self):
        """
        A list of key names in a format that can be understood
        by the Orange machine learning library.
        """
        return [k.to_csv() for k in self.keys]

    def csv_values(self):
        """
        A heterogeneous list of values representing one CSV row.
        Naturally, each item in the list should correspond to the
        item in the same position of `csv_headers()` and
        vice-versa
        """
        return [self[k.name] for k in self.keys]

    def help_text(self):
        """
        A multi-line block of help text listing the features in this group
        and some details about them.

        For autogenerated features with lots of repetition, it may make
        sense to override this function and emit something more compact

        By default this comes with a "level 2" section header, eg. ::

            my feature group
            ----------------
        """
        lines = [self.description,
                 "-" * len(self.description)]
        for k in self.keys:
            lines.append("[%s] %s %s" %
                         (k.code,
                          k.name.ljust(self.NAME_WIDTH),
                          k.description))
        return "\n".join(lines)


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

    def help_text(self):
        lines = ["=" * 70,
                 self.description,
                 "=" * 70]
        for group in self.groups:
            lines.append("")
            lines.append(group.help_text())
        return "\n".join(lines)


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

    def csv_headers(self):
        """
        A list of key names in a format that can be understood
        by the Orange machine learning library.
        """
        return super(ClassKeyGroup, self).csv_headers() +\
            self.group.csv_headers()

    def csv_values(self):
        """
        A heterogeneous list of values representing one CSV row.
        Naturally, each item in the list should correspond to the
        item in the same position of `csv_headers()` and
        vice-versa
        """
        return super(ClassKeyGroup, self).csv_values() +\
            self.group.csv_values()

    def help_text(self):
        return self.group.help_text()
# pylint: enable=too-many-public-methods, pointless-string-statement


class KeyGroupWriter(object):
    """
    A CSV writer which will write rows to CSV file "f".
    Enforced UTF-8 encoding

    See the Python CSV_ docs on DictWriter. This class
    is meant to resemble that

    .. _CSV: https://docs.python.org/2/library/csv.html
    """

    def __init__(self, f, keys, dialect=csv.excel, **kwds):
        self.keys = keys
        self.writer = csv.writer(f, dialect=dialect, **kwds)

    def writeheader(self):
        """
        Write a row representing the CSV header for the
        KeyGroup object.
        """
        self.writer.writerow(self.keys.csv_headers())

    def writerow(self, row):
        """
        Write a row of KeyGroup values. The values must be
        convertible have a Unicode text representation
        (via the Python `unicode` function). The row will be
        encoded as a UTF-8 bytestring.
        """
        def bytestr(val):
            "bytestring representation of an arbitary value"
            if isinstance(val, basestring):
                return unicode(val).encode('utf-8')
            else:
                return val
        self.writer.writerow(map(bytestr, row.csv_values()))

    def writerows(self, rows):
        """
        Write out a sequence of rows
        """
        for row in rows:
            self.writerow(row)
