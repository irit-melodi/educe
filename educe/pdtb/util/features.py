"""
Feature extraction library functions for PDTB corpus
"""

from collections import namedtuple
import copy
import itertools
import os
import re

from educe.learning.keys import Key, KeyGroup, MergedKeyGroup, ClassKeyGroup
import educe.pdtb
from educe.stac.util.features import tune_for_csv, treenode, CorpusConsistencyException

# pylint: disable=too-many-public-methods

# ---------------------------------------------------------------------
# features
# ---------------------------------------------------------------------


def _kg(*args):
    """
    Shorthand for KeyGroup, just to save on some indentation
    """
    return KeyGroup(*args)


# ---------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------

# The comments on these named tuples can be docstrings in Python3,
# or we can wrap the class, but eh...

# Global resources and settings used to extract feature vectors
FeatureInput = namedtuple('FeatureInput',
                          ['corpus', 'debug'])

# A document and relevant contextual information
DocumentPlus = namedtuple('DocumentPlus',
                          ['key', 'doc'])


def spans_to_str(spans):
    "string representation of a list of spans, meant to work as an id"
    return "__".join("%d_%d" % x for x in spans)

# ---------------------------------------------------------------------
# single Arg non-lexical features
# ---------------------------------------------------------------------


class SingleArgSubgroup(KeyGroup):
    """
    Abstract keygroup for subgroups of the merged SingleArgKeys.
    We use these subgroup classes to help provide modularity, to
    capture the idea that the bits of code that define a set of
    related feature vector keys should go with the bits of code
    that also fill them out
    """
    def __init__(self, description, keys):
        super(SingleArgSubgroup, self).__init__(description, keys)

    def fill(self, current, arg, target=None):
        """
        Fill out a vector's features (if the vector is None, then we
        just fill out this group; but in the case of a merged key
        group, you may find it desirable to fill out the merged
        group instead)
        """
        raise NotImplementedError("fill should be implemented by a subclass")


# TODO - what could we use for meta features for the arguments here?
class SingleArgSubgroup_Meta(SingleArgSubgroup):
    """
    arg-identification features
    """
    def __init__(self):
        desc = self.__doc__.strip()
        keys =\
            [Key.meta("id",
                      "some sort of unique identifier for the EDU")]
        super(SingleArgSubgroup_Meta, self).__init__(desc, keys)

    def fill(self, current, arg, target=None):
        vec = self if target is None else target
        vec["id"] = spans_to_str(arg.span)


class SingleArgSubgroup_Debug(SingleArgSubgroup):
    """
    debug features
    """
    def __init__(self):
        desc = self.__doc__.strip()
        keys = [Key.meta("text", "EDU text [debug only]")]
        super(SingleArgSubgroup_Debug, self).__init__(desc, keys)

    def fill(self, current, arg, target=None):
        vec = self if target is None else target
        doc = current.doc
        arg_span = arg.text_span()
        vec["text"] = tune_for_csv(doc.text(arg_span))


class SingleArgKeys(MergedKeyGroup):
    """
    Features for a single EDU
    """
    def __init__(self, inputs):
        groups = [SingleArgSubgroup_Meta()]
        if inputs.debug:
            groups.append(SingleArgSubgroup_Debug())
        super(SingleArgKeys, self).__init__("single arg features",
                                            groups)

    def fill(self, current, arg, target=None):
        """
        See `SingleArgSubgroup.fill`
        """
        vec = self if target is None else target
        for group in self.groups:
            group.fill(current, arg, vec)

# ---------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------


class RelSubgroup(KeyGroup):
    """
    Abstract keygroup for subgroups of the merged RelKeys.
    We use these subgroup classes to help provide modularity, to
    capture the idea that the bits of code that define a set of
    related feature vector keys should go with the bits of code
    that also fill them out
    """
    def __init__(self, description, keys):
        super(RelSubgroup, self).__init__(description, keys)

    def fill(self, current, rel, target=None):
        """
        Fill out a vector's features (if the vector is None, then we
        just fill out this group; but in the case of a merged key
        group, you may find it desirable to fill out the merged
        group instead)
        """
        raise NotImplementedError("fill should be implemented by a subclass")


class RelSubgroup_Debug(RelSubgroup):
    "debug features"

    def __init__(self):
        desc = self.__doc__.strip()
        keys = [Key.meta("text",
                         "text from DU1 start to DU2 end [debug only]")]
        super(RelSubgroup_Debug, self).__init__(desc, keys)

    def fill(self, current, rel, target=None):
        vec = self if target is None else target
        doc = current.doc
        vec["text"] = None  # TODO


class RelSubGroup_Core(RelSubgroup):
    "core features"

    def __init__(self):
        desc = self.__doc__.strip()
        keys =\
            [Key.meta("document", "document the relation appears in"),
             Key.meta("id", "id for this relation")]
        super(RelSubGroup_Core, self).__init__(desc, keys)

    def fill(self, current, rel, target=None):
        vec = self if target is None else target
        vec["document"] = current.key.doc
        vec["id"] = spans_to_str(rel.span)


class RelKeys(MergedKeyGroup):
    """
    Features for relations
    """
    def __init__(self, inputs):
        groups = [RelSubGroup_Core()]
        if inputs.debug:
            groups.append(RelSubgroup_Debug())

        self.arg1 = SingleArgKeys(inputs)
        self.arg2 = SingleArgKeys(inputs)

        super(RelKeys, self).__init__("relation features",
                                      groups)

    def csv_headers(self):
        return super(RelKeys, self).csv_headers() +\
            [h + "_Arg1" for h in self.arg1.csv_headers()] +\
            [h + "_Arg2" for h in self.arg2.csv_headers()]

    def csv_values(self):
        return super(RelKeys, self).csv_values() +\
            self.arg1.csv_values() +\
            self.arg2.csv_values()

    def help_text(self):
        lines = [super(RelKeys, self).help_text(),
                 "",
                 self.arg1.help_text()]
        return "\n".join(lines)

    def fill(self, current, rel, target=None):
        "See `RelSubgroup`"
        vec = self if target is None else target
        vec.arg1.fill(current, rel.arg1)
        vec.arg2.fill(current, rel.arg2)
        for group in self.groups:
            group.fill(current, rel, vec)

# ---------------------------------------------------------------------
# extraction generators
# ---------------------------------------------------------------------


def mk_current(inputs, k):
    """
    Pre-process and bundle up a representation of the current document
    """

    # this may be fleshed out in time with other companion docs,
    # for example the original PTB tools that go with it.
    doc = inputs.corpus[k]
    return DocumentPlus(k, doc)


def extract_rel_features(inputs):
    """
    Return a pair of dictionaries, one for attachments
    and one for relations
    """
    for k in inputs.corpus:
        current = mk_current(inputs, k)
        for rel in current.doc:
            vec = RelKeys(inputs)
            vec.fill(current, rel)
            yield vec
