"""
Feature extraction library functions for PDTB corpus
"""

from collections import namedtuple

from educe.learning.keys import KeyGroup, MergedKeyGroup

# pylint: disable=too-many-public-methods

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


class SingleArgKeys(MergedKeyGroup):
    """
    Features for a single EDU
    """
    def __init__(self, inputs):
        groups = []
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


class RelSubGroup_Core(RelSubgroup):
    "core features"

    def __init__(self):
        desc = self.__doc__.strip()
        keys = []
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
        self.arg1 = SingleArgKeys(inputs)
        self.arg2 = SingleArgKeys(inputs)

        super(RelKeys, self).__init__("relation features",
                                      groups)

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
