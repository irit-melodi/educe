"""
Basics for feature extraction
"""

from __future__ import print_function
from collections import namedtuple
from functools import wraps
import copy
import itertools
import os

import educe.util
from educe.internalutil import ifilter
from educe.learning.keys import KeyGroup, MergedKeyGroup, HeaderType,\
    ClassKeyGroup
from educe.external.postag import Token
from educe.rst_dt import (SimpleRSTTree, id_to_path,
                          ptb as r_ptb)
from educe.rst_dt.text import Sentence, Paragraph
from educe.rst_dt.deptree import RstDepTree


class FeatureExtractionException(Exception):
    """
    Exceptions related to RST trees not looking like we would
    expect them to
    """
    def __init__(self, msg):
        super(FeatureExtractionException, self).__init__(msg)


# The comments on these named tuples can be docstrings in Python3,
# or we can wrap the class, but eh...

# Global resources and settings used to extract feature vectors
FeatureInput = namedtuple('FeatureInput',
                          ['corpus', 'ptb', 'debug'])

# A document and relevant contextual information
DocumentPlus = namedtuple('DocumentPlus',
                          ['key',
                           'edus',
                           'rsttree',
                           'deptree',
                           'ptb_trees',  # only those that overlap
                           'ptb_tokens',  # only those that overlap
                           'surrounders'])


# ---------------------------------------------------------------------
# decorators for feature extraction
# ---------------------------------------------------------------------

def edu_feature(wrapped):
    """
    Lift a function from `edu -> feature` to
    `single_function_input -> feature`
    """
    @wraps(wrapped)
    def inner(_, edu):
        "drops the context"
        return wrapped(edu)
    return inner


def edu_pair_feature(wrapped):
    """
    Lifts a function from `(edu, edu) -> f` to
    `pair_function_input -> f`
    """
    @wraps(wrapped)
    def inner(_, edu1, edu2):
        "drops the context"
        return wrapped(edu1, edu2)
    return inner


def on_first_unigram(wrapped):
    """
    Lift a function from `a -> b` to `[a] -> b`
    taking the first item or returning None if empty list
    """
    @wraps(wrapped)
    def inner(things):
        "[a] -> b"
        return wrapped(things[0]) if things else None
    return inner


def on_last_unigram(wrapped):
    """
    Lift a function from `a -> b` to `[a] -> b`
    taking the last item or returning None if empty list
    """
    @wraps(wrapped)
    def inner(things):
        "[a] -> b"
        return wrapped(things[-1]) if things else None
    return inner


def on_first_bigram(wrapped):
    """
    Lift a function from `a -> string` to `[a] -> string`
    the function will be applied to the up to first two
    elements of the list and the result concatenated.
    It returns None if the list is empty
    """
    @wraps(wrapped)
    def inner(things):
        "[a] -> string"
        return " ".join(map(wrapped, things[:2])) if things else None
    return inner


def on_last_bigram(wrapped):
    """
    Lift a function from `a -> string` to `[a] -> string`
    the function will be applied to the up to the two
    elements of the list and the result concatenated.
    It returns None if the list is empty
    """
    @wraps(wrapped)
    def inner(things):
        "[a] -> string"
        return " ".join(map(wrapped, things[-2:])) if things else None
    return inner


# ---------------------------------------------------------------------
# single EDU features: meta features
# ---------------------------------------------------------------------

@edu_feature
def feat_start(edu):
    "text span start"
    return edu.text_span().char_start


@edu_feature
def feat_end(edu):
    "text span end"
    return edu.text_span().char_end


@edu_feature
def feat_id(edu):
    "some sort of unique identifier for the EDU"
    return edu.identifier()


# ---------------------------------------------------------------------
# pair of EDUs meta features
# ---------------------------------------------------------------------

def feat_grouping(current, edu1, edu2):
    "which file in the corpus this pair appears in"
    return os.path.basename(id_to_path(current.key))


# ---------------------------------------------------------------------
# single EDU key groups
# ---------------------------------------------------------------------

class SingleEduSubgroup(KeyGroup):
    """
    Abstract keygroup for subgroups of the merged SingleEduKeys.
    We use these subgroup classes to help provide modularity, to
    capture the idea that the bits of code that define a set of
    related feature vector keys should go with the bits of code
    that also fill them out
    """
    def __init__(self, description, keys):
        super(SingleEduSubgroup, self).__init__(description, keys)

    def fill(self, current, edu, target=None):
        """
        Fill out a vector's features (if the vector is None, then we
        just fill out this group; but in the case of a merged key
        group, you may find it desirable to fill out the merged
        group instead)

        This defaults to _magic_fill if you don't implement it.
        """
        self._magic_fill(current, edu, target)

    def _magic_fill(self, current, edu, target=None):
        """
        Possible fill implementation that works on the basis of
        features defined wholly as magic keys
        """
        vec = self if target is None else target
        for key in self.keys:
            vec[key.name] = key.function(current, edu)


class BaseSingleEduKeys(MergedKeyGroup):
    """Base class for single EDU features.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    def __init__(self, inputs, feature_groups):
        desc = "Single EDU features"
        super(BaseSingleEduKeys, self).__init__(desc, feature_groups)

    def fill(self, current, edu, target=None):
        """
        See `SingleEduSubgroup.fill`
        """
        vec = self if target is None else target
        for group in self.groups:
            group.fill(current, edu, vec)


# ---------------------------------------------------------------------
# EDU pairs
# ---------------------------------------------------------------------

class PairSubgroup(KeyGroup):
    """
    Abstract keygroup for subgroups of the merged PairKeys.
    We use these subgroup classes to help provide modularity, to
    capture the idea that the bits of code that define a set of
    related feature vector keys should go with the bits of code
    that also fill them out
    """
    def __init__(self, description, keys):
        super(PairSubgroup, self).__init__(description, keys)

    def fill(self, current, edu1, edu2, target=None):
        """
        Fill out a vector's features (if the vector is None, then we
        just fill out this group; but in the case of a merged key
        group, you may find it desirable to fill out the merged
        group instead)

        Defaults to _magic_fill if not defined
        """
        self._magic_fill(current, edu1, edu2, target)

    def _magic_fill(self, current, edu1, edu2, target=None):
        """
        Possible fill implementation that works on the basis of
        features defined wholly as magic keys
        """
        vec = self if target is None else target
        for key in self.keys:
            vec[key.name] = key.function(current, edu1, edu2)


class BasePairKeys(MergedKeyGroup):
    """Base class for EDU pair features.

    Parameters
    ----------

    sf_cache :  dict(EDU, SingleEduKeys), optional (default=None)
        Should only be None if you're just using this to generate help text.
    """

    def __init__(self, inputs, pair_feature_groups, sf_cache=None):
        self.sf_cache = sf_cache

        if sf_cache is None:
            self.edu1 = self.init_single_features(inputs)
            self.edu2 = self.init_single_features(inputs)
        else:
            self.edu1 = None  # will be filled out later
            self.edu2 = None  # from the feature cache

        desc = "pair features"
        super(BasePairKeys, self).__init__(desc, pair_feature_groups)

    def init_single_features(self, inputs):
        """Init features defined on single EDUs"""
        raise NotImplementedError()

    def csv_headers(self, htype=False):
        if htype in [HeaderType.OLD_CSV, HeaderType.NAME]:
            return (super(BasePairKeys, self).csv_headers(htype) +
                    [h + "_EDU1" for h in self.edu1.csv_headers(htype)] +
                    [h + "_EDU2" for h in self.edu2.csv_headers(htype)])
        else:
            return (super(BasePairKeys, self).csv_headers(htype) +
                    self.edu1.csv_headers(htype) +
                    self.edu2.csv_headers(htype))

    def csv_values(self):
        return (super(BasePairKeys, self).csv_values() +
                self.edu1.csv_values() +
                self.edu2.csv_values())

    def help_text(self):
        lines = [super(BasePairKeys, self).help_text(),
                 "",
                 self.edu1.help_text()]
        return "\n".join(lines)

    def fill(self, current, edu1, edu2, target=None):
        "See `PairSubgroup`"
        vec = self if target is None else target
        vec.edu1 = self.sf_cache[edu1]
        vec.edu2 = self.sf_cache[edu2]
        for group in self.groups:
            group.fill(current, edu1, edu2, vec)


# ---------------------------------------------------------------------
# extraction generators
# ---------------------------------------------------------------------

def get_sentence(current, edu):
    "get sentence surrounding this EDU"
    # TODO do we need bounds-checking or lookup failure handling?
    return current.surrounders[edu][1]


def get_paragraph(current, edu):
    "get paragraph surrounding this EDU"
    # TODO do we need bounds-checking or lookup failure handling?
    return current.surrounders[edu][0]


# tree utils
# TODO move to a more appropriate place
def simplify_deptree(dtree):
    """
    Boil a dependency tree down into a dictionary from (edu, edu) to rel
    """
    relations = {(src, tgt): rel
                 for src, tgt, rel in dtree.get_dependencies()}
    return relations


def lowest_common_parent(treepositions):
    """Find tree position of the lowest common parent of a list of nodes."""
    if not treepositions:
        return None

    leftmost_tpos = treepositions[0]
    rightmost_tpos = treepositions[-1]

    for i in range(len(leftmost_tpos)):
        if ((i == len(rightmost_tpos) or
             leftmost_tpos[i] != rightmost_tpos[i])):
            tpos_parent = leftmost_tpos[:i]
            break
    else:
        tpos_parent = leftmost_tpos

    return tpos_parent
# end of tree utils


def containing(span):
    """
    span -> anno -> bool

    if this annotation encloses the given span
    """
    return lambda x: x.text_span().encloses(span)


def _filter0(pred, iterable):
    """
    First item that satisifies a predicate in a given
    iterable, otherwise None
    """
    matches = ifilter(pred, iterable)
    try:
        return matches.next()
    except StopIteration:
        return None


def _surrounding_text(edu):
    """
    Determine which paragraph and sentence (if any) surrounds
    this EDU. Try to accomodate the occasional off-by-a-smidgen
    error by folks marking these EDU boundaries, eg. original
    text:

    Para1: "Magazines are not providing us in-depth information on
    circulation," said Edgar Bronfman Jr., .. "How do readers feel
    about the magazine?...
    Research doesn't tell us whether people actually do read the
    magazines they subscribe to."

    Para2: Reuben Mark, chief executive of Colgate-Palmolive, said...

    Marked up EDU is wide to the left by three characters:
    "

    Reuben Mark, chief executive of Colgate-Palmolive, said...
    """
    if edu.is_left_padding():
        sent = Sentence.left_padding()
        para = Paragraph.left_padding([sent])
        return para, sent
    # normal case
    espan = edu.text_span()
    para = _filter0(containing(espan), edu.context.paragraphs)
    # sloppy EDUs happen; try shaving off some characters
    # if we can't find a paragraph
    if para is None:
        espan = copy.copy(espan)
        espan.char_start += 1
        espan.char_end -= 1
        etext = edu.context.text(espan)
        # kill left whitespace
        espan.char_start += len(etext) - len(etext.lstrip())
        etext = etext.lstrip()
        # kill right whitespace
        espan.char_end -= len(etext) - len(etext.rstrip())
        etext = etext.rstrip()
        # try again
        para = _filter0(containing(espan), edu.context.paragraphs)

    sent = _filter0(containing(espan), para.sentences) if para else None
    return para, sent


def _ptb_stuff(doc_ptb_trees, edu):
    """
    The PTB trees and tokens which are relevant to any given edu
    """
    if doc_ptb_trees is None:
        return None, None
    if edu.is_left_padding():
        start_token = Token.left_padding()
        ptb_tokens = [start_token]
        ptb_trees = []
    else:
        ptb_trees = [t for t in doc_ptb_trees if t.overlaps(edu)]
        all_tokens = itertools.chain.from_iterable(t.leaves()
                                                   for t in ptb_trees)
        ptb_tokens = [tok for tok in all_tokens if tok.overlaps(edu)]
    return ptb_trees, ptb_tokens


def preprocess(inputs, k):
    """
    Pre-process and bundle up a representation of the current document
    """
    rtree = SimpleRSTTree.from_rst_tree(inputs.corpus[k])
    # convert to deptree
    dtree = RstDepTree.from_simple_rst_tree(rtree)
    edus = dtree.edus

    # align with document structure
    surrounders = {edu: _surrounding_text(edu) for edu in edus}
    # align with syntactic structure
    doc_ptb_trees = r_ptb.parse_trees(inputs.corpus, k, inputs.ptb)
    ptb_trees = {}
    ptb_tokens = {}
    for edu in edus:
        ptb_trees[edu], ptb_tokens[edu] = _ptb_stuff(doc_ptb_trees, edu)

    return DocumentPlus(key=k,
                        edus=edus,
                        rsttree=rtree,
                        deptree=dtree,
                        ptb_trees=ptb_trees,
                        ptb_tokens=ptb_tokens,
                        surrounders=surrounders)


def extract_pair_features(inputs, feature_set, live=False):
    """
    Return a pair of dictionaries, one for attachments
    and one for relations
    """

    for k in inputs.corpus:
        current = preprocess(inputs, k)
        edus = current.edus
        # reduced dependency graph as dictionary (edu to [edu])
        relations = simplify_deptree(current.deptree) if not live else {}

        # single edu features
        sf_cache = {}
        for edu in edus:
            sf_cache[edu] = feature_set.SingleEduKeys(inputs)
            sf_cache[edu].fill(current, edu)

        # pairs
        # the fake root cannot have any incoming edge
        for epair in itertools.product(edus, edus[1:]):
            edu1, edu2 = epair
            if edu1 == edu2:
                continue
            vec = feature_set.PairKeys(inputs, sf_cache=sf_cache)
            vec.fill(current, edu1, edu2)

            if live:
                yield vec, vec
            else:
                pairs_vec = ClassKeyGroup(vec)
                pairs_vec.set_class(epair in relations)
                rels_vec = ClassKeyGroup(vec)
                rels_vec.set_class(relations[epair] if epair in relations
                                   else 'UNRELATED')

                yield pairs_vec, rels_vec


# ---------------------------------------------------------------------
# input readers
# ---------------------------------------------------------------------


def read_common_inputs(args, corpus, ptb):
    """
    Read the data that is common to live/corpus mode.
    """
    return FeatureInput(corpus, ptb, args.debug)


def read_help_inputs(_):
    """
    Read the data (if any) that is needed just to produce
    the help text
    """
    return FeatureInput(None, None, True)


def read_corpus_inputs(args):
    """
    Read the data (if any) that is needed just to produce
    training data
    """
    is_interesting = educe.util.mk_is_interesting(args)
    reader = educe.rst_dt.Reader(args.corpus)
    anno_files = reader.filter(reader.files(), is_interesting)
    corpus = reader.slurp(anno_files, verbose=True)
    ptb = r_ptb.reader(args.ptb)
    return read_common_inputs(args, corpus, ptb)
