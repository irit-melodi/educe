"""
Feature extraction library functions for RST_DT corpus
"""

from __future__ import print_function
from collections import namedtuple, Counter
from functools import wraps
import copy
import itertools
import os
import re

import educe.util
from educe.internalutil import treenode, ifilter
from educe.rst_dt import SimpleRSTTree, deptree, id_to_path
from educe.rst_dt import ptb as r_ptb
from educe.learning.csv import tune_for_csv
from educe.learning.keys import\
    ClassKeyGroup, KeyGroup, MergedKeyGroup, HeaderType,\
    MagicKey
from educe.learning.util import tuple_feature, underscore

class FeatureExtractionException(Exception):
    """
    Exceptions related to RST trees not looking like we would
    expect them to
    """
    def __init__(self, msg):
        super(FeatureExtractionException, self).__init__(msg)


# ---------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------

# The comments on these named tuples can be docstrings in Python3,
# or we can wrap the class, but eh...

# Global resources and settings used to extract feature vectors
FeatureInput = namedtuple('FeatureInput',
                          ['corpus', 'ptb', 'debug'])

# A document and relevant contextual information
DocumentPlus = namedtuple('DocumentPlus',
                          ['key',
                           'rsttree',
                           'deptree',
                           'ptb_trees',  # only those that overlap
                           'ptb_tokens', # only those that overlap
                           'surrounders'])

# ---------------------------------------------------------------------
# single EDUs
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


def clean_edu_text(text):
    """
    Strip metadata from EDU text and compress extraneous whitespace
    """
    clean_text = text
    clean_text = re.sub(r'(\.|,)*$', r'', clean_text)
    clean_text = re.sub(r'^"', r'', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text


def clean_corpus_word(word):
    """
    Given a word from the corpus, return a slightly normalised
    version of that word
    """
    return word.lower()


def tokens_feature(wrapped):
    """
    Lift a function from `tokens -> feature` to
    `single_function_input -> feature`
    """
    @edu_feature
    @wraps(wrapped)
    def inner(edu):
        "(edu -> f) -> ((context, edu) -> f)"
        tokens = [tune_for_csv(clean_corpus_word(x))
                  for x in clean_edu_text(edu.text()).split()]
        return wrapped(tokens)
    return inner


def ptb_tokens_feature(wrapped):
    """
    Lift a function from `[ptb_token] -> feature`
    to `single_function_input -> feature`
    """
    @wraps(wrapped)
    def inner(context, edu):
        "([ptb_token] -> f) -> ((context, edu) -> f)"
        tokens = context.ptb_tokens[edu]
        return tune_for_csv(wrapped(tokens)) if tokens is not None else None
    return inner


# ---------------------------------------------------------------------
# single EDU features
# ---------------------------------------------------------------------


def on_first_unigram(wrapped):
    """
    Lift a function from `a -> b` to `[a] -> b`
    taking the first item or returning None if empty list
    """
    @wraps(wrapped)
    def inner(things):
        "[a] -> b"
        return clean_edu_text(wrapped(things[0])) if things else None
    return inner


def on_last_unigram(wrapped):
    """
    Lift a function from `a -> b` to `[a] -> b`
    taking the last item or returning None if empty list
    """
    @wraps(wrapped)
    def inner(things):
        "[a] -> b"
        return clean_edu_text(wrapped(things[-1])) if things else None
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
        return clean_edu_text(" ".join(map(wrapped, things[:2])))\
            if things else None
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
        return clean_edu_text(" ".join(map(wrapped, things[-2:])))\
            if things else None
    return inner


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


@tokens_feature
@on_first_unigram
def word_first(token):
    "first word in the EDU (normalised)"
    return token


@tokens_feature
@on_last_unigram
def word_last(token):
    "last word in the EDU (normalised)"
    return token


@tokens_feature
@on_first_bigram
def bigram_first(token):
    "first two words in the EDU (normalised)"
    return token


@tokens_feature
@on_last_bigram
def bigram_last(token):
    "first two words in the EDU (normalised)"
    return token


@tokens_feature
def num_tokens(tokens):
    "number of distinct tokens in EDU text"
    return len(tokens)


# PTB unigrams

@ptb_tokens_feature
@on_first_unigram
def ptb_pos_tag_first(token):
    "POS tag for first PTB token in the EDU"
    # note: PTB tokens may not necessarily correspond to words
    return token.tag


@ptb_tokens_feature
@on_last_unigram
def ptb_pos_tag_last(token):
    "POS tag for last PTB token in the EDU"
    # note: PTB tokens may not necessarily correspond to words
    return token.tag


@ptb_tokens_feature
@on_first_unigram
def ptb_word_first(token):
    "first PTB word in the EDU"
    return token.word


@ptb_tokens_feature
@on_last_unigram
def ptb_word_last(token):
    "last PTB word in the EDU"
    return token.word


# PTB bigrams

@ptb_tokens_feature
@on_first_bigram
def ptb_pos_tag_first2(token):
    "POS tag for first two PTB tokens in the EDU"
    # note: PTB tokens may not necessarily correspond to words
    return token.tag


@ptb_tokens_feature
@on_last_bigram
def ptb_pos_tag_last2(token):
    "POS tag for last two PTB tokens in the EDU"
    # note: PTB tokens may not necessarily correspond to words
    return token.tag


@ptb_tokens_feature
@on_first_bigram
def ptb_word_first2(token):
    "first two PTB words in the EDU"
    return token.word


@ptb_tokens_feature
@on_last_bigram
def ptb_word_last2(token):
    "last PTB words in the EDU"
    return token.word


# ---------------------------------------------------------------------
# pair EDU features
# ---------------------------------------------------------------------


@edu_pair_feature
def num_edus_between(edu1, edu2):
    "number of EDUs between the two EDUs"
    return abs(edu2.num - edu1.num) - 1


def feat_grouping(current, edu1, edu2):
    "which file in the corpus this pair appears in"
    return os.path.basename(id_to_path(current.key))


def same_paragraph(current, edu1, edu2):
    "if in the same paragraph"
    para1 = current.surrounders[edu1][0]
    para2 = current.surrounders[edu2][0]
    return para1 is not None and para2 is not None and\
        para1 == para2


def same_bad_sentence(current, edu1, edu2):
    "if in the same sentence (bad segmentation)"
    sent1 = current.surrounders[edu1][1]
    sent2 = current.surrounders[edu2][1]
    return sent1 is not None and sent2 is not None and\
        sent1 == sent2


def same_ptb_sentence(current, edu1, edu2):
    "if in the same sentence (ptb segmentation)"
    sents1 = current.ptb_trees[edu1]
    sents2 = current.ptb_trees[edu2]
    if sents1 is None or sents2 is None:
        return False
    context = edu1.context
    has_overlap = bool([s for s in sents1 if s in sents2])
    return has_overlap


@tuple_feature(underscore)
def word_first_pairs(_, cache, edu):
    "pair of the first words in the two EDUs"
    return cache[edu]["word_first"]


@tuple_feature(underscore)
def word_last_pairs(_, cache, edu):
    "pair of the last words in the two EDUs"
    return cache[edu]["word_last"]


@tuple_feature(underscore)
def bigram_first_pairs(_, cache, edu):
    "pair of the first bigrams in the two EDUs"
    return cache[edu]["bigram_first"]


@tuple_feature(underscore)
def bigram_last_pairs(_, cache, edu):
    "pair of the last bigrams in the two EDUs"
    return cache[edu]["bigram_last"]


@tuple_feature(underscore)
def ptb_pos_tag_first_pairs(_, cache, edu):
    "pair of the first POS in the two EDUs"
    # FIXME: should be POS tag of first non-nil ptb word
    return cache[edu]["ptb_pos_tag_first"] # or ["pos_first2"]


def ptb_pos_tags_in_first(current, edu1, _):
    "demonstrator for use of basket features"
    tokens = current.ptb_tokens[edu1]
    return Counter(t.tag for t in tokens) if tokens is not None else None


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


class SingleEduSubgroup_Meta(SingleEduSubgroup):
    """
    Basic EDU-identification features
    """

    _features =\
        [MagicKey.meta_fn(feat_id),
         MagicKey.meta_fn(feat_start),
         MagicKey.meta_fn(feat_end)]

    def __init__(self):
        desc = self.__doc__.strip()
        super(SingleEduSubgroup_Meta, self).__init__(desc, self._features)


class SingleEduSubgroup_Text(SingleEduSubgroup):
    """
    Properties of the EDU text itself
    """
    _features =\
        [MagicKey.discrete_fn(word_first),
         MagicKey.discrete_fn(word_last),
         MagicKey.discrete_fn(bigram_first),
         MagicKey.discrete_fn(bigram_last),
         MagicKey.continuous_fn(num_tokens)]

    def __init__(self):
        desc = self.__doc__.strip()
        super(SingleEduSubgroup_Text, self).__init__(desc, self._features)


class SingleEduSubgroup_Ptb(SingleEduSubgroup):
    """
    Penn Treebank properties for the EDU
    """
    _features =\
        [MagicKey.discrete_fn(ptb_word_first),
         MagicKey.discrete_fn(ptb_word_last),
         MagicKey.discrete_fn(ptb_pos_tag_first),
         MagicKey.discrete_fn(ptb_pos_tag_last),
         MagicKey.discrete_fn(ptb_word_first2),
         MagicKey.discrete_fn(ptb_word_last2),
         MagicKey.discrete_fn(ptb_pos_tag_first2),
         MagicKey.discrete_fn(ptb_pos_tag_last2)]

    def __init__(self):
        desc = self.__doc__.strip()
        super(SingleEduSubgroup_Ptb, self).__init__(desc, self._features)


class SingleEduKeys(MergedKeyGroup):
    """
    single EDU features
    """
    def __init__(self, inputs):
        groups = [SingleEduSubgroup_Meta(),
                  SingleEduSubgroup_Text(),
                  SingleEduSubgroup_Ptb()]
        #if inputs.debug:
        #    groups.append(SingleEduSubgroup_Debug())
        desc = self.__doc__.strip()
        super(SingleEduKeys, self).__init__(desc, groups)

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


class PairSubGroup_Core(PairSubgroup):
    "core features"

    def __init__(self):
        desc = self.__doc__.strip()
        keys =\
            [MagicKey.meta_fn(feat_grouping)]
        super(PairSubGroup_Core, self).__init__(desc, keys)


class PairSubgroup_Gap(PairSubgroup):
    """
    Features related to the combined surrounding context of the
    two EDUs
    """

    def __init__(self):
        desc = "the gap between EDUs"
        keys =\
            [MagicKey.continuous_fn(num_edus_between),
             MagicKey.discrete_fn(same_paragraph),
             MagicKey.discrete_fn(same_bad_sentence),
             MagicKey.discrete_fn(same_ptb_sentence)]
        super(PairSubgroup_Gap, self).__init__(desc, keys)


# largely c/c'ed from educe.stac.learning.features
class PairSubgroup_Tuple(PairSubgroup):
    "artificial tuple features"

    def __init__(self, inputs, sf_cache):
        self.corpus = inputs.corpus
        self.sf_cache = sf_cache
        desc = self.__doc__.strip()
        keys =\
            [MagicKey.discrete_fn(word_first_pairs),
             MagicKey.discrete_fn(word_last_pairs),
             MagicKey.discrete_fn(bigram_first_pairs),
             MagicKey.discrete_fn(bigram_last_pairs),
             MagicKey.discrete_fn(ptb_pos_tag_first_pairs)]
        super(PairSubgroup_Tuple, self).__init__(desc, keys)

    def fill(self, current, edu1, edu2, target=None):
        vec = self if target is None else target
        for key in self.keys:
            vec[key.name] = key.function(current, self.sf_cache, edu1, edu2)


class PairSubgroup_Basket(PairSubgroup):
    """
    Sparse features
    """
    _features =\
        [MagicKey.basket_fn(ptb_pos_tags_in_first)]

    def __init__(self):
        desc = self.__doc__.strip()
        super(PairSubgroup_Basket, self).__init__(desc, self._features)


class PairKeys(MergedKeyGroup):
    """
    pair features

    sf_cache should only be None if you're just using this
    to generate help text
    """
    def __init__(self, inputs, sf_cache=None):
        """
        """
        self.sf_cache = sf_cache
        groups = [PairSubGroup_Core(),
                  PairSubgroup_Gap(),
                  PairSubgroup_Tuple(inputs, sf_cache),
                  PairSubgroup_Basket()]
        #if inputs.debug:
        #    groups.append(PairSubgroup_Debug())

        if sf_cache is None:
            self.edu1 = SingleEduKeys(inputs)
            self.edu2 = SingleEduKeys(inputs)
        else:
            self.edu1 = None  # will be filled out later
            self.edu2 = None  # from the feature cache

        desc = "pair features"
        super(PairKeys, self).__init__(desc, groups)

    def csv_headers(self, htype=False):
        if htype in [HeaderType.OLD_CSV, HeaderType.NAME]:
            return super(PairKeys, self).csv_headers(htype) +\
                    [h + "_EDU1" for h in self.edu1.csv_headers(htype)] +\
                    [h + "_EDU2" for h in self.edu2.csv_headers(htype)]
        else:
            return super(PairKeys, self).csv_headers(htype) +\
                    self.edu1.csv_headers(htype) +\
                    self.edu2.csv_headers(htype)


    def csv_values(self):
        return super(PairKeys, self).csv_values() +\
            self.edu1.csv_values() +\
            self.edu2.csv_values()

    def help_text(self):
        lines = [super(PairKeys, self).help_text(),
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


def simplify_deptree(dtree):
    """
    Boil a dependency tree down into a dictionary from (edu, edu) to rel
    """
    relations = {}
    # recursive inner function
    def _simplify_deptree(tree):
        src = treenode(tree)
        for kid in tree:
            tgt = treenode(kid)
            relations[(src.edu, tgt.edu)] = tgt.rel
            _simplify_deptree(kid)
    #
    _simplify_deptree(dtree)
    return relations


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
    ptb_trees = [t for t in doc_ptb_trees
                 if t.text_span().overlaps(edu.text_span())]
    all_tokens = itertools.chain.from_iterable(t.leaves() for t in ptb_trees)
    ptb_tokens = [tok for tok in all_tokens
                  if tok.text_span().overlaps(edu.text_span())]
    return ptb_trees, ptb_tokens


def preprocess(inputs, k):
    """
    Pre-process and bundle up a representation of the current document
    """
    rtree = SimpleRSTTree.from_rst_tree(inputs.corpus[k])
    dtree = deptree.relaxed_nuclearity_to_deptree(rtree)
    surrounders = {edu: _surrounding_text(edu)
                   for edu in rtree.leaves()}
    doc_ptb_trees = r_ptb.parse_trees(inputs.corpus, k, inputs.ptb)
    ptb_trees = {}
    ptb_tokens = {}
    for edu in rtree.leaves():
        ptb_trees[edu], ptb_tokens[edu] = _ptb_stuff(doc_ptb_trees, edu)

    return DocumentPlus(key=k,
                        rsttree=rtree,
                        deptree=dtree,
                        ptb_trees=ptb_trees,
                        ptb_tokens=ptb_tokens,
                        surrounders=surrounders)


def extract_pair_features(inputs, live=False):
    """
    Return a pair of dictionaries, one for attachments
    and one for relations
    """

    for k in inputs.corpus:
        current = preprocess(inputs, k)
        edus = current.rsttree.leaves()
        # reduced dependency graph as dictionary (edu to [edu])
        relations = simplify_deptree(current.deptree) if not live else {}

        # single edu features
        sf_cache = {}
        for edu in edus:
            sf_cache[edu] = SingleEduKeys(inputs)
            sf_cache[edu].fill(current, edu)

        for epair in itertools.product(edus, edus):
            edu1, edu2 = epair
            if edu1 == edu2:
                continue
            vec = PairKeys(inputs, sf_cache=sf_cache)
            vec.fill(current, edu1, edu2)

            if live:
                yield vec, vec
            else:
                pairs_vec = ClassKeyGroup(vec)
                rels_vec = ClassKeyGroup(vec)
                rels_vec.set_class(relations[epair] if epair in relations
                                   else 'UNRELATED')
                pairs_vec.set_class(epair in relations)

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
