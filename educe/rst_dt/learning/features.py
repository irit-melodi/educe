"""
Feature extraction library functions for RST_DT corpus
"""

from collections import Counter
from functools import wraps
import re

from educe.learning.csv import tune_for_csv
from educe.learning.keys import MagicKey
from educe.learning.util import tuple_feature, underscore
from .base import (SingleEduSubgroup, PairSubgroup,
                   BaseSingleEduKeys, BasePairKeys,
                   on_first_unigram, on_last_unigram,
                   on_first_bigram, on_last_bigram,
                   edu_feature, edu_pair_feature,
                   feat_id, feat_start, feat_end,
                   feat_grouping,
                   get_sentence, get_paragraph)


# ---------------------------------------------------------------------
# single EDUs
# ---------------------------------------------------------------------

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
        return wrapped(tokens) if tokens is not None else None
    return inner


# ---------------------------------------------------------------------
# single EDU features
# ---------------------------------------------------------------------

# not-PTB-tokens

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


# PTB tokens

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


def same_paragraph(current, edu1, edu2):
    "if in the same paragraph"
    para1 = get_paragraph(current, edu1)
    para2 = get_paragraph(current, edu2)
    return para1 is not None and para2 is not None and\
        para1 == para2


def same_bad_sentence(current, edu1, edu2):
    "if in the same sentence (bad segmentation)"
    sent1 = get_sentence(current, edu1)
    sent2 = get_sentence(current, edu2)
    return sent1 is not None and sent2 is not None and\
        sent1 == sent2


def same_ptb_sentence(current, edu1, edu2):
    "if in the same sentence (ptb segmentation)"
    sents1 = current.ptb_trees[edu1]
    sents2 = current.ptb_trees[edu2]
    if sents1 is None or sents2 is None:
        return False
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
    return cache[edu]["ptb_pos_tag_first"]


def ptb_pos_tags_in_first(current, edu1, _):
    "demonstrator for use of basket features"
    tokens = current.ptb_tokens[edu1]
    return Counter(t.tag for t in tokens) if tokens is not None else None


# ---------------------------------------------------------------------
# single EDU key groups
# ---------------------------------------------------------------------

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



# ---------------------------------------------------------------------
# EDU pairs
# ---------------------------------------------------------------------

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


# export groups
class SingleEduKeys(BaseSingleEduKeys):
    """Single EDU features"""

    def __init__(self, inputs):
        groups = [
            SingleEduSubgroup_Meta(),
            SingleEduSubgroup_Text(),
            SingleEduSubgroup_Ptb()
        ]
        #if inputs.debug:
        #    groups.append(SingleEduSubgroup_Debug())
        super(SingleEduKeys, self).__init__(inputs, groups)


class PairKeys(BasePairKeys):
    """Features on a pair of EDUs"""

    def __init__(self, inputs, sf_cache=None):
        groups = [
            PairSubGroup_Core(),
            PairSubgroup_Gap(),
            PairSubgroup_Tuple(inputs, sf_cache),
            PairSubgroup_Basket()
        ]
        #if inputs.debug:
        #    groups.append(PairSubgroup_Debug())
        super(PairKeys, self).__init__(inputs, groups, sf_cache)

    def init_single_features(self, inputs):
        """Init features defined on single EDUs"""
        return SingleEduKeys(inputs)
