"""
Basics for feature extraction
"""

from __future__ import print_function

from functools import wraps
import itertools as it

from educe.learning.keys import KeyGroup, MergedKeyGroup, HeaderType


class FeatureExtractionException(Exception):
    """
    Exceptions related to RST trees not looking like we would
    expect them to
    """
    def __init__(self, msg):
        super(FeatureExtractionException, self).__init__(msg)


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
    def __init__(self, feature_groups):
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

    def __init__(self, pair_feature_groups, sf_cache=None):
        self.sf_cache = sf_cache

        if sf_cache is None:
            self.edu1 = self.init_single_features()
            self.edu2 = self.init_single_features()
        else:
            self.edu1 = None  # will be filled out later
            self.edu2 = None  # from the feature cache

        desc = "pair features"
        super(BasePairKeys, self).__init__(desc, pair_feature_groups)

    def init_single_features(self):
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


# tree utils
def lowest_common_parent(treepositions):
    """Find tree position of the lowest common parent of a list of nodes.

    treepositions is a list of tree positions
    see nltk.tree.Tree.treepositions()
    """
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


def relative_indices(group_indices, reverse=False):
    """Generate a list of relative indices inside each group.

    Each None value triggers a new group.
    """
    # TODO rewrite using np.ediff1d, np.where and the like

    groupby = it.groupby

    if reverse:
        group_indices = list(group_indices)
        group_indices.reverse()

    result = []
    for group_idx, dup_values in groupby(group_indices):
        if group_idx is None:
            rel_indices = (0 for dup_value in dup_values)
        else:
            rel_indices = (rel_idx for rel_idx, dv in enumerate(dup_values))
        result.extend(rel_indices)

    if reverse:
        result.reverse()

    return result


class DocumentPlusPreprocessor(object):
    """Preprocessor for feature extraction on a DocumentPlus"""

    def __init__(self, token_filter=None):
        """
        token_filter is a function that returns True if a token should be
        kept; if None is provided, all tokens are kept
        """
        if token_filter is None:
            token_filter = lambda token: True
        self.token_filter = token_filter

    def preprocess(self, doc):
        """Preprocess a document and output basic features for each EDU.

        Return a dict(EDU, (dict(basic_feat_name, basic_feat_val)))
        """
        token_filter = self.token_filter

        edus = doc.edus
        edu2sent = doc.edu2sent
        edu2para = doc.edu2para
        edu2raw_sent = doc.edu2raw_sent
        raw_words = doc.raw_words  # TEMPORARY
        ptb_tokens = doc.ptb_tokens
        ptb_trees = doc.ptb_trees
        # pre-compute relative indices (in sent, para) in one iteration
        idxes_in_sent = relative_indices(edu2sent)
        rev_idxes_in_sent = relative_indices(edu2sent, reverse=True)
        idxes_in_para = relative_indices(edu2para)
        rev_idxes_in_para = relative_indices(edu2para, reverse=True)

        result = dict()

        # special case: left padding EDU
        edu = edus[0]
        res = dict()
        res['edu'] = edu
        # raw words (temporary)
        res['raw_words'] = []
        # tokens
        res['tokens'] = []
        res['tags'] = []
        res['words'] = []
        # sentence
        res['edu_idx_in_sent'] = idxes_in_sent[0]
        res['edu_rev_idx_in_sent'] = rev_idxes_in_sent[0]
        res['sent_idx'] = 0
        # para
        res['edu_rev_idx_in_para'] = rev_idxes_in_para[0]
        # aka paragraphID
        res['para_idx'] = 0
        # raw sent
        res['raw_sent_idx'] = edu2raw_sent[0]
        result[edu] = res

        # regular EDUs
        for edu_idx, edu in enumerate(edus[1:], start=1):
            res = dict()
            res['edu'] = edu

            # raw words (temporary)
            res['raw_words'] = raw_words[edu]

            # tokens
            if ptb_tokens is not None:
                tokens = ptb_tokens[edu]
                if tokens is not None:
                    tokens = [tt for tt in tokens if token_filter(tt)]
                    res['tokens'] = tokens
                    res['tags'] = [tok.tag for tok in tokens]
                    res['words'] = [tok.word for tok in tokens]

            # doc structure

            # position of sentence containing EDU in doc
            # aka sentence_id
            res['sent_idx'] = edu2sent[edu_idx]

            # position of EDU in sentence
            # aka num_edus_from_sent_start aka offset
            res['edu_idx_in_sent'] = idxes_in_sent[edu_idx]
            # aka num_edus_to_sent_end aka revOffset
            res['edu_rev_idx_in_sent'] = rev_idxes_in_sent[edu_idx]

            # position of paragraph containing EDU in doc
            # aka paragraphID
            res['para_idx'] = edu2para[edu_idx]
            # position of raw sentence
            res['raw_sent_idx'] = edu2raw_sent[edu_idx]

            # position of EDU in paragraph
            # aka num_edus_to_para_end aka revSentenceID (?!)
            # TODO: check for the 10th time if this is a bug in Li et al.'s
            # parser
            res['edu_rev_idx_in_para'] = rev_idxes_in_para[edu_idx]

            # syntax
            if ptb_trees is not None:
                ptrees = ptb_trees[edu]
                if ptrees is not None:
                    res['ptrees'] = ptrees

            result[edu] = res

        return result
