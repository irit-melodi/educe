"""
Basics for feature extraction
"""

from __future__ import print_function

from functools import wraps
import itertools as it

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


def relative_indices(group_indices, reverse=False, valna=None):
    """Generate a list of relative indices inside each group.
    Missing (None) values are handled specifically: each missing
    value is mapped to `valna`.

    Parameters
    ----------
    reverse: boolean, optional
        If True, compute indices relative to the end of each group.
    valna: int or None, optional
        Relative index for missing values.
    """
    # TODO rewrite using np.ediff1d, np.where and the like

    groupby = it.groupby

    if reverse:
        group_indices = list(group_indices)
        group_indices.reverse()

    result = []
    for group_idx, dup_values in groupby(group_indices):
        if group_idx is None:
            rel_indices = (valna for dup_value in dup_values)
        else:
            rel_indices = (rel_idx for rel_idx, dv in enumerate(dup_values))
        result.extend(rel_indices)

    if reverse:
        result.reverse()

    return result


class DocumentPlusPreprocessor(object):
    """Preprocessor for feature extraction on a DocumentPlus

    This pre-processor currently does not explicitly impute missing values,
    but it probably should eventually.
    As the ultimate output is features in a sparse format, the current
    strategy amounts to imputing missing values as 0, which is most
    certainly not optimal.
    """

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

        TODO explicitly impute missing values, e.g. for (rev_)idxes_in_*
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
