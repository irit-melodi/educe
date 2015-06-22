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

    def preprocess(self, doc, strict=False):
        """Preprocess a document and output basic features for each EDU.

        Return a dict(EDU, (dict(basic_feat_name, basic_feat_val)))

        TODO explicitly impute missing values, e.g. for (rev_)idxes_in_*
        """
        token_filter = self.token_filter

        edus = doc.edus
        raw_words = doc.raw_words  # TEMPORARY
        tokens = doc.tkd_tokens
        trees = doc.tkd_trees
        paragraphs = doc.paragraphs  # NEW
        # mappings from EDU to other annotations
        edu2raw_sent = doc.edu2raw_sent
        edu2para = doc.edu2para
        edu2sent = doc.edu2sent
        edu2tokens = doc.edu2tokens
        lex_heads = doc.lex_heads  # EXPERIMENTAL
        
        # pre-compute relative indices (in sent, para) in one iteration
        # NB: moved to document_plus itself
        idxes_in_sent = doc.edu2idx_in_sent
        rev_idxes_in_sent = doc.edu2rev_idx_in_sent

        idxes_in_para = doc.edu2idx_in_para
        rev_idxes_in_para = doc.edu2rev_idx_in_para

        result = dict()

        # special case: left padding EDU
        edu = edus[0]
        res = dict()
        res['edu'] = edu
        # raw words (temporary)
        res['raw_words'] = []
        # tokens
        res['tokens'] = []  # TODO: __START__ / __START__ ?
        res['tags'] = []  # TODO: __START__ ?
        res['words'] = []  # TODO: __START__ ?
        res['tok_beg'] = 0  # EXPERIMENTAL
        res['tok_end'] = 0  # EXPERIMENTAL
        # sentence
        res['edu_idx_in_sent'] = idxes_in_sent[0]
        res['edu_rev_idx_in_sent'] = rev_idxes_in_sent[0]
        res['sent_idx'] = 0
        res['sent_rev_idx'] = len(trees) - 1  # NEW
        # para
        res['edu_rev_idx_in_para'] = rev_idxes_in_para[0]
        # aka paragraphID
        res['para_idx'] = 0
        res['para_rev_idx'] = (len(paragraphs) - 1 if paragraphs is not None
                               else None)  # NEW
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
            if tokens is not None:
                tok_idcs = edu2tokens[edu_idx]
                toks = [tokens[tok_idx] for tok_idx in tok_idcs]
                if toks:
                    filtd_toks = [tt for tt in toks if token_filter(tt)]
                    res['tokens'] = filtd_toks
                    res['tags'] = [tok.tag for tok in filtd_toks]
                    res['words'] = [tok.word for tok in filtd_toks]
                else:
                    if strict:
                        emsg = 'No token for EDU'
                        print(list(enumerate(tokens)))
                        print(tok_idcs)
                        print(edu.text())
                        raise ValueError(emsg)
                    # maybe I should fill res with empty lists? unclear

            # doc structure

            # position of sentence containing EDU in doc
            # aka sentence_id
            sent_idx = edu2sent[edu_idx]
            res['sent_idx'] = sent_idx
            res['sent_rev_idx'] = len(trees) - 1 - sent_idx  # NEW
            # position of EDU in sentence
            # aka num_edus_from_sent_start aka offset
            res['edu_idx_in_sent'] = idxes_in_sent[edu_idx]
            # aka num_edus_to_sent_end aka revOffset
            res['edu_rev_idx_in_sent'] = rev_idxes_in_sent[edu_idx]

            # position of paragraph containing EDU in doc
            # aka paragraphID
            para_idx = edu2para[edu_idx]
            res['para_idx'] = para_idx
            res['para_rev_idx'] = (len(paragraphs) - 1 - para_idx
                                   if paragraphs is not None
                                   else None)  # NEW
            # position of raw sentence
            res['raw_sent_idx'] = edu2raw_sent[edu_idx]

            # position of EDU in paragraph
            # aka num_edus_to_para_end aka revSentenceID (?!)
            # TODO: check for the 10th time if this is a bug in Li et al.'s
            # parser
            res['edu_rev_idx_in_para'] = rev_idxes_in_para[edu_idx]

            # syntax
            if len(trees) > 1:
                tree_idx = edu2sent[edu_idx]
                if tree_idx is not None:
                    tree = trees[tree_idx]
                    res['ptree'] = tree
                    res['pheads'] = lex_heads[tree_idx]
            result[edu] = res

        return result
