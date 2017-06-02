"""
Basics for feature extraction
"""

from __future__ import print_function

from functools import wraps

import numpy as np

from educe.ptb.annotation import syntactic_node_seq
from educe.ptb.head_finder import find_edu_head


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
        return " ".join(wrapped(x) for x in things[:2]) if things else None
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
        return " ".join(wrapped(x) for x in things[-2:]) if things else None
    return inner


# tree utils
def lowest_common_parent(treepositions):
    """Find tree position of the lowest common parent of a list of nodes.

    Parameters
    ----------
    treepositions : :obj:`list` of tree positions
        see nltk.tree.Tree.treepositions()

    Returns
    -------
    tpos_parent : tree position
        Tree position of the lowest common parent to all the given tree
        positions.
    """
    if not treepositions:
        return None

    leftmost_tpos = treepositions[0]
    rightmost_tpos = treepositions[-1]

    for i, lmost_idx in enumerate(leftmost_tpos):
        if ((i == len(rightmost_tpos) or
             lmost_idx != rightmost_tpos[i])):
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

    def __init__(self, token_filter=None, word2clust=None):
        """
        Parameters
        ----------
        token_filter: function from Token to boolean, optional
            Function that returns True if a token should be kept; if
            None is provided, all tokens are kept.
        word2clust: TODO
            TODO
        """
        self.token_filter = token_filter
        self.word2clust = word2clust

    def preprocess(self, doc, strict=False):
        """Preprocess a document and output basic features for each EDU.

        Parameters
        ----------
        doc: DocumentPlus
            Document to be processed.

        Returns
        -------
        edu_infos: list of dict of features
            List of basic features for each EDU ; each feature is a
            couple (basic_feat_name, basic_feat_val).
        para_infos: list of dict of features
            List of basic features for each paragraph ; each feature is
            a couple (basic_feat_name, basic_feat_val).

        TODO
        ----
        * [ ] explicitly impute missing values, e.g. for idxes_in_*
        """
        token_filter = self.token_filter
        word2clust = self.word2clust

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

        # paragraphs
        if paragraphs is None:
            para_infos = None
        else:
            para_infos = []

            # special case for the left padding paragraph
            pfeats = dict()
            pfeats['tokens'] = [tokens[0]]  # left padding token
            pfeats['syn_nodes'] = None
            para_infos.append(pfeats)

            # regular paragraphs
            for para_idx, para in enumerate(paragraphs[1:], start=1):
                pfeats = dict()
                para_beg = para.sentences[0].span.char_start
                para_end = para.sentences[-1].span.char_end
                trees_beg = doc.trees_beg
                trees_end = doc.trees_end
                toks_beg = doc.toks_beg
                toks_end = doc.toks_end

                # * token characterization of the paragraph
                encltoks_idc = np.where(
                    np.logical_and(toks_beg >= para_beg,
                                   toks_end <= para_end)
                )[0]
                encltoks = [tokens[i] for i in encltoks_idc]
                pfeats['tokens'] = encltoks

                # * syntactic characterization of the paragraph
                # find the syntactic trees that span this paragraph
                enclosed_idc = np.intersect1d(
                    np.where(trees_beg >= para_beg),
                    np.where(trees_end <= para_end))
                overlapd_idc = np.intersect1d(
                    np.where(trees_beg < para_end),
                    np.where(trees_end > para_beg))
                if np.array_equal(enclosed_idc, overlapd_idc):
                    # sentence seg and paragraph seg are compatible
                    syn_nodes = [trees[tree_idx]
                                 for tree_idx in overlapd_idc]
                else:
                    # mismatch between the sentence segmentation from the
                    # PTB and paragraph segmentation from the RST-WSJ
                    strad_idc = np.setdiff1d(overlapd_idc, enclosed_idc)
                    syn_nodes = []
                    for tree_idx in overlapd_idc:
                        syn_tree = trees[tree_idx]
                        if tree_idx not in strad_idc:
                            syn_nodes.append(syn_tree)
                            continue
                        # find the list of tokens that overlap this
                        # paragraph, and belong to this straddling
                        # tree
                        tree_beg = trees_beg[tree_idx]
                        tree_end = trees_end[tree_idx]
                        # here, reduce(np.logical_and(...)) was 2x
                        # faster than np.logical_and.reduce(...)
                        overtoks_idc = np.where(
                            reduce(np.logical_and,
                                   (toks_beg < para_end,
                                    toks_end > para_beg,
                                    toks_beg >= tree_beg,
                                    toks_end <= tree_end))
                        )[0]
                        overtoks = [tokens[i] for i in overtoks_idc]
                        syn_node_seq = syntactic_node_seq(
                            syn_tree, overtoks)
                        syn_nodes.extend(syn_node_seq)
                # add basic feature
                pfeats['syn_nodes'] = syn_nodes
                # store
                para_infos.append(pfeats)
        # EDUs
        edu_infos = []
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
        # EXPERIMENTAL: Brown clusters
        res['brown_clusters'] = []
        # end Brown clusters
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
        edu_infos.append(res)

        # regular EDUs
        for edu_idx, edu in enumerate(edus[1:], start=1):
            res = dict()
            res['edu'] = edu

            # raw words (temporary)
            res['raw_words'] = raw_words[edu_idx]

            # tokens
            if tokens is not None:
                tok_idcs = edu2tokens[edu_idx]
                toks = [tokens[tok_idx] for tok_idx in tok_idcs]
                # special case: no tokens
                if strict and not toks:
                    emsg = 'No token for EDU'
                    print(list(enumerate(tokens)))
                    print(tok_idcs)
                    print(edu.text())
                    raise ValueError(emsg)
                # filter tokens if relevant
                if token_filter is not None:
                    toks = [tt for tt in toks if token_filter(tt)]
                # store information
                res['tokens'] = toks
                res['tags'] = [tok.tag for tok in toks]
                res['words'] = [tok.word for tok in toks]
                # EXPERIMENTAL: Brown clusters
                if word2clust is not None:
                    res['brown_clusters'] = [word2clust[w]
                                             for w in res['words']
                                             if w in word2clust]
                # end Brown clusters

            # doc structure

            # position of sentence containing EDU in doc
            # aka sentence_id
            sent_idx = edu2sent[edu_idx]
            res['sent_idx'] = sent_idx
            res['sent_rev_idx'] = (len(trees) - 1 - sent_idx
                                   if sent_idx is not None
                                   else None)  # NEW
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
                                   if (paragraphs is not None and
                                       para_idx is not None)
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
                res['tkd_tree_idx'] = tree_idx
                if tree_idx is not None:
                    # head node of the EDU (for DS-LST features)
                    ptree = trees[tree_idx]
                    pheads = lex_heads[tree_idx]
                    # tree positions (in the syn tree) of the words of
                    # the EDU
                    tpos_leaves_edu = [x for x
                                       in ptree.treepositions('leaves')
                                       if ptree[x].overlaps(edu)]
                    tpos_words = set(tpos_leaves_edu)
                    res['tpos_words'] = tpos_words
                    edu_head = find_edu_head(ptree, pheads, tpos_words)
                    res['edu_head'] = edu_head

            edu_infos.append(res)

        return edu_infos, para_infos
