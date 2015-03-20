"""Partial re-implementation of the feature extraction procedure used in
[li2014text]_ for discourse dependency parsing on the RST-DT corpus.

.. [li2014text] Li, S., Wang, L., Cao, Z., & Li, W. (2014).
Text-level discourse dependency parsing.
In Proceedings of the 52nd Annual Meeting of the Association for
Computational Linguistics (Vol. 1, pp. 25-35).
http://www.aclweb.org/anthology/P/P14/P14-1003.pdf
"""

import re

from educe.internalutil import treenode
from educe.learning.keys import Substance
from .base import lowest_common_parent, DocumentPlusPreprocessor


# ---------------------------------------------------------------------
# preprocess EDUs
# ---------------------------------------------------------------------

# filter tags and tokens as in Li et al.'s parser
TT_PATTERN = r'.*[a-zA-Z_0-9].*'
TT_FILTER = re.compile(TT_PATTERN)


def token_filter_li2014(token):
    """Token filter defined in Li et al.'s parser.

    This filter only applies to tagged tokens.
    """
    return (TT_FILTER.match(token.word) is not None and
            TT_FILTER.match(token.tag) is not None)


def build_doc_preprocessor():
    """Build the preprocessor for feature extraction in each EDU of doc"""
    # TODO re-do in a better, more modular way
    return DocumentPlusPreprocessor(token_filter_li2014).preprocess


# ---------------------------------------------------------------------
# single EDU features
# ---------------------------------------------------------------------

SINGLE_WORD = [
    ('ptb_word_first', Substance.DISCRETE),
    ('ptb_word_last', Substance.DISCRETE),
    ('ptb_word_first2', Substance.DISCRETE),
    ('ptb_word_last2', Substance.DISCRETE)
]


def extract_single_word(edu_info):
    """word features for the EDU"""
    try:
        words = edu_info['words']
    except KeyError:
        return

    if words:
        yield ('ptb_word_first', words[0])
        yield ('ptb_word_last', words[-1])

    if len(words) > 1:
        yield ('ptb_word_first2', (words[0], words[1]))
        yield ('ptb_word_last2', (words[-2], words[-1]))


SINGLE_POS = [
    ('ptb_pos_tag_first', Substance.DISCRETE),
    ('ptb_pos_tag_last', Substance.DISCRETE),
    ('POS', Substance.BASKET)
]


def extract_single_pos(edu_info):
    """POS features for the EDU"""
    try:
        tags = edu_info['tags']
    except KeyError:
        return

    if tags:
        yield ('ptb_pos_tag_first', tags[0])
        yield ('ptb_pos_tag_last', tags[-1])
        for tag in tags:
            yield ('POS', tag)


SINGLE_LENGTH = [
    ('num_tokens', Substance.DISCRETE),
    ('num_tokens_div5', Substance.DISCRETE)
]


def extract_single_length(edu_info):
    """Sentence features for the EDU"""
    try:
        words = edu_info['words']
    except KeyError:
        return

    yield ('num_tokens', str(len(words)))
    yield ('num_tokens_div5', str(len(words) / 5))


# features on document structure

SINGLE_SENTENCE = [
    # offset
    ('num_edus_from_sent_start', Substance.DISCRETE),
    # revOffset
    ('num_edus_to_sent_end', Substance.DISCRETE),
    # sentenceID
    ('sentence_id', Substance.DISCRETE),
    # revSentenceID
    ('num_edus_to_para_end', Substance.DISCRETE)
]


def extract_single_sentence(edu_info):
    """Sentence features for the EDU"""
    try:
        offset = edu_info['edu_idx_in_sent']
        if offset is not None:
            yield ('num_edus_from_sent_start', str(offset))

        rev_offset = edu_info['edu_rev_idx_in_sent']
        if rev_offset is not None:
            yield ('num_edus_to_sent_end', str(rev_offset))

        # position of sentence in doc
        sent_id = edu_info['sent_idx']
        if sent_id is not None:
            yield ('sentence_id', str(sent_id))
    except KeyError:
        pass

    try:
        yield ('num_edus_to_para_end', str(edu_info['edu_rev_idx_in_para']))
    except KeyError:
        pass


SINGLE_PARA = [
    ('paragraph_id', Substance.DISCRETE),
    ('paragraph_id_div5', Substance.DISCRETE)
]


def extract_single_para(edu_info):
    """paragraph features for the EDU"""
    try:
        para_idx = edu_info['para_idx']
    except KeyError:
        pass
    else:
        if para_idx is not None:
            yield ('paragraph_id', str(para_idx))
            yield ('paragraph_id_div5', str(para_idx / 5))


# features on syntax
# helper
def get_syntactic_labels(edu_info):
    "Syntactic labels for this EDU"
    result = []

    try:
        ptree = edu_info['ptree']
    except KeyError:
        return None

    edu = edu_info['edu']

    # get the tree position of the leaves of the syntactic tree that are in
    # the EDU
    tpos_leaves_edu = [tpos_leaf
                       for tpos_leaf in ptree.treepositions('leaves')
                       if ptree[tpos_leaf].overlaps(edu)]
    # for each span of syntactic leaves in this EDU
    tpos_parent = lowest_common_parent(tpos_leaves_edu)
    # for each leaf between leftmost and rightmost, add its ancestors
    # up to the lowest common parent
    for leaf in tpos_leaves_edu:
        for i in reversed(range(len(leaf))):
            tpos_node = leaf[:i]
            node = ptree[tpos_node]
            node_lbl = treenode(node)
            if tpos_node == tpos_parent:
                result.append('top_' + node_lbl)
                break
            else:
                result.append(node_lbl)
    return result


SINGLE_SYNTAX = [
    ('SYN', Substance.BASKET)
]


def extract_single_syntax(edu_info):
    """syntactic features for the EDU"""
    syn_labels = get_syntactic_labels(edu_info)
    if syn_labels is not None:
        for syn_label in syn_labels:
            yield ('SYN', syn_label)


# TODO: features on semantic similarity

def build_edu_feature_extractor():
    """Build the feature extractor for single EDUs"""
    feats = []
    funcs = []

    # word
    feats.extend(SINGLE_WORD)
    funcs.append(extract_single_word)
    # pos
    feats.extend(SINGLE_POS)
    funcs.append(extract_single_pos)
    # length
    feats.extend(SINGLE_LENGTH)
    funcs.append(extract_single_length)
    # para
    feats.extend(SINGLE_PARA)
    funcs.append(extract_single_para)
    # sent
    feats.extend(SINGLE_SENTENCE)
    funcs.append(extract_single_sentence)
    # syntax (disabled)
    # feats.extend(SINGLE_SYNTAX)
    # funcs.append(extract_single_syntax)

    def _extract_all(edu_info):
        """inner helper because I am lost at sea here"""
        # TODO do this in a cleaner manner
        for fct in funcs:
            for feat in fct(edu_info):
                yield feat

    # header
    header = feats
    # extractor
    feat_extractor = _extract_all
    # return header and extractor
    return header, feat_extractor


# ---------------------------------------------------------------------
# EDU pairs
# ---------------------------------------------------------------------

PAIR_WORD = [
    ('ptb_word_first_pairs', Substance.DISCRETE),
    ('ptb_word_last_pairs', Substance.DISCRETE),
    ('ptb_word_first2_pairs', Substance.DISCRETE),
    ('ptb_word_last2_pairs', Substance.DISCRETE),
]


def extract_pair_word(edu_info1, edu_info2):
    """word tuple features"""
    try:
        words1 = edu_info1['words']
        words2 = edu_info2['words']
    except KeyError:
        return

    # pairs of unigrams
    if words1 and words2:
        yield ('ptb_word_first_pairs', (words1[0], words2[0]))
        yield ('ptb_word_last_pairs', (words1[-1], words2[-1]))

    # pairs of bigrams
    if len(words1) > 1 and len(words2) > 1:
        yield ('ptb_word_first2_pairs', (tuple(words1[:2]),
                                         tuple(words2[:2])))
        yield ('ptb_word_last2_pairs', (tuple(words1[-2:]),
                                        tuple(words2[-2:])))


# pos
PAIR_POS = [
    ('ptb_pos_tag_first_pairs', Substance.DISCRETE),
]


def extract_pair_pos(edu_info1, edu_info2):
    """POS tuple features"""
    try:
        tags1 = edu_info1['tags']
        tags2 = edu_info2['tags']
    except KeyError:
        return

    if tags1 and tags2:
        yield ('ptb_pos_tag_first_pairs', (tags1[0], tags2[0]))


PAIR_LENGTH = [
    ('num_tokens_div5_pair', Substance.DISCRETE),
    ('num_tokens_diff_div5', Substance.DISCRETE)
]


def extract_pair_length(edu_info1, edu_info2):
    """Sentence tuple features"""
    try:
        words1 = edu_info1['words']
        words2 = edu_info2['words']
    except KeyError:
        return

    num_toks1 = len(words1)
    num_toks2 = len(words2)

    yield ('num_tokens_div5_pair', (num_toks1 / 5, num_toks2 / 5))
    yield ('num_tokens_diff_div5', str((num_toks1 - num_toks2) / 5))


PAIR_PARA = [
    ('first_paragraph', Substance.DISCRETE),
    ('num_paragraphs_between', Substance.DISCRETE),
    ('num_paragraphs_between_div3', Substance.DISCRETE)
]


def extract_pair_para(edu_info1, edu_info2):
    """Paragraph tuple features"""
    try:
        para_id1 = edu_info1['para_idx']
        para_id2 = edu_info2['para_idx']
    except KeyError:
        return
    if para_id1 is not None and para_id2 is not None:
        if para_id1 < para_id2:
            first_para = 'first'
        elif para_id1 > para_id2:
            first_para = 'second'
        else:
            first_para = 'same'
        yield ('first_paragraph', first_para)

        yield ('num_paragraphs_between', str(para_id1 - para_id2))
        yield ('num_paragraphs_between_div3', str((para_id1 - para_id2) / 3))


PAIR_SENT = [
    ('offset_diff', Substance.DISCRETE),
    ('rev_offset_diff', Substance.DISCRETE),
    ('offset_diff_div3', Substance.DISCRETE),
    ('rev_offset_diff_div3', Substance.DISCRETE),
    ('offset_pair', Substance.DISCRETE),
    ('rev_offset_pair', Substance.DISCRETE),
    ('offset_div3_pair', Substance.DISCRETE),
    ('rev_offset_div3_pair', Substance.DISCRETE),
    ('line_id_diff', Substance.DISCRETE),
    ('same_bad_sentence', Substance.DISCRETE),
    ('sentence_id_diff', Substance.DISCRETE),
    ('sentence_id_diff_div3', Substance.DISCRETE),
    ('rev_sentence_id_diff', Substance.DISCRETE),
    ('rev_sentence_id_diff_div3', Substance.DISCRETE)
]


def extract_pair_sent(edu_info1, edu_info2):
    """Sentence tuple features"""
    # offset features
    try:
        offset1 = edu_info1['edu_idx_in_sent']
        offset2 = edu_info2['edu_idx_in_sent']
    except KeyError:
        pass
    else:
        if offset1 is not None and offset2 is not None:
            yield ('offset_diff', str(offset1 - offset2))
            yield ('offset_diff_div3', str((offset1 - offset2) / 3))
            yield ('offset_pair', (offset1, offset2))
            yield ('offset_div3_pair', (offset1 / 3, offset2 / 3))

    # rev_offset features
    try:
        rev_offset1 = edu_info1['edu_rev_idx_in_sent']
        rev_offset2 = edu_info2['edu_rev_idx_in_sent']
    except KeyError:
        pass
    else:
        if rev_offset1 is not None and offset2 is not None:
            yield ('rev_offset_diff', str(rev_offset1 - rev_offset2))
            yield ('rev_offset_diff_div3', str((rev_offset1 - rev_offset2) / 3))
            yield ('rev_offset_pair', (rev_offset1, rev_offset2))
            yield ('rev_offset_div3_pair', (rev_offset1 / 3, rev_offset2 / 3))

    # lineID: distance of edu in EDUs from document start
    line_id1 = edu_info1['edu'].num - 1  # real EDU numbers are in [1..]
    line_id2 = edu_info2['edu'].num - 1
    yield ('line_id_diff', str(line_id1 - line_id2))

    # sentenceID
    sent_id1 = edu_info1['sent_idx']
    sent_id2 = edu_info2['sent_idx']
    if sent_id1 is not None and sent_id2 is not None:
        yield ('same_sentence',
               'same' if sent_id1 == sent_id2 else 'different')
        yield ('sentence_id_diff', str(sent_id1 - sent_id2))
        yield ('sentence_id_diff_div3', str((sent_id1 - sent_id2) / 3))

    # revSentenceID
    rev_sent_id1 = edu_info1['edu_rev_idx_in_para']
    rev_sent_id2 = edu_info2['edu_rev_idx_in_para']
    if rev_sent_id1 is not None and rev_sent_id2 is not None:
        yield ('rev_sentence_id_diff', str(rev_sent_id1 - rev_sent_id2))
        yield ('rev_sentence_id_diff_div3',
               str((rev_sent_id1 - rev_sent_id2) / 3))


def build_pair_feature_extractor():
    """Build the feature extractor for pairs of EDUs

    TODO: properly emit features on single EDUs ;
    they are already stored in sf_cache, but under (slightly) different
    names
    """
    feats = []
    funcs = []

    # feature type: 1
    feats.extend(PAIR_WORD)
    funcs.append(extract_pair_word)
    # 2
    feats.extend(PAIR_POS)
    funcs.append(extract_pair_pos)
    # 3
    feats.extend(PAIR_PARA)
    funcs.append(extract_pair_para)
    feats.extend(PAIR_SENT)
    funcs.append(extract_pair_sent)
    # 4
    feats.extend(PAIR_LENGTH)
    funcs.append(extract_pair_length)
    # 5
    # feats.extend(PAIR_SYNTAX)  # NotImplemented
    # funcs.append(extract_pair_syntax)
    # 6
    # feats.extend(PAIR_SEMANTICS)  # NotImplemented
    # funcs.append(extract_pair_semantics)

    def _extract_all(edu_info1, edu_info2):
        """inner helper because I am lost at sea here, again"""
        # TODO do this in a cleaner manner
        for fct in funcs:
            for feat in fct(edu_info1, edu_info2):
                yield feat

    # header
    header = feats
    # extractor
    feat_extractor = _extract_all
    # return header and extractor
    return header, feat_extractor
