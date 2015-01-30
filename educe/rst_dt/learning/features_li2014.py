"""Partial re-implementation of the feature extraction procedure used in
[li2014text]_ for discourse dependency parsing on the RST-DT corpus.

.. [li2014text] Li, S., Wang, L., Cao, Z., & Li, W. (2014).
Text-level discourse dependency parsing.
In Proceedings of the 52nd Annual Meeting of the Association for
Computational Linguistics (Vol. 1, pp. 25-35).
http://www.aclweb.org/anthology/P/P14/P14-1003.pdf
"""

import re
from collections import Counter

from educe.internalutil import treenode
from educe.learning.keys import Substance
from .base import (feat_grouping, get_sentence, get_paragraph,
                   lowest_common_parent)


# ---------------------------------------------------------------------
# single EDU features
# ---------------------------------------------------------------------

# filter tags and tokens as in Li et al.'s parser
TT_PATTERN = r'.*[a-zA-Z_0-9].*'
TT_FILTER = re.compile(TT_PATTERN)


# ---------------------------------------------------------------------
# single EDU features: concrete features
# ---------------------------------------------------------------------

def num_edus_to_doc_end(current, edu):
    "distance of edu in EDUs to document end"
    edus = current.edus
    result = list(reversed(edus)).index(edu)
    return result


# ---------------------------------------------------------------------
# single EDU key groups
# ---------------------------------------------------------------------

# meta features
# in Orange terms, they all implicitly have Purpose.META
_single_meta = [
    ('id', Substance.STRING),
    ('start', Substance.STRING),
    ('end', Substance.STRING)
]


def extract_single_meta(doc, edu):
    """Basic EDU-identification features"""
    yield ('id', edu.identifier())
    yield ('start', edu.text_span().char_start)
    yield ('end', edu.text_span().char_end)


# concrete features
# features on tokens
_single_word = [
    ('ptb_word_first', Substance.DISCRETE),
    ('ptb_word_last', Substance.DISCRETE),
    ('ptb_word_first2', Substance.DISCRETE),
    ('ptb_word_last2', Substance.DISCRETE)
]


def extract_single_word(doc, edu):
    """word features for the EDU"""
    tokens = doc.ptb_tokens[edu]
    if tokens is not None:
        # filter tags and tokens as in Li et al.'s parser
        tokens = [tt for tt in tokens
                  if (TT_FILTER.match(tt.word) is not None and
                      TT_FILTER.match(tt.tag) is not None)]

        yield ('ptb_word_first', tokens[0].word)
        yield ('ptb_word_last', tokens[-1].word)
        try:
            yield ('ptb_word_first2', (tokens[0].word, tokens[1].word))
            yield ('ptb_word_last2', (tokens[-2].word, tokens[-1].word))
        except IndexError:
            # if there is only one token, just pass
            pass


_single_pos = [
    ('ptb_pos_tag_first', Substance.DISCRETE),
    ('ptb_pos_tag_last', Substance.DISCRETE),
    ('POS', Substance.BASKET)
]


def extract_single_pos(doc, edu):
    """POS features for the EDU"""
    tokens = doc.ptb_tokens[edu]
    if tokens is not None:
        # filter tags and tokens as in Li et al.'s parser
        tokens = [tt for tt in tokens
                  if (TT_FILTER.match(tt.word) is not None and
                      TT_FILTER.match(tt.tag) is not None)]

        yield ('ptb_pos_tag_first', tokens[0].tag)
        yield ('ptb_pos_tag_last', tokens[-1].tag)
        yield ('POS', [t.tag for t in tokens])


_single_length = [
    ('num_tokens', Substance.CONTINUOUS),
    ('num_tokens_div5', Substance.CONTINUOUS)
]


def extract_single_length(doc, edu):
    """Sentence features for the EDU"""
    tokens = doc.ptb_tokens[edu]
    if tokens is not None:
        # filter tags and tokens as in Li et al.'s parser
        tokens = [tt for tt in tokens
                  if (TT_FILTER.match(tt.word) is not None and
                      TT_FILTER.match(tt.tag) is not None)]

        yield ('num_tokens', len(tokens))
        yield ('num_tokens_div5', len(tokens) / 5)


# features on document structure

_single_sentence = [
    # offset
    ('num_edus_from_sent_start', Substance.CONTINUOUS),
    # revOffset
    ('num_edus_to_sent_end', Substance.CONTINUOUS),
    # sentenceID
    ('sentence_id', Substance.CONTINUOUS),
    # revSentenceID
    ('num_edus_to_para_end', Substance.CONTINUOUS)
]


def extract_single_sentence(doc, edu):
    """Sentence features for the EDU"""
    edus = doc.edus

    sent = get_sentence(doc, edu)
    if sent is not None:
        # position of EDU in sentence
        edus_sent = [e for e in edus
                     if get_sentence(doc, e) == sent]
        yield ('num_edus_from_sent_start', edus_sent.index(edu))
        yield ('num_edus_to_sent_end', list(reversed(edus_sent)).index(edu))
        # position of sentence in doc
        yield ('sentence_id', sent.num)

    # TODO: check for the 10th time if this is a bug in Li et al.'s parser
    para = get_paragraph(doc, edu)
    if para is not None:
        edus_para = [e for e in edus
                     if get_paragraph(doc, e) == para]
        yield ('num_edus_to_para_end', list(reversed(edus_para)).index(edu))


_single_para = [
    ('paragraph_id', Substance.CONTINUOUS),
    ('paragraph_id_div5', Substance.CONTINUOUS)
]


def extract_single_para(doc, edu):
    """paragraph features for the EDU"""
    para = get_paragraph(doc, edu)
    if para is not None:
        yield ('paragraph_id', para.num)
        yield ('paragraph_id_div5', para.num / 5)


# features on syntax

# helper
def get_syntactic_labels(current, edu):
    "Syntactic labels for this EDU"
    result = []

    ptrees = current.ptb_trees[edu]
    if ptrees is None:
        return None
    # for each PTB tree, get the tree position of its leaves that are in the
    # EDU
    tpos_leaves_edu = ((ptree, [tpos_leaf
                                for tpos_leaf in ptree.treepositions('leaves')
                                if ptree[tpos_leaf].overlaps(edu)])
                       for ptree in ptrees)
    # for each span of syntactic leaves in this EDU
    for ptree, leaves in tpos_leaves_edu:
        tpos_parent = lowest_common_parent(leaves)
        # for each leaf between leftmost and rightmost, add its ancestors
        # up to the lowest common parent
        for leaf in leaves:
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


_single_syntax = [
    ('SYN', Substance.BASKET)
]


def extract_single_syntax(doc, edu):
    """syntactic features for the EDU"""
    syn_labels = get_syntactic_labels(doc, edu)
    if syn_labels is not None:
        yield ('SYN', syn_labels)


# TODO: features on semantic similarity

def build_edu_feature_extractor():
    """Build the feature extractor for single EDUs"""
    feats = []
    funcs = []

    # meta
    # feats.extend(_single_meta)
    # funcs.append(extract_single_meta)

    # concrete features
    # word
    feats.extend(_single_word)
    funcs.append(extract_single_word)
    # pos
    feats.extend(_single_pos)
    funcs.append(extract_single_pos)
    # length
    feats.extend(_single_length)
    funcs.append(extract_single_length)
    # para
    feats.extend(_single_para)
    funcs.append(extract_single_para)
    # sent
    feats.extend(_single_sentence)
    funcs.append(extract_single_sentence)
    # syntax (disabled)
    # feats.extend(_single_syntax)
    # funcs.append(extract_single_syntax)

    def _extract_all(doc, edu):
        """inner helper because I am lost at sea here"""
        # TODO do this in a cleaner manner
        for fct in funcs:
            for feat in fct(doc, edu):
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

# meta features
_pair_meta = [
    ('grouping', Substance.STRING)
]


def extract_pair_meta(doc, sf_cache, edu1, edu2):
    """core features"""
    yield ('grouping', feat_grouping(doc, edu1, edu2))


# concrete features
# word
_pair_word = [
    ('ptb_word_first_pairs', Substance.DISCRETE),
    ('ptb_word_last_pairs', Substance.DISCRETE),
    ('ptb_word_first2_pairs', Substance.DISCRETE),
    ('ptb_word_last2_pairs', Substance.DISCRETE),
]


def extract_pair_word(doc, sf_cache, edu1, edu2):
    """word tuple features"""
    feats_edu1 = sf_cache[edu1]
    feats_edu2 = sf_cache[edu2]
    yield ('ptb_word_first_pairs', (feats_edu1['ptb_word_first'],
                                    feats_edu2['ptb_word_first']))
    yield ('ptb_word_last_pairs', (feats_edu1['ptb_word_last'],
                                   feats_edu2['ptb_word_last']))
    try:
        yield ('ptb_word_first2_pairs', (feats_edu1['ptb_word_first2'],
                                         feats_edu2['ptb_word_first2']))
        yield ('ptb_word_last2_pairs', (feats_edu1['ptb_word_last2'],
                                        feats_edu2['ptb_word_last2']))
    except KeyError:
        # ignore when there are less than 2 tokens in either EDU
        pass


# pos
_pair_pos = [
    ('ptb_pos_tag_first_pairs', Substance.DISCRETE),
    ('POSF', Substance.BASKET),
    ('POSS', Substance.BASKET)
]


def extract_pair_pos(doc, sf_cache, edu1, edu2):
    """POS tuple features"""
    feats_edu1 = sf_cache[edu1]
    feats_edu2 = sf_cache[edu2]
    yield ('ptb_pos_tag_first_pairs', (feats_edu1['ptb_pos_tag_first'],
                                       feats_edu2['ptb_pos_tag_first']))

    try:
        pos_tags1 = feats_edu1['POS']
    except KeyError:
        pass
    else:
        for tag in pos_tags1:
            yield ('POSF', tag)

    try:
        pos_tags2 = feats_edu2['POS']
    except KeyError:
        pass
    else:
        for tag in pos_tags2:
            yield ('POSS', tag)


_pair_length = [
    ('num_tokens_div5_pair', Substance.DISCRETE),
    ('num_tokens_diff_div5', Substance.CONTINUOUS)
]


def extract_pair_length(doc, sf_cache, edu1, edu2):
    """Sentence tuple features"""

    feats_edu1 = sf_cache[edu1]
    feats_edu2 = sf_cache[edu2]

    num_toks1 = feats_edu1['num_tokens']
    num_toks2 = feats_edu2['num_tokens']
    num_toks1_div5 = feats_edu1['num_tokens_div5']
    num_toks2_div5 = feats_edu2['num_tokens_div5']

    yield ('num_tokens_div5_pair', (num_toks1_div5, num_toks2_div5))
    yield ('num_tokens_diff_div5', (num_toks1 - num_toks2) / 5)


_pair_para = [
    ('first_paragraph', Substance.DISCRETE),
    ('num_paragraphs_between', Substance.CONTINUOUS),
    ('num_paragraphs_between_div3', Substance.CONTINUOUS)
]


def extract_pair_para(doc, sf_cache, edu1, edu2):
    """Paragraph tuple features"""
    feats_edu1 = sf_cache[edu1]
    feats_edu2 = sf_cache[edu2]

    try:
        para_id1 = feats_edu1['paragraph_id']
        para_id2 = feats_edu2['paragraph_id']
    except KeyError:
        pass
    else:
        if para_id1 < para_id2:
            first_para = 'first'
        elif para_id1 > para_id2:
            first_para = 'second'
        else:
            first_para = 'same'
        yield ('first_paragraph', first_para)

        yield ('num_paragraphs_between', para_id1 - para_id2)
        yield ('num_paragraphs_between_div3', (para_id1 - para_id2) / 3)


_pair_sent = [
    ('offset_diff', Substance.CONTINUOUS),
    ('rev_offset_diff', Substance.CONTINUOUS),
    ('offset_diff_div3', Substance.CONTINUOUS),
    ('rev_offset_diff_div3', Substance.CONTINUOUS),
    ('offset_pair', Substance.DISCRETE),
    ('rev_offset_pair', Substance.DISCRETE),
    ('offset_div3_pair', Substance.DISCRETE),
    ('rev_offset_div3_pair', Substance.DISCRETE),
    ('line_id_diff', Substance.CONTINUOUS),
    ('same_bad_sentence', Substance.DISCRETE),
    ('sentence_id_diff', Substance.CONTINUOUS),
    ('sentence_id_diff_div3', Substance.CONTINUOUS),
    ('rev_sentence_id_diff', Substance.CONTINUOUS),
    ('rev_sentence_id_diff_div3', Substance.CONTINUOUS)
]


def extract_pair_sent(doc, sf_cache, edu1, edu2):
    """Sentence tuple features"""
    feats_edu1 = sf_cache[edu1]
    feats_edu2 = sf_cache[edu2]

    # offset features
    try:
        offset1 = feats_edu1['num_edus_from_sent_start']
        offset2 = feats_edu2['num_edus_from_sent_start']
    except KeyError:
        pass
    else:
        yield ('offset_diff', offset1 - offset2)
        yield ('offset_diff_div3', (offset1 - offset2) / 3)
        yield ('offset_pair', (offset1, offset2))
        yield ('offset_div3_pair', (offset1 / 3, offset2 / 3))
    # rev_offset features
    try:
        rev_offset1 = feats_edu1['num_edus_to_sent_end']
        rev_offset2 = feats_edu2['num_edus_to_sent_end']
    except KeyError:
        pass
    else:
        yield ('rev_offset_diff', rev_offset1 - rev_offset2)
        yield ('rev_offset_diff_div3', (rev_offset1 - rev_offset2) / 3)
        yield ('rev_offset_pair', (rev_offset1, rev_offset2))
        yield ('rev_offset_div3_pair', (rev_offset1 / 3, rev_offset2 / 3))

    # lineID: distance of edu in EDUs from document start
    line_id1 = edu1.num - 1  # real EDU numbers are in [1..]
    line_id2 = edu2.num - 1
    yield ('line_id_diff', line_id1 - line_id2)

    # sentenceID
    try:
        sent_id1 = feats_edu1['sentence_id']
        sent_id2 = feats_edu2['sentence_id']
    except KeyError:
        pass
    else:
        yield ('same_bad_sentence',
               'same' if sent_id1 == sent_id2 else 'different')
        yield ('sentence_id_diff', sent_id1 - sent_id2)
        yield ('sentence_id_diff_div3', (sent_id1 - sent_id2) / 3)
    # revSentenceID
    try:
        rev_sent_id1 = feats_edu1['num_edus_to_para_end']
        rev_sent_id2 = feats_edu2['num_edus_to_para_end']
    except KeyError:
        pass
    else:
        yield ('rev_sentence_id_diff', rev_sent_id1 - rev_sent_id2)
        yield ('rev_sentence_id_diff_div3', (rev_sent_id1 - rev_sent_id2) / 3)


_pair_syntax = [
    ('SYNF', Substance.BASKET),
    ('SYNS', Substance.BASKET)
]


def extract_pair_syntax(doc, sf_cache, edu1, edu2):
    """Syntax pair features"""
    feats_edu1 = sf_cache[edu1]
    feats_edu2 = sf_cache[edu2]

    try:
        syn_labs1 = feats_edu1['SYN']
    except KeyError:
        pass
    else:
        for lab in syn_labs1:
            yield ('SYNF', lab)

    try:
        syn_labs2 = feats_edu2['SYN']
    except KeyError:
        pass
    else:
        for lab in syn_labs2:
            yield ('SYNS', lab)


def build_pair_feature_extractor():
    """Build the feature extractor for pairs of EDUs

    TODO: properly emit features on single EDUs ;
    they are already stored in sf_cache, but under (slightly) different
    names
    """
    feats = []
    funcs = []

    # meta
    # feats.extend(_pair_meta)
    # funcs.append(extract_pair_meta)

    # concrete features
    # feature type: 1
    feats.extend(_pair_word)
    funcs.append(extract_pair_word)
    # 2
    feats.extend(_pair_pos)
    funcs.append(extract_pair_pos)
    # 3
    feats.extend(_pair_para)
    funcs.append(extract_pair_para)
    feats.extend(_pair_sent)
    funcs.append(extract_pair_sent)
    # 4
    feats.extend(_pair_length)
    funcs.append(extract_pair_length)
    # 5
    # feats.extend(_pair_syntax)
    # funcs.append(extract_pair_syntax)
    # 6
    # feats.extend(_pair_semantics)  # NotImplemented
    # funcs.append(extract_pair_semantics)

    def _extract_all(doc, sf_cache, edu1, edu2):
        """inner helper because I am lost at sea here, again"""
        # TODO do this in a cleaner manner
        for fct in funcs:
            for feat in fct(doc, sf_cache, edu1, edu2):
                yield feat

    # header
    header = feats
    # extractor
    feat_extractor = _extract_all
    # return header and extractor
    return header, feat_extractor
