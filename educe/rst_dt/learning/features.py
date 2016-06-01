"""
Feature extraction library functions for RST_DT corpus
"""

from educe.learning.keys import Substance
from .base import DocumentPlusPreprocessor


def build_doc_preprocessor():
    """Build the preprocessor for feature extraction in each EDU of doc"""
    return DocumentPlusPreprocessor().preprocess


# ---------------------------------------------------------------------
# single EDUs
# ---------------------------------------------------------------------

# formerly Text subgroup
# raw words aka not-PTB-tokens
SINGLE_RAW_WORD = [
    ('word_first', Substance.DISCRETE),
    ('word_last', Substance.DISCRETE),
    ('bigram_first', Substance.DISCRETE),
    ('bigram_last', Substance.DISCRETE),
    ('num_tokens', Substance.CONTINUOUS)
]


def extract_single_raw_word(edu_info):
    """raw word features for the EDU"""
    raw_words = edu_info['raw_words']
    if raw_words:
        yield ('word_first', raw_words[0])
        yield ('word_last', raw_words[-1])
    if len(raw_words) > 1:
        yield ('bigram_first', (raw_words[0], raw_words[1]))
        yield ('bigram_last', (raw_words[-2], raw_words[-1]))
    yield ('num_tokens', len(raw_words))


# PTB tokens
# note: PTB tokens may not necessarily correspond to words

SINGLE_PTB_TOKEN_WORD = [
    ('ptb_word_first', Substance.DISCRETE),
    ('ptb_word_last', Substance.DISCRETE),
    ('ptb_word_first2', Substance.DISCRETE),
    ('ptb_word_last2', Substance.DISCRETE)
]


def extract_single_ptb_token_word(edu_info):
    """word features on PTB tokens for the EDU"""
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


SINGLE_PTB_TOKEN_POS = [
    ('ptb_pos_tag_first', Substance.DISCRETE),
    ('ptb_pos_tag_last', Substance.DISCRETE),
    ('ptb_pos_tag_first2', Substance.DISCRETE),
    ('ptb_pos_tag_last2', Substance.DISCRETE),
]


def extract_single_ptb_token_pos(edu_info):
    """POS features on PTB tokens for the EDU"""
    try:
        tags = edu_info['tags']
    except KeyError:
        return

    if tags:
        yield ('ptb_pos_tag_first', tags[0])
        yield ('ptb_pos_tag_last', tags[-1])

    if len(tags) > 1:
        yield ('ptb_pos_tag_first2', (tags[0], tags[1]))
        yield ('ptb_pos_tag_last2', (tags[-2], tags[-1]))


def build_edu_feature_extractor():
    """Build the feature extractor for single EDUs"""
    funcs = []

    # raw word
    funcs.append(extract_single_raw_word)
    # PTB word
    funcs.append(extract_single_ptb_token_word)
    # PTB pos
    funcs.append(extract_single_ptb_token_pos)

    def _extract_all(edu_info):
        """inner helper because I am lost at sea here"""
        # TODO do this in a cleaner manner
        for fct in funcs:
            for feat in fct(edu_info):
                yield feat

    # extractor
    feat_extractor = _extract_all
    return feat_extractor


# ---------------------------------------------------------------------
# pair EDU features
# ---------------------------------------------------------------------

PAIR_GAP = [
    ('num_edus_between', Substance.CONTINUOUS),
    ('same_paragraph', Substance.DISCRETE),
    ('same_bad_sentence', Substance.DISCRETE),
    ('same_ptb_sentence', Substance.DISCRETE)
]


def extract_pair_gap(edu_info1, edu_info2):
    """Document tuple features"""
    edu_num1 = edu_info1['edu'].num
    edu_num2 = edu_info2['edu'].num

    edu_num_diff = abs(edu_num2 - edu_num1) - 1
    yield ('num_edus_between', edu_num_diff)

    try:
        para_id1 = edu_info1['para_idx']
        para_id2 = edu_info2['para_idx']
    except KeyError:
        pass
    else:
        same_para = (para_id1 is not None and
                     para_id2 is not None and
                     para_id1 == para_id2)
        yield ('same_paragraph', same_para)

    # same sentence, raw segmentation
    raw_sent_id1 = edu_info1['raw_sent_idx']
    raw_sent_id2 = edu_info2['raw_sent_idx']
    same_bad_sent = (raw_sent_id1 is not None and
                     raw_sent_id2 is not None and
                     raw_sent_id1 == raw_sent_id2)
    yield ('same_bad_sentence', same_bad_sent)

    # same sentence, ptb segmentation
    # FIXME fix behaviour for sentences with no gold segmentation,
    # as then edu2sent = edu2raw_sent
    sent_id1 = edu_info1['sent_idx']
    sent_id2 = edu_info2['sent_idx']
    same_ptb_sent = (sent_id1 is not None and
                     sent_id2 is not None and
                     sent_id1 == sent_id2)
    yield ('same_ptb_sentence', same_ptb_sent)


PAIR_RAW_WORD = [
    ('word_first_pairs', Substance.DISCRETE),
    ('word_last_pairs', Substance.DISCRETE),
    ('bigram_first_pairs', Substance.DISCRETE),
    ('bigram_last_pairs', Substance.DISCRETE),
]


def extract_pair_raw_word(edu_info1, edu_info2):
    """raw word features on EDU pairs"""
    raw_words1 = edu_info1['raw_words']
    raw_words2 = edu_info2['raw_words']

    if raw_words1 and raw_words2:
        yield ('word_first_pairs', (raw_words1[0], raw_words2[0]))
        yield ('word_last_pairs', (raw_words1[-1], raw_words2[0]))

    if len(raw_words1) > 1 and len(raw_words2) > 1:
        yield ('bigram_first_pairs', ((raw_words1[0], raw_words1[1]),
                                      (raw_words2[0], raw_words2[1])))
        yield ('bigram_last_pairs', ((raw_words1[-2], raw_words1[-1]),
                                     (raw_words2[-2], raw_words2[-1])))


PAIR_POS_TAGS = [
    ('ptb_pos_tag_first_pairs', Substance.DISCRETE)
]


def extract_pair_pos_tags(edu_info1, edu_info2):
    """POS tag features on EDU pairs"""
    try:
        tags1 = edu_info1['tags']
        tags2 = edu_info2['tags']
    except KeyError:
        return

    if tags1 and tags2:
        yield ('ptb_pos_tag_first_pairs', (tags1[0], tags2[0]))


def build_pair_feature_extractor():
    """Build the feature extractor for pairs of EDUs

    TODO: properly emit features on single EDUs ;
    they are already stored in sf_cache, but under (slightly) different
    names
    """
    funcs = []

    funcs.append(extract_pair_gap)

    funcs.append(extract_pair_raw_word)

    funcs.append(extract_pair_pos_tags)

    def _extract_all(edu_info1, edu_info2, edu_info_bwn):
        """inner helper because I am lost at sea here, again"""
        # TODO do this in a cleaner manner
        for fct in funcs:
            for feat in fct(edu_info1, edu_info2):
                yield feat

    # extractor
    feat_extractor = _extract_all
    return feat_extractor


def product_features(feats_g, feats_d, feats_gd):
    """Generate features by taking the product of features.

    Parameters
    ----------
    feats_g: dict(feat_name, feat_val)
        features of the gov EDU
    feats_d: dict(feat_name, feat_val)
        features of the dep EDU
    feats_gd: dict(feat_name, feat_val)
        features of the (gov, dep) edge

    Returns
    -------
    pf: dict(feat_name, feat_val)
        product features
    """
    pf = dict()
    return pf


def combine_features(feats_g, feats_d, feats_gd):
    """Generate features by taking a (linear) combination of features.

    I suspect these do not have a great impact, if any, on results.

    Parameters
    ----------
    feats_g: dict(feat_name, feat_val)
        features of the gov EDU
    feats_d: dict(feat_name, feat_val)
        features of the dep EDU
    feats_gd: dict(feat_name, feat_val)
        features of the (gov, dep) edge

    Returns
    -------
    cf: dict(feat_name, feat_val)
        combined features
    """
    cf = dict()
    return cf
