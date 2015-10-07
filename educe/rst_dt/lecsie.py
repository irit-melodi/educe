"""Loader for LECSIE features extracted by J. Conrath on the RST corpus.
"""


from __future__ import print_function

import itertools
import os

import numpy as np


FNAMES_TRAIN = [
    'rstdt_lecsie_feats_attached_train.txt',
    'rstdt_lecsie_feats_unattached_train.txt',
]

FNAMES_TEST = [
    'rstdt_lecsie_feats_attached_test.txt',
    'rstdt_lecsie_feats_unattached_test.txt',
]

LINE_FORMAT = [
    'pair_id',
    'is_adjacent',
    'is_intra',
    # 'VERB1',  # not here?
    # 'VERB2',  # not here?
    # comb
    'elaboration_wcomb',
    'continuation_wcomb',
    'alternation_wcomb',
    'cause_wcomb',
    'temporal_wcomb',
    'contrast_wcomb',
    # specificity
    'elaboration_specificity',
    'continuation_specificity',
    'alternation_specificity',
    'cause_specificity',
    'temporal_specificity',
    'contrast_specificity',
    # norm pmi
    'elaboration_normpmi',
    'continuation_normpmi',
    'alternation_normpmi',
    'cause_normpmi',
    'temporal_normpmi',
    'contrast_normpmi',
]


def _load_lecsie_feats_file(f):
    """Actually do load"""
    lecsie_feats = []

    header_line = next(f)
    for line in f:
        # should have "VERB1 VERB2" between "is_intra" and
        # "elaboration_wcomb" ?
        fields = line.strip().split('\t')
        # disassemble the identifier of the pair
        doc_name, s1b, s1e, s2b, s2e = fields[0].rsplit('_', 4)
        pair_id = [doc_name, int(s1b), int(s1e), int(s2b), int(s2e)]
        assoc_scores = [float(feat) if feat != 'None' else np.nan
                        for feat in fields[3:]]
        pair_feats = pair_id
        pair_feats.extend(assoc_scores)
        lecsie_feats.append(pair_feats)

    return lecsie_feats


def load_lecsie_feats_file(f):
    """Load a LECSIE features file."""
    with open(f) as f:
        lecsie_feats = _load_lecsie_feats_file(f)
    return lecsie_feats


def load_lecsie_feats(data_dir):
    """Load LECSIE features for the RST-WSJ corpus."""
    lecsie_feats = list(itertools.chain.from_iterable(
        load_lecsie_feats_file(os.path.join(data_dir, fn))
        for fn in itertools.chain(FNAMES_TRAIN, FNAMES_TEST)))
    lecsie_feats = sorted(lecsie_feats)
    return lecsie_feats
