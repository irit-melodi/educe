"""This module is a loose collection of diagnostic functions on an RST corpus.

"""

from __future__ import print_function

from collections import Counter
import os

import pandas as pd

from educe.internalutil import treenode
from educe.rst_dt.annotation import RSTTree
from educe.rst_dt.corpus import (Reader as RstReader,
                                 RstRelationConverter)


# RST corpus
# TODO import CORPUS_DIR/CD_TRAIN e.g. from educe.rst_dt.rst_wsj_corpus
CORPUS_DIR = os.path.join('/home/mathieu/travail/recherche/melodi/educe',
                          'data/rst_discourse_treebank/data',
                          'RSTtrees-WSJ-main-1.0/')
CD_TRAIN = os.path.join(CORPUS_DIR, 'TRAINING')
# relation converter (fine- to coarse-grained labels)
# TODO same as above
RELMAP_FILE = os.path.join('/home/mathieu/travail/recherche/melodi/educe',
                           'educe', 'rst_dt',
                           'rst_112to18.txt')
REL_CONV = RstRelationConverter(RELMAP_FILE).convert_tree


def load_training_as_dataframe():
    """Load training section of the RST-WSJ corpus as a pandas.DataFrame.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame of all instances of relations in the training section.
        Interesting columns are 'rel', 'nuc_sig', 'arity'
    """
    rst_phrases = []  # list of rows, each represented as a dict

    rst_reader = RstReader(CD_TRAIN)
    rst_corpus = rst_reader.slurp()
    for doc_id, rtree_ref in sorted(rst_corpus.items()):
        # convert labels to coarse
        coarse_rtree_ref = REL_CONV(rtree_ref)
        # store "same-unit" subtrees
        heterogeneous_nodes = []
        internal_nodes = lambda t: isinstance(t, RSTTree) and len(t) > 1
        for su_subtree in coarse_rtree_ref.subtrees(filter=internal_nodes):
            # get each kid's relation
            kid_rels = tuple(treenode(kid).rel for kid in su_subtree)
            # filter out nodes whose kids have different relations
            rels = [r for r in set(kid_rels) if r != 'span']
            if len(rels) > 1:
                heterogeneous_nodes.append(kid_rels)
                continue

            # process homogeneous nodes
            res = dict()
            rel = rels[0]
            res['rel'] = rel
            # arity
            res['arity'] = len(su_subtree)  # number of kids
            # nuclearity signature
            kid_nucs = tuple(treenode(kid).nuclearity for kid in su_subtree)
            nuc_sig = ''.join('S' if kid_nuc == 'Satellite' else 'N'
                              for kid_nuc in kid_nucs)
            res['nuc_sig'] = (nuc_sig if nuc_sig in frozenset(['SN', 'NS'])
                              else 'NN')
            # TODO len(kid_rels) - 1 is the nb of bin rels

            # height
            rel_hgt = su_subtree.height()
            res['height'] = rel_hgt

            # TODO disc relations of the grandchildren
            #

            rst_phrases.append(res)
    # turn into a DataFrame
    df = pd.DataFrame(rst_phrases)
    # add calculated columns
    # * "undirected" nuclearity, e.g. NS == SN
    df['unuc_sig'] = map(lambda nuc_sig: ('NS' if nuc_sig in ['NS', 'SN']
                                          else 'NN'),
                         df.nuc_sig)
    return df


def get_most_frequent_unuc(df, verbose=False):
    """Get the most frequent undirected nuclearity for each relation.

    Returns
    -------
    most_freq_unuc: dict(str, str)
        Map each relation to its most frequent (aka mode) undirected
        nuclearity signature.
    """
    # get number of occurrences of each relation
    # * "directed" nuclearity, e.g. NS != SN
    # TODO check and try pandas' hierarchical indexing ;
    # my understanding is we would get relation as 1st and nuclearity as 2nd
    # levels
    if verbose:
        grouped_rel_nuc = df.groupby(['rel', 'nuc_sig'])
        print('\n'.join('{}: {}'.format(rel_nuc, len(occs))
                        for rel_nuc, occs in grouped_rel_nuc))
    # * "undirected" nuclearity, e.g. NS == SN
    # TODO hierarchical indexing (again)
    if verbose:
        grouped_rel_unuc = df.groupby(['rel', 'unuc_sig'])
        print('\n'.join(
            '{:{}s}\t{}\t{}'.format(
                rel, max(len(r) for r in df.rel.unique()), unuc, len(occs))
            for (rel, unuc), occs in grouped_rel_unuc))

    # use this data to get:
    # * unambiguously mono-nuclear relations
    # * unambiguously multi-nuclear relations
    # * most common nuclearity for the remaining (ambiguous wrt nuclearity)
    # relations
    most_freq_unuc = {rel: occs['unuc_sig'].mode()[0]
                      for rel, occs in df.groupby('rel')}
    if verbose:
        print('\n'.join('{}\t{}'.format(rel, unuc)
                        for rel, unuc in sorted(most_freq_unuc.items())))

    return most_freq_unuc


def check_label_ranks():
    """Examine label and rank of attachment"""
    # FIXME rewrite entirely
    labels_ranks_gold = []  # TODO
    labels_ranks_no_nuc = [(lbl[:-3] if lbl is not None and lbl != 'ROOT'
                            else lbl, rnk)
                           for lbl, rnk in labels_ranks_gold]
    print('\n'.join('{}\t{}\t{}'.format(lbl, rnk, occ)
                    for (lbl, rnk), occ
                    in sorted(Counter(labels_ranks_no_nuc).items())))
    print(sorted(set(lbl for lbl, rnk in labels_ranks_gold)))
    print('labels inc. nuc: {}'.format(
        len(set(lbl for lbl, rnk in labels_ranks_gold))))
    print(('labels inc. rank: {}'.format(
        len(Counter(labels_ranks_no_nuc)))))
    print('\n'.join('{}\t{}\t{}'.format(lbl, rnk, occ)
                    for (lbl, rnk), occ
                    in sorted(Counter(labels_ranks_gold).items())))
    print('labels inc. nuc and rank :', len(Counter(labels_ranks_gold)))
    print('\n\n\n')
