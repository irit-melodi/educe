"""This module enables to generate dependency versions of the RST-DT.

It also contains methods to collect and analyze the dependencies.
"""

from __future__ import absolute_import, print_function

import os

import pandas as pd

import educe.rst_dt
from educe.rst_dt.corpus import RstRelationConverter, RELMAP_112_18_FILE
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.pseudo_relations import (
    merge_same_units, rewrite_pseudo_rels
)


def read_corpus(data_dir, section='all', verbose=True):
    """Read a corpus.

    See `educe.rst_dt.util.args.read_corpus()`.

    Parameters
    ----------
    data_dir : str
        Path to the corpus folder.
    section : str, one of {'train', 'test', 'all'}
        Section of interest in the RST-DT.
    verbose : boolean, defaults to True
        Verbosity.

    Returns
    -------
    corpus : dict from str to dict from FileId to RSTTree
        Corpus as a dict from {'train', 'test'} to a dict from FileId
        to RST c-tree.
    """
    if section not in ('train', 'test', 'all'):
        raise ValueError("section needs to be one of {'train', 'test', "
                         "'all'}")
    if not os.path.exists(data_dir):
        raise ValueError('Unable to find data_dir {}'.format(
            os.path.abspath(data_dir)))
    # read c-trees from treebank
    corpus = dict()
    if section in ('train', 'all'):
        dir_train = os.path.join(data_dir, 'TRAINING')
        reader_train = educe.rst_dt.Reader(dir_train)
        corpus['train'] = reader_train.slurp(cfiles=None, verbose=verbose)
    if section in ('test', 'all'):
        dir_test = os.path.join(data_dir, 'TEST')
        reader_test = educe.rst_dt.Reader(dir_test)
        corpus['test'] = reader_test.slurp(cfiles=None, verbose=verbose)
    return corpus


def read_deps(corpus, section='all', nary_enc='chain',
              rew_pseudo_rels=False, mrg_same_units=False):
    """Collect dependencies from the corpus.

    Parameters
    ----------
    corpus : dict from str to dict from FileId to RSTTree
        Corpus of RST c-trees indexed by {'train', 'test'} then FileId.
    section : str, one of {'train', 'test', 'all'}
        Section of interest in the RST-DT.
    nary_enc : str, one of {'tree', 'chain'}
        Encoding of n-ary relations used in the c-to-d conversion.
    rew_pseudo_rels : boolean, defaults to False
        If True, rewrite pseudo relations ; see
        `educe.rst_dt.pseudo_relations`.
    mrg_same_units : boolean, defaults to False
        If True, merge fragmented EDUs ; see
        `educe.rst_dt.pseudo_relations`.

    Returns
    -------
    edu_df : pandas.DataFrame
        Table of EDUs read from the corpus.
    dep_df : pandas.DataFrame
        Table of dependencies read from the corpus.
    """
    # experimental: rewrite pseudo-relations
    if rew_pseudo_rels:
        for sec_name, sec_corpus in corpus.items():
            corpus[sec_name] = {
                doc_id: rewrite_pseudo_rels(doc_id, rst_ctree)
                for doc_id, rst_ctree in sec_corpus.items()
            }
    if mrg_same_units:
        for sec_name, sec_corpus in corpus.items():
            corpus[sec_name] = {
                doc_id: merge_same_units(doc_id, rst_ctree)
                for doc_id, rst_ctree in sec_corpus.items()
            }
    # convert to d-trees, collect dependencies
    edus = []
    deps = []
    for sec_name, sec_corpus in corpus.items():
        for doc_id, rst_ctree in sorted(sec_corpus.items()):
            doc_name = doc_id.doc
            doc_text = rst_ctree.text()
            # DIRTY infer (approximate) sentence and paragraph indices
            # from newlines in the text (\n and \n\n)
            sent_idx = 0
            para_idx = 0
            # end DIRTY
            rst_dtree = RstDepTree.from_rst_tree(rst_ctree, nary_enc='chain')
            for dep_idx, (edu, hd_idx, lbl, nuc, hd_order) in enumerate(
                    zip(rst_dtree.edus[1:],
                        rst_dtree.heads[1:], rst_dtree.labels[1:],
                        rst_dtree.nucs[1:], rst_dtree.ranks[1:]),
                    start=1):
                char_beg = edu.span.char_start
                char_end = edu.span.char_end
                edus.append(
                    (sec_name, doc_name,
                     dep_idx, char_beg, char_end, sent_idx, para_idx)
                )
                deps.append(
                    (doc_name,
                     dep_idx, hd_idx, lbl, nuc, hd_order)
                )
                # DIRTY search for paragraph or sentence breaks in the
                # text of the EDU *plus the next three characters* (yerk)
                edu_txt_plus = doc_text[char_beg:char_end + 3]
                if '\n\n' in edu_txt_plus:
                    para_idx += 1
                    sent_idx += 1  # sometimes wrong ; to be fixed
                elif '\n' in edu_txt_plus:
                    sent_idx += 1
                # end DIRTY
    # turn into DataFrame
    edu_df = pd.DataFrame(edus, columns=[
        'section', 'doc_name', 'dep_idx', 'char_beg', 'char_end',
        'sent_idx', 'para_idx']
    )
    dep_df = pd.DataFrame(deps, columns=[
        'doc_name', 'dep_idx',
        'hd_idx', 'rel', 'nuc', 'hd_order']
    )
    # additional columns
    # * attachment length in EDUs
    dep_df['len_edu'] = dep_df['dep_idx'] - dep_df['hd_idx']
    dep_df['len_edu_abs'] = abs(dep_df['len_edu'])
    # * attachment length, in sentences and paragraphs
    if False:
        # TODO rewrite in a pandas-ic manner ; my previous attempts have
        # failed but I think I got pretty close
        # NB: the current implementation is *extremely* slow: 155 seconds
        # on my laptop for the RST-DT, just for this (minor) computation
        len_sent = []
        len_para = []
        for _, row in dep_df[['doc_name', 'dep_idx', 'hd_idx']].iterrows():
            edu_dep = edu_df[
                (edu_df['doc_name'] == row['doc_name']) &
                (edu_df['dep_idx'] == row['dep_idx'])
            ]
            if row['hd_idx'] == 0:
                # {sent,para}_idx + 1 for dependents of the fake root
                lsent = edu_dep['sent_idx'].values[0] + 1
                lpara = edu_dep['para_idx'].values[0] + 1
            else:
                edu_hd = edu_df[
                    (edu_df['doc_name'] == row['doc_name']) &
                    (edu_df['dep_idx'] == row['hd_idx'])
                ]
                lsent = (edu_dep['sent_idx'].values[0] -
                         edu_hd['sent_idx'].values[0])
                lpara = (edu_dep['para_idx'].values[0] -
                         edu_hd['para_idx'].values[0])
            len_sent.append(lsent)
            len_para.append(lpara)
        dep_df['len_sent'] = pd.Series(len_sent)
        dep_df['len_sent_abs'] = abs(dep_df['len_sent'])
        dep_df['len_para'] = pd.Series(len_para)
        dep_df['len_para_abs'] = abs(dep_df['len_para'])
    # * class of relation (FIXME we need to handle interaction with
    #   rewrite_pseudo_rels)
    rel_conv = RstRelationConverter(RELMAP_112_18_FILE).convert_label
    dep_df['rel_class'] = dep_df['rel'].apply(rel_conv)
    # * boolean indicator for pseudo-relations ; NB: the 'Style-' prefix
    # can only apply if rew_pseudo_rels (otherwise no occurrence)
    dep_df['pseudo_rel'] = (
        (dep_df['rel'].str.startswith('Style')) | 
        (dep_df['rel'].str.endswith('Same-Unit')) |
        (dep_df['rel'].str.endswith('TextualOrganization'))
    )
    return edu_df, dep_df
