"""This module converts the RST-DT corpus to dependency trees.

"""
from __future__ import absolute_import, print_function

import argparse
import os

from educe.rst_dt.corpus import Reader
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.rst_wsj_corpus import TRAIN_FOLDER, TEST_FOLDER
from educe.rst_dt.disdep_format import dump_disdep_files


def dump_dep_rstdt(corpus_dir, out_dir, nary_enc):
    """Convert and dump the RST-DT corpus as dependency trees."""
    # convert and dump RST trees from train
    dir_train = os.path.join(corpus_dir, TRAIN_FOLDER)
    if not os.path.isdir(dir_train):
        raise ValueError('No such folder: {}'.format(dir_train))
    reader_train = Reader(dir_train)
    trees_train = reader_train.slurp()
    dtrees_train = {doc_name: RstDepTree.from_rst_tree(rst_tree,
                                                       nary_enc=nary_enc)
                    for doc_name, rst_tree in trees_train.items()}
    dump_disdep_files(dtrees_train.values(),
                      os.path.join(out_dir, os.path.basename(dir_train)))

    # convert and dump RST trees from test
    dir_test = os.path.join(corpus_dir, TEST_FOLDER)
    if not os.path.isdir(dir_test):
        raise ValueError('No such folder: {}'.format(dir_test))
    reader_test = Reader(dir_test)
    trees_test = reader_test.slurp()
    dtrees_test = {doc_name: RstDepTree.from_rst_tree(rst_tree,
                                                      nary_enc=nary_enc)
                   for doc_name, rst_tree in trees_test.items()}
    dump_disdep_files(dtrees_test.values(),
                      os.path.join(out_dir, os.path.basename(dir_test)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export a dependency version of the RST-DT corpus.'
    )
    # TODO obtain corpus_dir through a global(?) variable elsewhere, eg.
    # in the educe.rst_dt module ; the idea is to have a central place
    # to gather this info so it's accessible for the various utilities
    parser.add_argument('--corpus_dir',
                        default=os.path.join(
                            os.path.expanduser('~'),  # user home
                            'corpora', 'rst-dt',
                            'rst_discourse_treebank/data'
                        ),
                        help='Base folder of the corpus')
    # TODO ibid, although what would make a "good" mechanism to access
    # a "good" default value is less clear in this instance
    parser.add_argument('--out_dir',
                        default=os.path.join(
                            os.path.expanduser('~'),  # user home
                            'melodi/irit-rst-dt',
                            'TMP_disdep_chain_true'
                        ),
                        help='Output folder')
    # 'chain' corresponds to the de facto standard setting in the
    # evaluation of RST parsers in the literature, but 'tree' enables
    # lossless roundtrips between c-trees and head-ordered d-trees
    parser.add_argument('--nary_enc',
                        default='chain',
                        choices=['chain', 'tree'],
                        help='Encoding of n-ary nodes')
    args = parser.parse_args()
    dump_dep_rstdt(args.corpus_dir, args.out_dir, args.nary_enc)
