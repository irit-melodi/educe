#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""
Extract features to CSV files
"""

from __future__ import print_function
import os
import itertools

import educe.corpus
import educe.glozz
import educe.stac
import educe.util

from educe.learning.svmlight_format import dump_svmlight_file
from educe.learning.edu_input_format import dump_all
from educe.learning.vocabulary_format import dump_vocabulary
from ..args import add_usual_input_args
from ..doc_vectorizer import DocumentCountVectorizer, DocumentLabelExtractor
from educe.rst_dt.corpus import RstDtParser
from educe.rst_dt.ptb import PtbParser


NAME = 'extract'


# ----------------------------------------------------------------------
# options
# ----------------------------------------------------------------------

def config_argparser(parser):
    """
    Subcommand flags.
    """
    add_usual_input_args(parser)
    parser.add_argument('corpus', metavar='DIR',
                        help='Corpus dir (eg. data/pilot)')
    parser.add_argument('ptb', metavar='DIR',
                        help='PTB directory (eg. PTBIII/parsed/wsj)')
    parser.add_argument('output', metavar='DIR',
                        help='Output directory')
    # add flags --doc, --subdoc, etc to allow user to filter on these things
    educe.util.add_corpus_filters(parser,
                                  fields=['doc'])
    parser.add_argument('--verbose', '-v', action='count',
                        default=1)
    parser.add_argument('--quiet', '-q', action='store_const',
                        const=0,
                        dest='verbose')
    parser.add_argument('--parsing', action='store_true',
                        help='Extract features for parsing')
    parser.add_argument('--debug', action='store_true',
                        help='Emit fields used for debugging purposes')
    parser.add_argument('--experimental', action='store_true',
                        help='Enable experimental features '
                             '(currently none)')
    parser.set_defaults(func=main)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main(args):
    "main for feature extraction mode"
    # retrieve parameters
    feature_set = args.feature_set
    live = args.parsing
    # RST data
    rst_reader = RstDtParser(args.corpus, args, coarse_rels=True)
    rst_corpus = rst_reader.corpus
    # PTB data
    ptb_parser = PtbParser(args.ptb)
    # instance generator
    instance_generator = lambda doc: doc.sorted_all_inv_edu_pairs()
    # TODO: change rst_corpus, e.g. to return an OrderedDict,
    # so that the order in which docs are enumerated is guaranteed
    # to be always the same

    # generate all instances
    vzer = DocumentCountVectorizer(rst_reader.decode,
                                   instance_generator,
                                   rst_reader.segment, rst_reader.parse,
                                   ptb_parser.tokenize, ptb_parser.parse,
                                   feature_set,
                                   min_df=1)
    X_gen = vzer.fit_transform(rst_corpus)

    # extract class label for each instance
    if live:
        y_gen = itertools.repeat(0)
    else:
        labtor = DocumentLabelExtractor(rst_reader.decode,
                                        instance_generator,
                                        rst_reader.segment, rst_reader.parse)
        # y_gen = labtor.fit_transform(rst_corpus)
        # fit then transform enables to get classes_ for the dump
        labtor.fit(rst_corpus)
        y_gen = labtor.transform(rst_corpus)

    # dump instances to files
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # data file
    of_ext = '.sparse'
    if live:
        out_file = os.path.join(args.output, 'extracted-features' + of_ext)
    else:
        of_bn = os.path.join(args.output, os.path.basename(args.corpus))
        out_file = '{}.relations{}'.format(of_bn, of_ext)

    # dump
    # edu_input_format.dump_all() in turn calls dump_svmlight_file
    # this is a poor man's solution but it is good enough for the time being
    dump_all(X_gen, y_gen, out_file, labtor.labelset_)
    # dump_svmlight_file(X_gen, y_gen, out_file)

    # dump vocabulary
    vocab_file = out_file + '.vocab'
    dump_vocabulary(vzer.vocabulary_, vocab_file)
