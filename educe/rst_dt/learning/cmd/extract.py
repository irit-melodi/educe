#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""
Extract features to CSV files
"""

from __future__ import print_function
import os
import sys

import educe.corpus
import educe.glozz
import educe.stac
import educe.util
from ..args import add_usual_input_args
from ..base import read_corpus_inputs, extract_pair_features
from ....learning.orange_format import dump_orange_tab_file


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
    inputs = read_corpus_inputs(args)
    feature_set = args.feature_set
    live = args.parsing

    # extract instances
    X = extract_pair_features(inputs, feature_set=feature_set, live=live)
    
    # dump instances to file
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if live:
        out_file = os.path.join(args.output, 'extracted-features.tab')
    else:
        of_bn = os.path.join(args.output, os.path.basename(args.corpus))
        of_ext = '.tab'
        out_file = '{}.relations{}'.format(of_bn, of_ext)
    dump_orange_tab_file(X, [], out_file)
