#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""
Extract features to CSV files
"""

from __future__ import print_function
import codecs
import csv
import os
import sys

import educe.corpus
import educe.learning.keys
import educe.glozz
import educe.stac
import educe.util
from .. import features

NAME = 'extract'


def mk_csv_writer(keys, fstream):
    """
    start off csv writer for a given mode
    """
    csv_quoting = csv.QUOTE_MINIMAL
    writer = educe.learning.keys.KeyGroupWriter(fstream,
                                                keys,
                                                quoting=csv_quoting)
    writer.writeheader()
    return writer


# ----------------------------------------------------------------------
# options
# ----------------------------------------------------------------------


def config_argparser(parser):
    """
    Subcommand flags.
    """
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


def main_parsing_pairs(args):
    """
    Main to call when live data are passed in (--parsing). Live data are data
    that we want to discourse parsing on, so we don't know if they are attached
    or what the label is.

    There used to be an expectation that live data was also flat data
    (given with --live), but as of 2014-03-24, we are experimenting with
    have hierarchical live data
    """
    inputs = features.read_corpus_inputs(args)
    features_file = os.path.join(args.output, 'extracted-features.csv')
    with codecs.open(features_file, 'wb') as ofile:
        header = features.PairKeys(inputs)
        writer = mk_csv_writer(header, ofile)
        feats = features.extract_pair_features(inputs,
                                               live=True)
        for row, _ in feats:
            writer.writerow(row)


def _write_pairs(gen, r_ofile, p_ofile):
    """
    Given a generator of pairs and the relations/pairs output
    file handles:

    * use first row as header, then write the first row
    * write the rest of the rows

    If there are no rows, this will throw an StopIteration
    exception
    """
    # first row
    p_row0, r_row0 = gen.next()
    p_writer = mk_csv_writer(p_row0, p_ofile)
    r_writer = mk_csv_writer(r_row0, r_ofile)
    p_writer.writerow(p_row0)
    r_writer.writerow(r_row0)
    # now the rest of them
    for p_row, r_row in gen:
        p_writer.writerow(p_row)
        r_writer.writerow(r_row)


def main_corpus_pairs(args):
    """
    The usual main. Extract feature vectors from the corpus
    """
    inputs = features.read_corpus_inputs(args)
    of_bn = os.path.join(args.output, os.path.basename(args.corpus))
    of_ext = '.csv'
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    relations_file = of_bn + '.relations' + of_ext
    edu_pairs_file = of_bn + '.edu-pairs' + of_ext
    with codecs.open(relations_file, 'wb') as r_ofile:
        with codecs.open(edu_pairs_file, 'wb') as p_ofile:
            gen = features.extract_pair_features(inputs)
            try:
                _write_pairs(gen, r_ofile, p_ofile)
            except StopIteration:
                # FIXME: I have a nagging feeling that we should properly
                # support this by just printing a CSV header and nothing
                # else, but I'm trying to minimise code paths and for now
                # failing in this corner case feels like a lesser evil :-/
                sys.exit("No features to extract!")


def main(args):
    "main for feature extraction mode"

    if args.parsing:
        main_parsing_pairs(args)
    else:
        main_corpus_pairs(args)
