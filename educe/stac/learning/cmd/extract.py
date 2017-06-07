#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: BSD3

"""
Extract features to CSV files
"""

from __future__ import print_function
import os
from os import path as fp
import sys

import educe.corpus
import educe.glozz
from educe.learning.edu_input_format import (dump_all,
                                             labels_comment,
                                             dump_svmlight_file,
                                             dump_edu_input_file)
from educe.learning.keygroup_vectorizer import (KeyGroupVectorizer)
from educe.learning.vocabulary_format import (dump_vocabulary,
                                              load_vocabulary)
import educe.stac
from educe.stac.annotation import (DIALOGUE_ACTS,
                                   SUBORDINATING_RELATIONS,
                                   COORDINATING_RELATIONS)
from educe.stac.learning.doc_vectorizer import (
    DialogueActVectorizer, LabelVectorizer)
from educe.stac.learning.features import (
    extract_pair_features, extract_single_features,
    mk_high_level_dialogues, read_corpus_inputs, strip_cdus)
import educe.util


NAME = 'extract'


# ----------------------------------------------------------------------
# options
# ----------------------------------------------------------------------


def config_argparser(parser):
    """
    Subcommand flags.
    """
    parser.add_argument('corpus', metavar='DIR',
                        help='Corpus dir (eg. data/pilot)')
    parser.add_argument('resources', metavar='DIR',
                        help='Resource dir (eg. data/resource)')
    parser.add_argument('output', metavar='DIR',
                        help='Output directory')
    # add flags --doc, --subdoc, etc to allow user to filter on these things
    educe.util.add_corpus_filters(parser,
                                  fields=['doc', 'subdoc', 'annotator'])
    parser.add_argument('--verbose', '-v', action='count',
                        default=1)
    parser.add_argument('--quiet', '-q', action='store_const',
                        const=0,
                        dest='verbose')
    parser.add_argument('--single', action='store_true',
                        help="Features for single EDUs (instead of pairs)")
    parser.add_argument('--parsing', action='store_true',
                        help='Extract features for parsing')
    parser.add_argument('--vocabulary',
                        metavar='FILE',
                        help='Vocabulary file (for --parsing mode)')
    parser.add_argument('--ignore-cdus', action='store_true',
                        help='Avoid going into CDUs')
    parser.add_argument('--strip-mode',
                        choices=['head', 'broadcast', 'custom'],
                        default='head',
                        help='CDUs stripping method (if going into CDUs)')
    parser.set_defaults(func=main)

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main_single(args):
    """Extract feature vectors for single EDUs in the corpus."""
    inputs = read_corpus_inputs(args)
    stage = 'unannotated' if args.parsing else 'units'
    dialogues = list(mk_high_level_dialogues(inputs, stage))
    instance_generator = lambda x: x.edus[1:]  # drop fake root

    # pylint: disable=invalid-name
    # scikit-convention
    feats = extract_single_features(inputs, stage)
    vzer = KeyGroupVectorizer()
    # TODO? just transform() if args.parsing or args.vocabulary?
    X_gen = vzer.fit_transform(feats)
    # pylint: enable=invalid-name
    labels = DIALOGUE_ACTS
    labtor = DialogueActVectorizer(instance_generator, labels)
    y_gen = labtor.transform(dialogues)

    # create directory structure: {output}/
    outdir = args.output
    if not fp.exists(outdir):
        os.makedirs(outdir)

    corpus_name = fp.basename(args.corpus)

    # list dialogue acts
    comment = labels_comment(labtor.labelset_)

    # dump: EDUs, pairings, vectorized pairings with label
    # these paths should go away once we switch to a proper dumper
    out_file = fp.join(
        outdir,
        '{corpus_name}.dialogue-acts.sparse'.format(
            corpus_name=corpus_name))
    edu_input_file = '{out_file}.edu_input'.format(out_file=out_file)
    dump_edu_input_file(dialogues, edu_input_file)
    dump_svmlight_file(X_gen, y_gen, out_file, comment=comment)

    # dump vocabulary
    vocab_file = fp.join(outdir,
                         '{corpus_name}.dialogue-acts.sparse.vocab'.format(
                             corpus_name=corpus_name))
    dump_vocabulary(vzer.vocabulary_, vocab_file)


def main_pairs(args):
    """Extract feature vectors for pairs of EDUs in the corpus."""
    inputs = read_corpus_inputs(args)
    stage = 'units' if args.parsing else 'discourse'
    dialogues = list(mk_high_level_dialogues(inputs, stage))
    instance_generator = lambda x: x.edu_pairs()

    labels = frozenset(SUBORDINATING_RELATIONS +
                       COORDINATING_RELATIONS)

    # pylint: disable=invalid-name
    # X, y follow the naming convention in sklearn
    feats = extract_pair_features(inputs, stage)
    vzer = KeyGroupVectorizer()
    if args.parsing or args.vocabulary:
        vzer.vocabulary_ = load_vocabulary(args.vocabulary)
        X_gen = vzer.transform(feats)
    else:
        X_gen = vzer.fit_transform(feats)
    # pylint: enable=invalid-name
    labtor = LabelVectorizer(instance_generator, labels,
                             zero=args.parsing)
    y_gen = labtor.transform(dialogues)

    # create directory structure
    outdir = args.output
    if not fp.exists(outdir):
        os.makedirs(outdir)

    corpus_name = fp.basename(args.corpus)
    # these paths should go away once we switch to a proper dumper
    out_file = fp.join(
        outdir,
        '{corpus_name}.relations.sparse'.format(
            corpus_name=corpus_name))

    dump_all(X_gen, y_gen, out_file, labtor.labelset_, dialogues,
             instance_generator)
    # dump vocabulary
    vocab_file = fp.join(outdir,
                         '{corpus_name}.relations.sparse.vocab'.format(
                             corpus_name=corpus_name))
    dump_vocabulary(vzer.vocabulary_, vocab_file)


def main(args):
    "main for feature extraction mode"

    if args.parsing and not args.vocabulary:
        sys.exit("Need --vocabulary if --parsing is enabled")
    if args.parsing and args.single:
        sys.exit("Can't mixing --parsing and --single")
    elif args.single:
        main_single(args)
    else:
        main_pairs(args)
