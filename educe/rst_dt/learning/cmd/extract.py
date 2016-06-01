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

from educe.learning.edu_input_format import (dump_all,
                                             load_labels)
from educe.learning.vocabulary_format import (dump_vocabulary,
                                              load_vocabulary)
from ..args import add_usual_input_args
from ..doc_vectorizer import DocumentCountVectorizer, DocumentLabelExtractor
from educe.rst_dt.corpus import RstDtParser
from educe.rst_dt.ptb import PtbParser
from educe.rst_dt.corenlp import CoreNlpParser


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
    # TODO make optional and possibly exclusive from corenlp below
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
    parser.add_argument('--vocabulary',
                        metavar='FILE',
                        help='Use given vocabulary for feature output '
                        '(when extracting test data, you may want to '
                        'use the feature vocabulary from the training '
                        'set ')
    # labels
    # TODO restructure ; the aim is to have three options:
    # * fine-grained labelset (no transformation from treebank),
    # * coarse-grained labelset (mapped from fine-grained),
    # * manually specified list of labels (important here is the order
    # of the labels, that implicitly maps labels as strings to integers)
    # ... but what to do is not 100% clear right now
    parser.add_argument('--labels',
                        metavar='FILE',
                        help='Read label set from given feature file '
                        '(important when extracting test data)')
    parser.add_argument('--coarse',
                        action='store_true',
                        help='use coarse-grained labels')
    parser.add_argument('--fix_pseudo_rels',
                        action='store_true',
                        help='fix pseudo-relation labels')
    # NEW use CoreNLP's output for tokenization and syntax (+coref?)
    parser.add_argument('--corenlp_out_dir', metavar='DIR',
                        help='CoreNLP output directory')
    # end NEW
    # NEW lecsie features
    parser.add_argument('--lecsie_data_dir', metavar='DIR',
                        help='LECSIE features directory')
    # end NEW

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

    # NEW lecsie features
    lecsie_data_dir = args.lecsie_data_dir

    # RST data
    # fileX docs are currently not supported by CoreNLP
    exclude_file_docs = args.corenlp_out_dir

    rst_reader = RstDtParser(args.corpus, args,
                             coarse_rels=args.coarse,
                             fix_pseudo_rels=args.fix_pseudo_rels,
                             exclude_file_docs=exclude_file_docs)
    rst_corpus = rst_reader.corpus
    # TODO: change educe.corpus.Reader.slurp*() so that they return an object
    # which contains a *list* of FileIds and a *list* of annotations
    # (see sklearn's Bunch)
    # on creation of these lists, one can impose the list of names to be
    # sorted so that the order in which docs are iterated is guaranteed
    # to be always the same

    # syntactic preprocessing
    if args.corenlp_out_dir:
        # get the precise path to CoreNLP parses for the corpus currently used
        # the folder layout of CoreNLP's output currently follows that of the
        # corpus: RSTtrees-main-1.0/{TRAINING,TEST}, RSTtrees-double-1.0
        # FIXME clean rewrite ; this could mean better modelling of the corpus
        # subparts/versions, e.g. RST corpus have "version: 1.0", annotators
        # "main" or "double"

        # find the suffix of the path name that starts with RSTtrees-*
        # FIXME find a cleaner way to do this ;
        # should probably use pathlib, included in the standard lib
        # for python >= 3.4
        try:
            rel_idx = (args.corpus).index('RSTtrees-WSJ-')
        except ValueError:
            # if no part of the path starts with "RSTtrees", keep the
            # entire path (no idea whether this is good)
            relative_corpus_path = args.corpus
        else:
            relative_corpus_path = args.corpus[rel_idx:]

        corenlp_out_dir = os.path.join(args.corenlp_out_dir,
                                       relative_corpus_path)
        csyn_parser = CoreNlpParser(corenlp_out_dir)
    else:
        # TODO improve switch between gold and predicted syntax
        # PTB data
        csyn_parser = PtbParser(args.ptb)
    # FIXME
    print('offline syntactic preprocessing: ready')

    # align EDUs with sentences, tokens and trees from PTB
    def open_plus(doc):
        """Open and fully load a document

        doc is an educe.corpus.FileId
        """
        # create a DocumentPlus
        doc = rst_reader.decode(doc)
        # populate it with layers of info
        # tokens
        doc = csyn_parser.tokenize(doc)
        # syn parses
        doc = csyn_parser.parse(doc)
        # disc segments
        doc = rst_reader.segment(doc)
        # disc parse
        doc = rst_reader.parse(doc)
        # pre-compute the relevant info for each EDU
        doc = doc.align_with_doc_structure()
        # logical order is align with tokens, then align with trees
        # but aligning with trees first for the PTB enables
        # to get proper sentence segmentation
        doc = doc.align_with_trees()
        doc = doc.align_with_tokens()
        # dummy, fallback tokenization if there is no PTB gold or silver
        doc = doc.align_with_raw_words()

        return doc

    # generate DocumentPluses
    # TODO remove sorted() once educe.corpus.Reader is able
    # to iterate over a stable (sorted) list of FileIds
    docs = [open_plus(doc) for doc in sorted(rst_corpus)]
    # instance generator
    instance_generator = lambda doc: doc.all_edu_pairs()
    split_feat_space = 'dir_sent'
    # extract vectorized samples
    if args.vocabulary is not None:
        vocab = load_vocabulary(args.vocabulary)
        vzer = DocumentCountVectorizer(instance_generator,
                                       feature_set,
                                       lecsie_data_dir=lecsie_data_dir,
                                       vocabulary=vocab,
                                       split_feat_space=split_feat_space)
        X_gen = vzer.transform(docs)
    else:
        vzer = DocumentCountVectorizer(instance_generator,
                                       feature_set,
                                       lecsie_data_dir=lecsie_data_dir,
                                       min_df=5,
                                       split_feat_space=split_feat_space)
        X_gen = vzer.fit_transform(docs)

    # extract class label for each instance
    if live:
        y_gen = itertools.repeat(0)
    elif args.labels is not None:
        labelset = load_labels(args.labels)
        labtor = DocumentLabelExtractor(instance_generator,
                                        labelset=labelset)
        labtor.fit(docs)
        y_gen = labtor.transform(docs)
    else:
        labtor = DocumentLabelExtractor(instance_generator)
        # y_gen = labtor.fit_transform(rst_corpus)
        # fit then transform enables to get classes_ for the dump
        labtor.fit(docs)
        y_gen = labtor.transform(docs)

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
    # dump EDUs and features in svmlight format
    dump_all(X_gen, y_gen, out_file, labtor.labelset_, docs,
             instance_generator)
    # dump vocabulary
    vocab_file = out_file + '.vocab'
    dump_vocabulary(vzer.vocabulary_, vocab_file)
