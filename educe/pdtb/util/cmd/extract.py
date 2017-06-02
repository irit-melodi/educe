# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Extract features

2017-01-27 this code is broken ; it relies on stac.keys.KeyGroupWriter
which was deprecated and removed a while back (MM to self: way to go!).
"""

import codecs
import csv
import os

import stac.util.stac_csv_format
import stac.keys

from ..args import\
    add_usual_input_args, add_usual_output_args,\
    read_corpus, get_output_dir, announce_output_dir
from ..features import\
    extract_rel_features, FeatureInput, RelKeys

NAME = 'extract'


def mk_csv_writer(keys, fstream):
    """
    start off csv writer for a given mode
    """
    csv_quoting = csv.QUOTE_MINIMAL
    writer = stac.keys.KeyGroupWriter(fstream, keys, quoting=csv_quoting)
    writer.writeheader()
    return writer


def read_corpus_inputs(args):
    """
    Read the data needed to read features from the PDTB corpus
    """
    corpus = read_corpus(args)
    return FeatureInput(corpus=corpus,
                        debug=args.debug)


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser)
    add_usual_output_args(parser)
    parser.add_argument('--debug', action='store_true',
                        help='Emit fields used for debugging purposes')
    parser.set_defaults(func=main)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    odir = get_output_dir(args)
    inputs = read_corpus_inputs(args)
    header = RelKeys(inputs)

    of_bn = os.path.join(odir, os.path.basename(args.corpus))
    relations_file = of_bn + '.relations.csv'
    with codecs.open(relations_file, 'wb') as r_ofile:
        r_writer = mk_csv_writer(header, r_ofile)
        for r_row in extract_rel_features(inputs):
            r_writer.writerow(r_row)
    announce_output_dir(odir)
