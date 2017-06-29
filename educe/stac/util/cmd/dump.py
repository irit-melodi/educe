"""Produce a (non-XML) dump of the corpus"""

import re
import sys

from ..args import (
    add_usual_input_args, add_usual_output_args, read_corpus,
    get_output_dir, announce_output_dir, anno_id
)
from ..situated_stats import (
    dump_corpus_dataframes, read_corpus_as_dataframes
)

NAME = 'dump'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser)
    add_usual_output_args(parser)
    # * args of read_corpus_as_dataframes()
    # add_usual_input_args() already adds 'corpus' which we'll reuse for
    # 'stac_data_dir'
    parser.add_argument('--corpus_version',
                        choices=['ling', 'situated'],
                        default='situated',
                        help='version of the corpus')
    parser.add_argument('--split',
                        choices=['all', 'train', 'test'],
                        default='all',
                        help='split to include')
    parser.add_argument('--strip_cdus',
                        action='store_true',
                        help="Strip CDUs (sloppy=True, mode='head')")
    parser.add_argument('--attach_len',
                        action='store_true',
                        help='Compute attachment length for discourse rels'
                        '(currently requires --strip_cdus)')
    # * args of dump_corpus_dataframes()
    # add_usual_output_args() already adds '--output', which we'll reuse
    # for 'out_dir'
    parser.add_argument('--out_fmt',
                        choices=['csv', 'pickle'],
                        default='csv',
                        help='Output format for the dump')
    parser.set_defaults(func=main)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    output_dir = get_output_dir(args)
    corpus_dfs = read_corpus_as_dataframes(
        args.corpus, version=args.corpus_version, split=args.split,
        strip_cdus=args.strip_cdus, attach_len=args.attach_len)
    dump_corpus_dataframes(corpus_dfs, args.output, out_fmt=args.out_fmt)
