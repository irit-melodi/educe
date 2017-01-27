# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Experimental noodling (ignore)
"""

from __future__ import print_function

from educe.internalutil import treenode
from ..args import\
    add_usual_input_args, add_usual_output_args,\
    read_corpus, get_output_dir, announce_output_dir

NAME = 'tmp'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser)
    add_usual_output_args(parser)
    parser.set_defaults(func=main)


def dump(corpus, odir):
    """
    Dump a text file for every RST tree in the corpus
    """
    for k in corpus:
        ctx = treenode(corpus[k]).context
        for para in ctx.paragraphs:
            print("PARA", ctx.text(para.text_span()))


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    odir = get_output_dir(args)
    corpus = read_corpus(args)
    dump(corpus, odir)
    announce_output_dir(odir)
