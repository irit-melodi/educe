# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Draw RST trees
"""

from __future__ import print_function
from os import path as fp

# pylint: disable=no-name-in-module
from nltk.draw.util import CanvasFrame
# pylint: enable=no-name-in-module
from nltk.draw import TreeWidget

from ..args import (add_usual_input_args, add_usual_output_args,
                    get_output_dir, announce_output_dir,
                    read_corpus)


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser)
    add_usual_output_args(parser)
    parser.set_defaults(func=main)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    corpus = read_corpus(args)
    odir = get_output_dir(args)
    for key in corpus:
        cframe = CanvasFrame()
        widget = TreeWidget(cframe.canvas(), corpus[key])
        cframe.add_widget(widget, 10, 10)
        ofilename = fp.join(odir, key.doc) + '.ps'
        cframe.print_to_file(ofilename)
        cframe.destroy()
    announce_output_dir(odir)
