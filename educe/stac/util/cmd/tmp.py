# Author:
# License: CeCILL-B (French BSD3-like)

"""
Expiremental sandbox (ignore)
"""

from __future__ import print_function
from ..args import\
    (add_usual_input_args, add_usual_output_args,
     read_corpus)


def config_argparser(psr):
    """
    Subcommand flags.
    """
    add_usual_input_args(psr)
    add_usual_output_args(psr)
    psr.set_defaults(func=main)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    corpus = read_corpus(args)
    for key in corpus:
        print(corpus[key])
