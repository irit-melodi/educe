# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
One-off experiments
"""

from __future__ import print_function
import collections

from ..args import\
    add_usual_input_args,\
    read_corpus

NAME = 'tmp'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser)
    parser.set_defaults(func=main)

def gorn_depths(arg):
    return [len(x) for x in arg.gorn]


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    corpus = read_corpus(args)
    d = collections.defaultdict(int)
    for k in sorted(corpus):
        print("--------------------" * 3)
        print("doc:", k.doc)
        print("--------------------" * 3)
        print()
        for rel in corpus[k]:
            #if (len(rel.arg1.span) > 2 or len(rel.arg2.span) > 2):
            #    print(unicode(rel).encode('utf-8'))
            #    print()
            print(gorn_depths(rel.arg1))
            print(gorn_depths(rel.arg2))
