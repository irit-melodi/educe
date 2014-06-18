# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
One-off experiments
"""

from __future__ import print_function
import collections
import sys

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

def sentence_nums(arg):
    return [x.parts[0] for x in arg.gorn]

def is_multisentential(args):
    return len(frozenset(sentence_nums(args))) > 1

def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    corpus = read_corpus(args)
    counts = collections.defaultdict(int)
    total = 0
    for k in sorted(corpus):
        print("--------------------" * 3)
        print("doc:", k.doc)
        print("--------------------" * 3)
        print()
        for rel in corpus[k]:
            #if (len(rel.arg1.span) > 2 or len(rel.arg2.span) > 2):
            #    print(unicode(rel).encode('utf-8'))
            #    print()
            if is_multisentential(rel.arg1):
                counts[k] += 1
            if is_multisentential(rel.arg2):
                counts[k] += 1
            if is_multisentential(rel.arg1) or is_multisentential(rel.arg2):
                print(rel)

        total += counts[k]
    for k in sorted(corpus):
        print(k.doc, counts[k], "multisentential")
    print("total", total, "multisentential")

