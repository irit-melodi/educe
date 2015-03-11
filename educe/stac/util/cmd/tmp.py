# Author:
# License: CeCILL-B (French BSD3-like)

"""
Expiremental sandbox (ignore)
"""

from __future__ import print_function
from os import path as fp

import educe.stac
from ..args import\
    (add_usual_input_args, add_usual_output_args,
     read_corpus)
from ..context import (enclosed)


def friendly_dialogue_id(k, span):
    """
    Dialogue identifier which may be easier to understand when debugging
    the feature vector (based on its text span).

    The regular timestamp based identifiers look too much like each other.
    """
    bname = fp.basename(educe.stac.id_to_path(k))
    start = span.char_start
    end = span.char_end
    return '%s_%04d_%04d' % (bname, start, end)


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
        doc = corpus[key]
        dialogues = [x for x in doc.units if educe.stac.is_dialogue(x)]
        edus = [x for x in doc.units if educe.stac.is_edu(x)]
        for anno in dialogues:
            dspan = anno.text_span()
            edus_within = enclosed(dspan, edus)
            cols = [friendly_dialogue_id(key, dspan),
                    anno.local_id(),
                    len(edus_within)]
            print('\t'.join(map(str, cols)))
