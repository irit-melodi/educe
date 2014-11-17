#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""
Guess the addresee for each EDU
"""

from __future__ import print_function

from educe.stac import postag, corenlp
from educe.stac.annotation import is_edu
from educe.util import\
    add_corpus_filters, fields_without, mk_is_interesting
import educe.corpus
import educe.glozz
import educe.learning.keys
import educe.stac
import educe.util

from ..features import\
    mk_env, get_players,\
    FeatureInput
from ..addressee import guess_addressees

NAME = 'addressee'


def _on_doc(inputs, people, key, live):
    "compute all EDU addresees for a document"
    env = mk_env(inputs, people, key, live)
    doc = inputs.corpus[key]
    print()
    print(key)
    print("=" * len(str(key)))
    print("Players", list(env.current.players))
    print()
    for edu in filter(is_edu, doc.units):
        txt = doc.text(edu.text_span())
        addressees = guess_addressees(env.current, edu)
        addressees_str = "unknown" if addressees is None\
            else ";".join(addressees)
        msg = u'{addr:14} {txt}'.format(addr=addressees_str,
                                        txt=txt)
        print(msg.encode('utf-8'))


# ----------------------------------------------------------------------
# options
# ----------------------------------------------------------------------


def config_argparser(parser):
    """
    Subcommand flags.
    """
    parser.add_argument('corpus', metavar='DIR',
                        help='Corpus dir (eg. data/pilot)')
    parser.add_argument('--output', metavar='FILE',
                        help='Output file')
    add_corpus_filters(parser, fields=fields_without(["stage"]))
    parser.set_defaults(func=main)

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def _read_corpus_inputs(args):
    """
    Read and filter the part of the corpus we want features for
    """
    is_interesting = mk_is_interesting(args,
                                       preselected={"stage": ["units"]})
    reader = educe.stac.Reader(args.corpus)
    anno_files = reader.filter(reader.files(), is_interesting)
    corpus = reader.slurp(anno_files, verbose=True)

    postags = postag.read_tags(corpus, args.corpus)
    parses = corenlp.read_results(corpus, args.corpus)
    return FeatureInput(corpus, postags, parses,
                        [], None, None, None,
                        True, True, True)


def main(args):
    """
    The usual main. Extract feature vectors from the corpus
    (single edus only)
    """
    inputs = _read_corpus_inputs(args)
    players = get_players(inputs)
    for key in inputs.corpus:
        _on_doc(inputs, players, key, False)
