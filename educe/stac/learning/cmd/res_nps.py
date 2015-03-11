#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: BSD3

"""
Extract 'resource NPs' from all EDUs in the documents
"""

from __future__ import print_function
from collections import defaultdict, namedtuple
from itertools import chain
import csv
import sys

from educe.stac import postag, corenlp
from educe.stac.annotation import is_edu
from educe.stac.learning import features
from educe.util import\
     add_corpus_filters, fields_without, mk_is_interesting,\
     concat, concat_l
import educe.corpus
import educe.glozz
import educe.learning.keys
import educe.stac

from ..features import\
    mk_env, get_players, enclosed_trees, is_nplike,\
    FeatureInput


NAME = 'resource-nps'

LEXICON = features.LexWrapper('domain', 'stac_domain.txt', True)


def nplike_trees(current, edu):
    "any trees within an EDU that look like nps (smallest match)"
    trees = enclosed_trees(edu.text_span(),
                           current.parses.trees)
    return concat_l(t.topdown_smallest(is_nplike)
                    for t in trees)


def _mk_lexlookup(lexicons):
    """
    [LexWrapper] -> ([Class], tree -> set(Class)

    return

    1. list of classes we search for
    2. a function that given a tree,
       returns a frozenset of classes found
    """
    subclass_words = defaultdict(list)
    for lex in lexicons:
        for cname, lclass in lex.lexicon.entries.items():
            for cl, words in lclass.subclass_to_words.items():
                subclass_words[cl].extend(words)

    def inner(tree):
        twords = [t.word.lower() for t in tree.leaves()]
        res = []
        for cl, words in subclass_words.items():
            if any(x in words for x in twords):
                res.append(cl)
        return frozenset(res)

    return subclass_words.keys(), inner


class NpItem(namedtuple('NpItem_', "edu tree resources")):
    """
    An NP and some local information
    """
    pass


def np_info(inputs, lexinfo, key, item):
    "row of interesting facts about a resource np"
    doc = inputs.corpus[key]
    tag = item.tree.label()
    span = item.tree.text_span()
    text = doc.text(span)
    fields = [text,
              tag,
              str(span.char_start),
              str(span.char_end),
              item.edu.identifier()]
    lex_classes, _ = lexinfo
    fields.extend(c if c in item.resources else "-"
                  for c in lex_classes)
    return fields


def _on_doc(inputs, lexinfo, people, key):
    "all resource nps for a document"
    env = mk_env(inputs, people, key)
    doc = inputs.corpus[key]
    results = []
    _, lex_lookup = lexinfo
    for edu in filter(is_edu, doc.units):
        trees = nplike_trees(env.current, edu)
        for tree in trees:
            found = lex_lookup(tree)
            if found:
                item = NpItem(edu, tree, found)
                info = np_info(inputs, lexinfo, key, item)
                results.append(info)
    return results


# ----------------------------------------------------------------------
# options
# ----------------------------------------------------------------------


def config_argparser(parser):
    """
    Subcommand flags.
    """
    parser.add_argument('corpus', metavar='DIR',
                        help='Corpus dir (eg. data/pilot)')
    parser.add_argument('resources', metavar='DIR',
                        help='Resource dir (eg. data/resource)')
    parser.add_argument('--output', metavar='FILE',
                        help='Output file')
    # add flags --doc, --subdoc, etc to allow user to filter on these things
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
    LEXICON.read(args.resources)
    return FeatureInput(corpus=corpus,
                        postags=postags,
                        parses=parses,
                        lexicons=[LEXICON],
                        pdtb_lex=None,
                        verbnet_entries=None,
                        inquirer_lex=None)


def _conll_writer(args):
    """
    Return iterator of rows that constitute resource nps
    """
    stream = open(args.output, 'wb')\
        if args.output else sys.stdout
    return csv.writer(stream, dialect=csv.excel_tab)


def main(args):
    """
    The usual main. Extract feature vectors from the corpus
    (single edus only)
    """
    inputs = _read_corpus_inputs(args)
    lexinfo = _mk_lexlookup(inputs.lexicons)
    players = get_players(inputs)
    rows = concat(_on_doc(inputs, lexinfo, players, key)
                  for key in inputs.corpus)

    writer = _conll_writer(args)
    for row in rows:
        writer.writerow(row)
