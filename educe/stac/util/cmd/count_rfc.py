# Author: Jeremy Perret
# License: CeCILL-B (French BSD3-like)

"""
Count RFC violations
"""
from __future__ import print_function

import re
import sys
from tabulate import tabulate
from collections import defaultdict, Counter

import educe.stac
import educe.stac.util.context as context
import educe.stac.graph as graph
from educe.util import (
    add_corpus_filters, fields_without)
from educe.stac.rfc import BasicRfc, ThreadedRfc
from ..args import (
    add_usual_input_args, add_usual_output_args,
    read_corpus,
    get_output_dir, announce_output_dir,
    anno_id)
from ..glozz import anno_id_to_tuple
from ..output import save_document

NAME = 'count-rfc'

class DummyRfc:
    def __init__(self, graph):
        self.graph = graph
        
    def violations(self):
        return self.graph.relations()

rfc_methods = (
    ('total', DummyRfc),
    ('basic', BasicRfc),
    ('mlast', ThreadedRfc)     # Multiple lasts (one for each speaker)
    )

def process_doc_violations(corpus, key, strip=False):
    """ Tests document against RFC definitions.

    Returns dict of method:Counter """
    res = Counter()
    dgraph = graph.Graph.from_doc(corpus, key)
    if strip:
        dgraph.strip_cdus(sloppy=True)
    relations = dgraph.relations()

    for name, method in rfc_methods:
        v_rels = [dgraph.annotation(n)
            for n in method(dgraph).violations()]
        for rel in v_rels:
            is_forward = rel.source.text_span() <= rel.target.text_span()
            for label in ('Both', 'Forwards' if is_forward else 'Backwards'):
                res[(name, label, 'TOTAL')] += 1
                res[(name, label, rel.type)] += 1
    return res

def process_doc_power(corpus, key, strip=False):
    """ Computes filtering power of RFC definitions """
    res = Counter()
    dgraph = graph.Graph.from_doc(corpus, key)
    if strip:
        dgraph.strip_cdus(sloppy=True)
    doc = corpus[key]
    # Computing list of EDUs for each dialogue
    ctxs = context.Context.for_edus(doc)
    dia_edus = defaultdict(list)
    for u in doc.units:
        if educe.stac.is_edu(u):
            dia_edus[ctxs[u].dialogue].append(u)
    for dia, edus in dia_edus.items():
        for i in range(len(edus)):
            res[i+1] += 1
    return res

def display_violations(res):
    """ Display results for violation count """
    table_names = ('Both', 'Forwards', 'Backwards')
    col_names = list(n for n, _ in rfc_methods)
    col_0 = col_names[0]
    row_names = sorted(set(n for _, _, n in res),
        key=lambda x:res[(col_0, 'Both', x)], reverse=True)

    for table_name in table_names:
        tres = list()
        for row_name in row_names:
            tres.append([row_name]
                + list(res[(col_name, table_name, row_name)]
                    for col_name in col_names))
        print(tabulate(tres, headers=[table_name]+col_names)+'\n')

def display_power(res):
    col_names = ['length', 'dialogues']
    row_names = sorted(res)
    tres = list()
    for row_name in row_names:
        tres.append([row_name, res[row_name]])
    print(tabulate(tres, headers=col_names)+'\n')

def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    parser.add_argument('corpus', metavar='DIR',
                        nargs='?',
                        help='corpus dir')
    parser.add_argument('--strip-cdus', action='store_true',
                       help='remove CDUs from graphs')
    parser.add_argument('--mode', choices=['violations', 'power'],
        default='violations',
        help='count RFC violations or filtering power')
    add_corpus_filters(parser, fields=fields_without(["stage"]))
    add_usual_output_args(parser)
    parser.set_defaults(func=main)

def main_violations(corpus, strip):
    """ Main for violation counting """
    res = Counter()
    for key in corpus:
        part_res = process_doc_violations(corpus, key, strip=strip)
        res.update(part_res.elements())

    display_violations(res)

def main_power(corpus, strip):
    """ Main for filtering power computation """
    res = Counter()
    for key in corpus:
        part_res = process_doc_power(corpus, key, strip=strip)
        res.update(part_res.elements())

    display_power(res)

def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    output_dir = get_output_dir(args)
    corpus = read_corpus(args, verbose=True,
        preselected=dict(stage=['discourse']))

    if args.mode == 'violations':
        main_violations(corpus, strip=args.strip_cdus)
    elif args.mode == 'power':
        main_power(corpus, strip=args.strip_cdus)

    # announce_output_dir(output_dir)
