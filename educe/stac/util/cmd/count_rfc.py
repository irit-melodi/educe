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

import educe.stac.graph as graph
from educe.util import (
    add_corpus_filters, fields_without)
from educe.stac.rfc import BasicRfc
from ..args import (
    add_usual_input_args, add_usual_output_args,
    read_corpus,
    get_output_dir, announce_output_dir,
    anno_id)
from ..glozz import anno_id_to_tuple
from ..output import save_document

NAME = 'count-rfc'

# Total
# Violation by algo
# Split 

class DummyRfc:
    def __init__(self, graph):
        self.graph = graph
        
    def violations(self):
        return dict(x=self.graph.relations())

rfc_methods = (
    ('total', DummyRfc),
    ('basic', BasicRfc)
    )

def process_doc(corpus, key):
    """ Tests document against RFC definitions.

    Returns dict of method:Counter """
    res = Counter()
    dgraph = graph.Graph.from_doc(corpus, key)
    relations = dgraph.relations()
    # for rel in relations:
        # res['total', dgraph.annotation(rel).type] += 1

    for name, method in rfc_methods:
        violations = method(dgraph).violations()
        v_rels = [dgraph.annotation(n)
            for ns in violations.values()
            for n in ns]
        for rel in v_rels:
            is_forward = rel.source.text_span() <= rel.target.text_span()
            for label in ('Both', 'Forwards' if is_forward else 'Backwards'):
                res[(name, label, 'TOTAL')] += 1
                res[(name, label, rel.type)] += 1
    return res

def display(res):
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
    # print(res)

def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    parser.add_argument('corpus', metavar='DIR',
                        nargs='?',
                        help='corpus dir')
    add_corpus_filters(parser, fields=fields_without(["stage"]))
    add_usual_output_args(parser)
    parser.set_defaults(func=main)

def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    output_dir = get_output_dir(args)
    corpus = read_corpus(args, verbose=True,
        preselected=dict(stage=['discourse']))

    res = Counter()
    for key in corpus:
        print(key)
        # doc = corpus[key]
        part_res = process_doc(corpus, key)
        res.update(part_res.elements())

    # Printing res...
    display(res)
    
    # announce_output_dir(output_dir)
