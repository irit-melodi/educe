# Author: Jeremy Perret
# License: CeCILL-B (French BSD3-like)

"""
Count RFC violations
"""
from __future__ import print_function

from tabulate import tabulate
from collections import defaultdict, Counter

import educe.stac
import educe.stac.context as context
from educe.stac.graph import Graph
from educe.stac.rfc import BasicRfc, ThreadedRfc
from educe.util import add_corpus_filters, fields_without
from ..args import add_usual_output_args, read_corpus

NAME = 'count-rfc'


class DummyRfc:
    def __init__(self, graph):
        self.graph = graph

    def violations(self):
        return self.graph.relations()


RFC_METHODS = (
    ('total', DummyRfc),
    ('basic', BasicRfc),
    ('mlast', ThreadedRfc)     # Multiple lasts (one for each speaker)
    )


def process_doc_violations(corpus, key, strip=False):
    """ Tests document against RFC definitions.

    Returns dict of method:Counter """
    res = Counter()
    dgraph = Graph.from_doc(corpus, key)
    if strip:
        dgraph.strip_cdus(sloppy=True)
    relations = dgraph.relations()

    for name, method in RFC_METHODS:
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
    doc_graph = Graph.from_doc(corpus, key)
    if strip:
        doc_graph.strip_cdus(sloppy=True)
    anno_to_nodes = dict((doc_graph.annotation(n), n)
                         for n in doc_graph.edus())
    doc = corpus[key]
    # Computing list of EDUs for each dialogue
    ctxs = context.Context.for_edus(doc)
    dia_edus = defaultdict(list)
    for u in doc.units:
        if educe.stac.is_edu(u):
            dia_edus[ctxs[u].dialogue].append(u)
    for dia, edus in dia_edus.items():
        for i in range(len(edus)):
            res[('dia', i+1)] += 1
        dia_edu_nodes = list(anno_to_nodes[edu] for edu in edus)
        dia_graph = doc_graph.copy(dia_edu_nodes)
        sorted_nodes = dia_graph.first_outermost_dus()
        sorted_edus = [n for n in sorted_nodes if n in dia_edu_nodes]
        for name, method in RFC_METHODS[1:]:
            rfc = method(dia_graph)
            for i, last in enumerate(sorted_edus):
                frontier = rfc._build_frontier(last)
                frontier = list(n for n in frontier if dia_graph.is_edu(n))
                # Corner case: backwards links
                frontier = list(n for n in frontier if
                                (dia_graph.annotation(n).text_span() <=
                                 dia_graph.annotation(last).text_span()))
                res[(name, i+1)] += len(frontier)
                if len(frontier) > i+1:
                    print(i+1, len(frontier), frontier)
                assert len(frontier) <= i+1
    return res


def display_violations(res):
    """ Display results for violation count """
    table_names = ('Both', 'Forwards', 'Backwards')
    col_names = list(n for n, _ in RFC_METHODS)
    col_0 = col_names[0]
    row_names = sorted(set(n for _, _, n in res),
                       key=lambda x: res[(col_0, 'Both', x)],
                       reverse=True)

    for table_name in table_names:
        tres = list()
        for row_name in row_names:
            tres.append([row_name]
                        + list(res[(col_name, table_name, row_name)]
                               for col_name in col_names))
        print(tabulate(tres, headers=[table_name]+col_names)+'\n')


def display_power(res):
    """ Display results for RFC filtering power

    res is a Counter[(nb_edus, method)]"""
    methods = list(n for n, _ in RFC_METHODS)[1:]
    col_names = ['length', 'dia']
    for method in methods:
        col_names += [method]
    tres = list()
    for nb_edus in sorted(set(i for _, i in res)):
        nb_dialogues = res[('dia', nb_edus)]
        row = [nb_edus, nb_dialogues]
        for method in methods:
            total_frontier_size = res[(method, nb_edus)]
            avg_frontier_size = float(total_frontier_size)/nb_dialogues
            rfc_power = (100 * avg_frontier_size) / nb_edus
            row.append(rfc_power)
        tres.append(row)
    print(tabulate(tres, headers=col_names, floatfmt='.1f')+'\n')


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
    # output_dir = get_output_dir(args)
    corpus = read_corpus(args, verbose=True,
                         preselected=dict(stage=['discourse']))

    if args.mode == 'violations':
        main_violations(corpus, strip=args.strip_cdus)
    elif args.mode == 'power':
        main_power(corpus, strip=args.strip_cdus)

    # announce_output_dir(output_dir)
