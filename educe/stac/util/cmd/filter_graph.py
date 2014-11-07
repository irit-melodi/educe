# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""
Pick out interesting subgraphs
"""

from __future__ import print_function
import sys

from educe import graph
import educe.corpus
import educe.stac
import educe.stac.graph as stacgraph

from ..args import\
    get_output_dir, read_corpus
from ..output import write_dot_graph


def _keep(doc, desired):
    """
    Return a predicate that selects relations with desired types
    and any edus/cdus they connect.
    """
    whitelist = []
    relations = [x for x in doc.relations if x.type in desired]
    excluded = [x for x in doc.relations if x.type not in desired]
    endpoints = [x.source for x in relations] + [x.target for x in relations]
    whitelist.extend(relations)

    def add_endpoint(epoint):
        """
        add this endpoint and if cdu recursively anything inside it
        along with any relations whose endpoints are contained
        within
        """
        whitelist.append(epoint)
        if educe.stac.is_cdu(epoint):
            for anno in epoint.members:
                add_endpoint(anno)
                for rel in excluded:
                    if anno in [rel.source, rel.target]:
                        whitelist.append(rel)
                        if rel.source not in whitelist:
                            add_endpoint(rel.source)
                        if rel.target not in whitelist:
                            add_endpoint(rel.target)

    for epoint in endpoints:
        add_endpoint(epoint)

    def _inner(anno):
        "the actual filter"
        return anno in whitelist

    return _inner

# ---------------------------------------------------------------------
# args
# ---------------------------------------------------------------------

NAME = 'filter-graph'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    # note: not the usual input args
    parser.add_argument('corpus', metavar='DIR', help='corpus dir')
    parser.add_argument('rel_types', metavar='REL_LABEL', nargs='+',
                        help='relation labels to keep')
    parser.add_argument('--output', metavar='DIR', required=True,
                        help='output  dir')
    parser.add_argument('--no-draw', action='store_false',
                        dest='draw',
                        default=True,
                        help='Do not actually draw the graph')

    educe_group = parser.add_argument_group('corpus filtering arguments')
    # doesn't make sense to filter on stage for graphs
    educe.util.add_corpus_filters(educe_group,
                                  fields=['doc', 'subdoc', 'annotator'])
    parser.set_defaults(func=main)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    args.stage = 'discourse|units'
    corpus = read_corpus(args, verbose=True)
    output_dir = get_output_dir(args)

    keys = [k for k in corpus if k.stage == 'discourse']
    for k in sorted(keys):
        try:
            gra = stacgraph.Graph.from_doc(corpus, k,
                                           pred=_keep(corpus[k],
                                                      args.rel_types))
            dot_gra = stacgraph.DotGraph(gra)
            if dot_gra.get_nodes():
                write_dot_graph(k, output_dir, dot_gra,
                                run_graphviz=args.draw)
            else:
                print("Skipping %s (empty graph)" % k, file=sys.stderr)
        except graph.DuplicateIdException:
            warning = "WARNING: %s has duplicate annotation ids" % k
            print(warning, file=sys.stderr)

# vim: syntax=python:
