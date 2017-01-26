# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""Count and display subgraphs matching shapes of interest

"""

from __future__ import print_function
from collections import Counter

from educe.util import (add_corpus_filters, concat_l, fields_without)
import educe.stac.graph as stacgraph

from ..args import (get_output_dir, read_corpus)
from ..output import write_dot_graph


NAME = 'count-shapes'


def _outgoing(gra, node):
    """Return the outgoing edges for a given node"""
    return [e for e in gra.links(node)
            if gra.is_relation(e) and gra.rel_links(e)[0] == node]


def _maybe_lozenge(gra, node):
    """Return (if applicable) lozenge nodes/edges starting from the given node

    If the given node looks like the start of a lozenge, return all of
    the nodes participating in the lozenge. If not, return None

    Parameters
    ----------
    gra: educe.stac.graph.Graph

    node: string
        hypergraph node name

    Returns
    -------
    nodes: set(string)
        nodes in lozenge (if whole tuple not None)

    edges:
        edges in lozenge (if whole tuple not None)
    """
    top = [node]
    top_out = _outgoing(gra, node)
    if len(top_out) < 2:
        return None
    mid = [gra.rel_links(e)[1] for e in top_out]
    if len(mid) != len(set(mid)):
        # must all point to different nodes
        return None
    mid_outs = [_outgoing(gra, m) for m in mid]
    # for each mid point: find the set of bottoms it points
    # to (we're happy if non-empty intersection)
    bots = [frozenset(gra.rel_links(e)[1]
                      for e in es)
            for es in mid_outs]
    bot = bots[0]
    for cand in bots[1:]:
        bot &= cand
    if len(bot) < 1:
        # no intersection
        return None
    loz_nodes = frozenset(top + mid) | bot
    loz_edges = frozenset(top_out + concat_l(mid_outs))
    return loz_nodes, loz_edges


def _main_lozenge_graph(args):
    """Display any lozenge shaped subgraphs
    """
    corpus = read_corpus(args,
                         preselected={'stage': ['discourse', 'units']})
    output_dir = get_output_dir(args)
    keys = [k for k in corpus if k.stage == 'discourse']

    loz_count = Counter()
    loz_edges = Counter()
    for key in sorted(keys):
        gra = stacgraph.Graph.from_doc(corpus, key)
        if args.strip_cdus:
            gra = gra.without_cdus(sloppy=True)
        interesting = set()
        for node in gra.nodes():
            mloz = _maybe_lozenge(gra, node)
            if mloz is not None:
                l_n, l_e = mloz
                loz_count[key] += 1
                loz_edges[key] += len(l_e)
                interesting |= l_n
        gra = gra.copy(interesting)
        dot_gra = stacgraph.DotGraph(gra)
        if dot_gra.get_nodes():
            write_dot_graph(key, output_dir, dot_gra,
                            run_graphviz=args.draw)
    for key in sorted(loz_count):
        print(key, loz_count[key], '({})'.format(loz_edges[key]))
    print('TOTAL lozenges:', sum(loz_count.values()))
    print('TOTAL edges in lozenges:', sum(loz_edges.values()))

# ---------------------------------------------------------------------
# args
# ---------------------------------------------------------------------


def config_argparser(psr):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    # note: not the usual input args
    psr.add_argument('corpus', metavar='DIR', help='corpus dir')
    psr.add_argument('--output', metavar='DIR', required=True,
                     help='output  dir')
    psr.add_argument('--no-draw', action='store_false',
                     dest='draw',
                     default=True,
                     help='Do not actually draw the graph')
    psr.add_argument('--strip-cdus', action='store_true',
                     help='Strip away CDUs (substitute w heads)')

    educe_group = psr.add_argument_group('corpus filtering arguments')
    add_corpus_filters(educe_group, fields=fields_without(["stage"]))
    psr.set_defaults(func=main)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    if True:  # may one day want to select diff shapes
        _main_lozenge_graph(args)

# vim: syntax=python:
    #
