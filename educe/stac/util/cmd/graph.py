# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""
Visualise discourse and enclosure graphs
"""

from __future__ import print_function
import sys

from educe import graph
from educe.util import add_corpus_filters, fields_without
from educe.stac.rfc import (BasicRfc)
import educe.corpus
import educe.stac
import educe.stac.graph as stacgraph
import educe.stac.postag

from ..args import (get_output_dir, anno_id)
from ..glozz import (anno_id_from_tuple)
from ..output import write_dot_graph


# slightly different from the stock stac-util version because it
# supports live corpus mode
def _read_corpus(args):
    """
    Read and return the corpus specified by the command line arguments
    """
    is_interesting = educe.util.mk_is_interesting(args)
    if args.live:
        reader = educe.stac.LiveInputReader(args.corpus)
        anno_files = reader.files()
    else:
        reader = educe.stac.Reader(args.corpus)
        anno_files = reader.filter(reader.files(), is_interesting)
    return reader.slurp(anno_files, verbose=True)


def _main_rel_graph(args):
    """
    Draw graphs showing relation instances between EDUs
    """
    args.stage = 'discourse|units'
    corpus = _read_corpus(args)
    output_dir = get_output_dir(args)

    if args.live:
        keys = corpus
    else:
        keys = [k for k in corpus if k.stage == 'discourse']

    for k in sorted(keys):
        if args.highlight:
            highlights = [anno_id_from_tuple(x) for x in args.highlight]
            for anno in corpus[k].annotations():
                if anno.local_id() in highlights:
                    anno.features['highlight'] = 'orange'
        try:
            gra = stacgraph.Graph.from_doc(corpus, k)
            if args.strip_cdus:
                gra = gra.without_cdus()
            dot_gra = stacgraph.DotGraph(gra)
            if dot_gra.get_nodes():
                write_dot_graph(k, output_dir, dot_gra,
                                run_graphviz=args.draw)
                if args.split:
                    ccs = gra.connected_components()
                    for part, nodes in enumerate(ccs, 1):
                        gra2 = gra.copy(nodes)
                        write_dot_graph(k, output_dir,
                                        stacgraph.DotGraph(gra2),
                                        part=part,
                                        run_graphviz=args.draw)
            else:
                print("Skipping %s (empty graph)" % k, file=sys.stderr)
        except graph.DuplicateIdException:
            warning = "WARNING: %s has duplicate annotation ids" % k
            print(warning, file=sys.stderr)


def _main_rfc_graph(args):
    """
    Draw graphs showing relation instances between EDUs
    """
    args.stage = 'discourse|units'
    corpus = _read_corpus(args)
    output_dir = get_output_dir(args)

    if args.live:
        keys = corpus
    else:
        keys = [k for k in corpus if k.stage == 'discourse']

    for key in sorted(keys):
        gra = stacgraph.Graph.from_doc(corpus, key)
        rfc = BasicRfc(gra)
        for subgraph in gra.connected_components():
            sublast = gra.sorted_first_widest(subgraph)[-1]
            for node in rfc.frontier(sublast):
                gra.annotation(node).features['highlight'] = 'green'
        for node, links in rfc.violations().items():
            # gra.annotation(node).features['highlight'] = 'orange'
            for link in links:
                gra.annotation(link).features['highlight'] = 'red'
        dot_gra = stacgraph.DotGraph(gra)
        if dot_gra.get_nodes():
            write_dot_graph(key, output_dir, dot_gra,
                            run_graphviz=args.draw)
        else:
            print("Skipping %s (empty graph)" % key, file=sys.stderr)


def _main_enclosure_graph(args):
    """
    Draw graphs showing which annotations' spans include the others
    """
    corpus = _read_corpus(args)
    output_dir = get_output_dir(args)
    keys = corpus
    if args.tokens:
        postags = educe.stac.postag.read_tags(corpus, args.corpus)
    else:
        postags = None

    for k in sorted(keys):
        if postags:
            gra_ = stacgraph.EnclosureGraph(corpus[k], postags[k])
        else:
            gra_ = stacgraph.EnclosureGraph(corpus[k])
        dot_gra = stacgraph.EnclosureDotGraph(gra_)
        if dot_gra.get_nodes():
            dot_gra.set("ratio", "compress")
            write_dot_graph(k, output_dir, dot_gra,
                            run_graphviz=args.draw)
        else:
            print("Skipping %s (empty graph)" % k, file=sys.stderr)

# ---------------------------------------------------------------------
# args
# ---------------------------------------------------------------------

NAME = 'graph'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    # note: not the usual input args
    parser.add_argument('corpus', metavar='DIR', help='corpus dir')
    parser.add_argument('--output', metavar='DIR', required=True,
                        help='output  dir')
    parser.add_argument('--no-draw', action='store_false',
                        dest='draw',
                        default=True,
                        help='Do not actually draw the graph')
    parser.add_argument('--highlight', nargs='+',
                        metavar='ANNO_ID', type=anno_id,
                        help='Highlight these annotations')
    parser.add_argument('--live', action='store_true',
                        help='Input is a flat collection of aa/ac files)')

    # TODO: would be nice to enforce these groups of args mutually excluding
    # but not sure if the library actually supports it
    psr_rel = parser.add_argument_group("relation graphs")
    psr_rel.add_argument('--split', action='store_true',
                         help='Separate file for each connected component')
    psr_rel.add_argument('--strip-cdus', action='store_true',
                         help='Strip away CDUs (substitute w heads)')

    psr_rfc = parser.add_argument_group("RFC graphs")
    psr_rfc.add_argument('--basic-rfc', action='store_true',
                         help='Highlight RFC frontier and violations')

    psr_enc = parser.add_argument_group("enclosure graphs")
    psr_enc.add_argument('--enclosure', action='store_true',
                         help='Generate enclosure graphs')
    psr_enc.add_argument('--tokens', action='store_true',
                         help='Include pos-tagged tokens')

    educe_group = parser.add_argument_group('corpus filtering arguments')
    add_corpus_filters(educe_group, fields=fields_without(["stage"]))
    parser.set_defaults(func=main)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    if args.enclosure:
        _main_enclosure_graph(args)
    elif args.basic_rfc:
        _main_rfc_graph(args)
    else:
        _main_rel_graph(args)

# vim: syntax=python:
