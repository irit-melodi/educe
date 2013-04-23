# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
Tests for educe
"""

import copy
import pygraph.classes.hypergraph as gr
import educe.graph as educe
import sys
from pygraph.algorithms import accessibility, traversal, searching

# ---------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------

class FakeGraph(educe.HypergraphBase):
    """
    Stand-in for educe.graph.Graph
    """
    def __init__(self):
        educe.HypergraphBase.__init__(self)

    def _add_fake_node(self, anno_id, type):
        attrs = { 'type'       : type
                }
        self.add_node(anno_id)
        for x in attrs.items():
            self.add_node_attribute(anno_id, x)

    def _add_fake_edge(self, anno_id, type, members):
        attrs   = { 'type'       : type
                  }
        self.add_node(anno_id)
        self.add_edge(anno_id)
        for x in attrs.items():
            self.add_edge_attribute(anno_id, x)
            self.add_node_attribute(anno_id, x)

        for l in members: self.link(l, anno_id)

    def add_edus(self, *anno_ids):
        for anno_id in anno_ids: self.add_edu(str(anno_id))

    def add_edu(self, anno_id):
        self._add_fake_node(anno_id, 'EDU')

    def add_rel(self, anno_id, node1, node2):
        self._add_fake_edge(anno_id, 'rel', [str(node1), str(node2)])

    def add_cdu(self, anno_id, members):
        self._add_fake_edge(anno_id, 'CDU', map(str,members))

def test_cdu_members_trivial():
    "trivial CDU membership"
    gr = FakeGraph()
    gr.add_edus(1,2,3)
    gr.add_rel('a',1,2)
    gr.add_cdu('X',[1,2])

    members  = gr.cdu_members('X')
    expected = frozenset(['1','2'])
    assert members == expected

# this is probably not a desirable property, but is a consequence
# of CDUs being represented as hyperedges
#
# the API may have to expose a notion of being a neighbor only
# via relations
def test_cdu_neighbors():
    "does belong in the same CDU make you a neighbour?"

    gr = FakeGraph()
    gr.add_edus('a1','a2','b')
    gr.add_cdu('A',['a1','a2'])

    ns1       = frozenset(gr.neighbors('a1'))
    expected1 = frozenset(['a2'])
    assert ns1 == expected1

    ns2       = frozenset(gr.neighbors('a2'))
    expected2 = frozenset(['a1'])
    assert ns2 == expected2

    ns3       = frozenset(gr.neighbors('b'))
    expected3 = frozenset([])
    assert ns3 == expected3

#def test_indirect_links():
#    gr = FakeGraph()
#    gr.add_edus(1,2,3,4,5)
#    gr.add_rel('1.2',1,2)
#    gr.add_rel('2.3',2,3)
#    gr.add_rel('2.4',3,4)
#
#    print >> sys.stderr, accessibility.accessibility(gr)
#
#    for n in traversal.traversal(gr, '1', 'pre'):
#        print >> sys.stderr, "traverse: " + n
#    pass

def test_flattening():
    gr = FakeGraph()
    gr.add_edus(*range(1,6))
    gr.add_rel('1.2', 1, 2)
    gr.add_rel('1.3', 1, 3)
    gr.add_rel('3.5', 3, 5)
    gr.add_rel('r', '1.2', '5')
    gr.add_cdu('X', [2,3,4])
    gr.add_cdu('Y', [5,'X'])

    dgr = educe.FlatGraph(gr)
    expected_nodes = frozenset(map(str,range(1,6)) + ['X','Y','1.2'])
    actual_nodes   = frozenset(dgr.nodes())
    assert actual_nodes == expected_nodes
     # relates a relation but isn't itself pointed to
    assert 'r' not in actual_nodes
    assert ('3','5') in dgr.edges()
    assert ('X','3') in dgr.edges()
    assert ('Y','X') in dgr.edges()

    assert dgr.is_cdu('Y')
    assert not dgr.is_edu('Y')
    assert dgr.is_relation('1.2')
    assert dgr.is_edu('5')


# TODO: is this test legitimate?
#gr_fancy_cdus = FakeGraph()
#gr_fancy_cdus.add_edus('1', '1.1', '1.2')
#gr_fancy_cdus.add_edus('2', '2.1', '2.1.1')
#gr_fancy_cdus.add_edus('3')
#gr_fancy_cdus.add_rel('a','1','2')
#gr_fancy_cdus.add_rel('b','2','2.1')
#gr_fancy_cdus.add_rel('c','2.1','2.1.1')
#gr_fancy_cdus.add_rel('from cdu to 1.1','1','1.1')
#gr_fancy_cdus.add_rel('distractor','3','1.1')
#gr_fancy_cdus.add_cdu('X',['1','2'])
#
#def test_cdu_members():
#    "CDU membership with a bit of depth"
#    members  = gr_fancy_cdus.cdu_members('X')
#    expected = frozenset(['1','2','2.1','2.1.1'])
#    print >> sys.stderr, members
#    assert members == expected
