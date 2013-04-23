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

# ---------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------

class FakeGraph(educe.GraphBase):
    """
    Stand-in for educe.graph.Graph
    """
    def __init__(self):
        educe.GraphBase.__init__(self)

    def _add_fake_node(self, anno_id, type):
        attrs = { 'type'       : type
                }
        self.add_node(anno_id)
        for x in attrs:
            self.add_node_attribute(anno_id, x)

    def _add_fake_edge(self, anno_id, type, members):
        attrs   = { 'type'       : type
                  }
        self.add_edge(anno_id)
        self.add_edge_attributes(anno_id, attrs.items())
        for l in members: self.link(l, anno_id)

    def add_edus(self, *anno_ids):
        for anno_id in anno_ids: self.add_edu(str(anno_id))

    def add_edu(self, anno_id):
        self._add_fake_node(anno_id, 'EDU')

    def add_rel(self, anno_id, node1, node2):
        self._add_fake_edge(anno_id, 'rel', [str(node1), str(node2)])

    def add_cdu(self, anno_id, members):
        self._add_fake_edge(anno_id, 'rel', map(str,members))


gr_simple_cdus = FakeGraph()
gr_simple_cdus.add_edus(1,2,3)
gr_simple_cdus.add_rel('a',1,2)
gr_simple_cdus.add_cdu('X',[1,2])

# TODO: is this test legitimate?
gr_fancy_cdus = FakeGraph()
gr_fancy_cdus.add_edus('1', '1.1', '1.2')
gr_fancy_cdus.add_edus('2', '2.1', '2.1.1')
gr_fancy_cdus.add_edus('3')
gr_fancy_cdus.add_rel('a','1','2')
gr_fancy_cdus.add_rel('b','2','2.1')
gr_fancy_cdus.add_rel('c','2.1','2.1.1')
gr_fancy_cdus.add_rel('from cdu to 1.1','1','1.1')
gr_fancy_cdus.add_rel('distractor','3','1.1')
gr_fancy_cdus.add_cdu('X',['1','2'])

def test_cdu_members_trivial():
    "trivial CDU membership"
    members  = gr_simple_cdus.cdu_members('X')
    expected = frozenset(['1','2'])
    assert members == expected

gr_cdu_neighbors = FakeGraph()
gr_cdu_neighbors.add_edus('a1','a2','b')
gr_cdu_neighbors.add_cdu('A',['a1','a2'])

def test_cdu_neighbors():
    "does belong in the same CDU make you a neighbour?"

    # this is probably not a desirable property, but is a consequence
    # of CDUs being represented as hyperedges
    #
    # the API may have to expose a notion of being a neighbor only
    # via relations
    ns1       = frozenset(gr_cdu_neighbors.neighbors('a1'))
    expected1 = frozenset(['a2'])
    assert ns1 == expected1

    ns2       = frozenset(gr_cdu_neighbors.neighbors('a2'))
    expected2 = frozenset(['a1'])
    assert ns2 == expected2

    ns3       = frozenset(gr_cdu_neighbors.neighbors('b'))
    expected3 = frozenset([])
    assert ns3 == expected3


#def test_cdu_members():
#    "CDU membership with a bit of depth"
#    members  = gr_fancy_cdus.cdu_members('X')
#    expected = frozenset(['1','2','2.1','2.1.1'])
#    print >> sys.stderr, members
#    assert members == expected
