# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
Tests for educe
"""

import copy
import pydot
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

    def add_edu(self, anno_id):
        self._add_fake_node(anno_id, 'EDU')

    def add_rel(self, anno_id, node1, node2):
        self._add_fake_edge(anno_id, 'rel', [node1, node2])

    def add_cdu(self, anno_id, members):
        self._add_fake_edge(anno_id, 'rel', members)


gr_simple_cdus = FakeGraph()
gr_simple_cdus.add_edu('1')
gr_simple_cdus.add_edu('2')
gr_simple_cdus.add_edu('3')
gr_simple_cdus.add_rel('a','1','2')
gr_simple_cdus.add_cdu('X',['1','2'])

# TODO: is this test legitimate?
gr_fancy_cdus = FakeGraph()
gr_fancy_cdus.add_edu('1')
gr_fancy_cdus.add_edu('1.1')
gr_fancy_cdus.add_edu('2')
gr_fancy_cdus.add_edu('2.1')
gr_fancy_cdus.add_edu('2.1.1')
gr_fancy_cdus.add_edu('3')
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

#def test_cdu_members():
#    "CDU membership with a bit of depth"
#    members  = gr_fancy_cdus.cdu_members('X')
#    expected = frozenset(['1','2','2.1','2.1.1'])
#    print >> sys.stderr, members
#    assert members == expected
