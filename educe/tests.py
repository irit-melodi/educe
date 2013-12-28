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
from educe.annotation import *
import unittest

# ---------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------

def test_span():
    assert not Span(5,10).overlaps(Span(11,12))
    assert not Span(11,12).overlaps(Span(5,10))
    def matches((x1,y1),(x2,y2),(rx,ry)):
        o = Span(x1,y1).overlaps(Span(x2,y2))
        assert o
        assert o == Span(rx,ry)
    matches((5,10),(6,9),(6,9))
    matches((6,9),(5,10),(6,9))
    matches((5,10),(7,12),(7,10))
    matches((7,12),(5,10),(7,10))

# ---------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------

class TestUnit(Unit):
    def __init__(self, id, start, end):
        Unit.__init__(self, id, Span(start, end), '', {})

class TestRelation(Relation):
    def __init__(self, id, start, end):
        Relation.__init__(self, id, RelSpan(start, end), '', {})

class TestSchema(Schema):
    def __init__(self, id, units, relations, schemas):
        Schema.__init__(self, id, frozenset(units), frozenset(relations), frozenset(schemas), '', {})

class TestDocument(Document):
    def __init__(self, units, rels, schemas, txt):
        Document.__init__(self, units, rels, schemas, txt)

def test_members():
    u1  = TestUnit('u1', 2, 4)
    u2  = TestUnit('u2', 3, 9)
    u3  = TestUnit('distractor', 1,10)
    u4  = TestUnit('u4', 12,13)
    u5  = TestUnit('u5', 4,12)
    u6  = TestUnit('u6', 7,14)
    s1  = TestSchema('s1', ['u4','u5','u6'], [], [])
    r1  = TestRelation('r1', 's1','u2')

    doc = TestDocument([u1,u2,u3,u4,u5,u6],[r1],[s1], "why hello there!")
    assert u1._members() is None
    assert sorted(s1._members()) == sorted([u4,u5,u6])
    assert sorted(r1._members()) == sorted([u2,s1])

    assert u1._terminals() == [u1]
    assert sorted(s1._terminals()) == sorted([u4,u5,u6])
    assert sorted(r1._terminals()) == sorted([u2,u4,u5,u6])

    doc_sp = doc.text_span()
    for x in doc.annotations():
        sp = x.text_span()
        assert sp.char_start <= sp.char_end
        assert sp.char_start >= doc_sp.char_start
        assert sp.char_end   <= doc_sp.char_end

# ---------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------

class FakeGraph(educe.Graph):
    """
    Stand-in for educe.graph.Graph
    """
    def __init__(self):
        educe.Graph.__init__(self)
        self.corpus  = {}
        self.doc_key = None
        self.doc     = None

    def _add_fake_node(self, anno_id, type):
        attrs = { 'type'       : type
                }
        self.add_node(anno_id)
        for x in attrs.items():
            self.add_node_attribute(anno_id, x)

    def _add_fake_edge(self, anno_id, type, members):
        attrs   = { 'type'       : type
                  , 'mirror'     : anno_id
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

class BasicGraphTest(unittest.TestCase):
    def test_cdu_members_trivial(self):
        "trivial CDU membership"
        gr = FakeGraph()
        gr.add_edus(1,2,3)
        gr.add_rel('a',1,2)
        gr.add_cdu('X',[1,2])

        members  = gr.cdu_members('X')
        expected = frozenset(['1','2'])
        self.assertEqual(members, expected)

    # this is probably not a desirable property, but is a consequence
    # of CDUs being represented as hyperedges
    #
    # the API may have to expose a notion of being a neighbor only
    # via relations
    def test_cdu_neighbors(self):
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

    def test_copy(self):
        """
        graph in essentially two components but some links
        what happens if we copy just one or the other of the
        components wrt links etc?
        """

        gr = FakeGraph()
        gr.add_edus(*range(1,4))
        gr.add_edus(*range(10,14))
        gr.add_rel('1.2', 1, 2)
        gr.add_rel('1.3', 1, 3)
        gr.add_rel('2.11', 2, 11) # bridge!
        gr.add_rel('11.12', 11, 12)
        gr.add_rel('12.13', 11, 12)
        gr.add_cdu('X1', [2,3])
        gr.add_cdu('X2', ['X1',1]) # should be copied
        gr.add_cdu('Y1', [12,13])
        gr.add_cdu('XY', [1,13])   # should not be copied

        xset2 = set(map(str,[1,2,3]))
        gr2   = gr.copy(nodeset=xset2)
        assert gr2.edus()      == xset2
        assert gr2.relations() == set(['1.2','1.3'])
        assert gr2.cdus()      == set(['X1', 'X2'])
        assert gr2.links('X2') == gr.links('X2')

        # some nonsense copies
        xset3 = xset2 | set(['X1'])
        gr3   = gr.copy(nodeset=xset3)
        assert gr3.edus() == xset2 # not xset3

        # including CDU should also result in members being included
        xset4 = set(['X2'])
        gr4   = gr.copy(nodeset=xset4)
        assert gr4.edus() == xset2
        assert gr4.cdus() == set(['X1', 'X2'])
