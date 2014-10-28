# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3
# pylint: disable=too-many-public-methods, invalid-name

"""
Tests for educe
"""

import unittest

from educe.annotation import\
    Span, RelSpan,\
    Annotation,\
    Unit, Relation, Schema, Document
import educe.graph as educe
from   educe.graph import EnclosureGraph

# ---------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------


class SpanTest(unittest.TestCase):
    "tests for educe.annotation.Span"

    def __init__(self, *args, **kwargs):
        super(SpanTest, self).__init__(*args, **kwargs)
        self.addTypeEqualityFunc(Span, self.assertEqualStrFail)

    def assertEqualStrFail(self, a, b, msg):
        """
        just like assertEqual but display both sides with str on failure
        """
        if a != b:
            msg = msg or "{0} != {1}".format(a, b)
            raise self.failureException(msg)

    def assertOverlap(self, expected, pair1, pair2, **kwargs):
        "true if `pair1.overlaps(pair2) == expected` (modulo boxing)"
        (x1, y1) = pair1
        (x2, y2) = pair2
        (rx, ry) = expected
        o = Span(x1, y1).overlaps(Span(x2, y2), **kwargs)
        self.assertTrue(o)
        self.assertEqual(Span(rx, ry), o)

    def assertNotOverlap(self, pair1, pair2, **kwargs):
        "true if `pair1.overlaps(pair2) == expected` (modulo boxing)"
        (x1, y1) = pair1
        (x2, y2) = pair2
        self.assertFalse(Span(x1, y1).overlaps(Span(x2, y2), **kwargs))

    def test_overlap(self):
        "Span.overlaps() function"

        self.assertNotOverlap((5, 10), (11, 12))
        self.assertNotOverlap((11, 12), (5, 10))

        # should not overlap at edges
        self.assertNotOverlap((5, 10), (10, 15))
        self.assertOverlap((10, 10), (5, 10), (10, 15), inclusive=True)

        self.assertOverlap((6, 9), (5, 10), (6, 9))
        self.assertOverlap((6, 9), (6, 9), (5, 10))
        self.assertOverlap((7, 10), (5, 10), (7, 12))
        self.assertOverlap((7, 10), (7, 12), (5, 10))

    def test_overlap_empty(self):
        "Span.overlaps() on empty spans"

        self.assertOverlap((5, 5), (5, 5), (4, 6))
        self.assertOverlap((5, 5), (5, 5), (4, 5))
        self.assertOverlap((5, 5), (5, 5), (5, 6))

        self.assertOverlap((5, 5), (5, 5), (4, 6), inclusive=True)
        self.assertOverlap((5, 5), (5, 5), (4, 5), inclusive=True)
        self.assertOverlap((5, 5), (5, 5), (5, 6), inclusive=True)



class NullAnno(Span, Annotation):
    def __init__(self, start, end, type="null"):
        super(NullAnno, self).__init__(start, end)
        self.span = self
        self.type = type

    def local_id(self):
        return str(self)

    def __eq__(self, other):
        return self.char_start == other.char_start\
            and self.char_end == other.char_end\
            and self.type == other.type

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.char_start, self.char_end, self.type))

    def __repr__(self):
        return "%s [%s]" % (super(NullAnno,self).__str__(),self.type)

    def __str__(self):
        return repr(self)

class EnclosureTest(unittest.TestCase):
    def test_trivial(self):
        g = EnclosureGraph([])
        self.assertEqual(0, len(g.nodes()))

    def test_singleton(self):
        s0 = NullAnno(1,5)
        g = EnclosureGraph([s0])
        self.assertEqual([s0.local_id()], g.nodes())
        self.assertEqual([], g.inside(s0))
        self.assertEqual([], g.outside(s0))

    def test_simple_enclosure(self):
        s1_5 = NullAnno(1,5)
        s2_3 = NullAnno(2,3)
        g = EnclosureGraph([s1_5, s2_3])
        self.assertEqual([s2_3], g.inside(s1_5))
        self.assertEqual([s1_5], g.outside(s2_3))

    def test_indirect_enclosure(self):
        s1_5 = NullAnno(1,5,'a')
        s2_4 = NullAnno(2,4,'b')
        s3_4 = NullAnno(3,4,'c')
        g = EnclosureGraph([s1_5, s2_4, s3_4])
        self.assertEqual([s2_4], g.inside(s1_5))
        self.assertEqual([s1_5], g.outside(s2_4))
        self.assertEqual([s2_4], g.outside(s3_4))

    def test_same_span(self):
        s1_5  = NullAnno(1,5,'out')
        s2_4a = NullAnno(2,4,'a')
        s2_4b = NullAnno(2,4,'b')
        s3_4  = NullAnno(3,4,'in')
        g = EnclosureGraph([s1_5, s2_4a, s2_4b, s3_4])
        g.reduce()
        self.assertEqual([s2_4a, s2_4b], g.inside(s1_5))
        self.assertEqual([s3_4], g.inside(s2_4a))
        self.assertEqual([s3_4], g.inside(s2_4b))
        self.assertEqual([s2_4a, s2_4b], g.outside(s3_4))

    def test_layers(self):
        s3_4 = NullAnno(3, 4)
        s2_4 = NullAnno(2, 4)
        s3_5 = NullAnno(3, 5)
        g = EnclosureGraph([s3_4, s2_4, s3_5])
        g.reduce()
        self.assertEqual([s3_4], g.inside(s3_5))
        self.assertEqual([s3_4], g.inside(s2_4))
        self.assertEqual([s2_4, s3_5], g.outside(s3_4))

    def test_indirect_enclosure_untyped(self):
        """
        reduce only pays attention to nodes of different type
        """
        s_1_5 = NullAnno(1,5)
        s_2_4 = NullAnno(2,4)
        s_3_4 = NullAnno(3,4)
        g = EnclosureGraph([s_1_5, s_2_4, s_3_4])
        g.reduce()
        self.assertEqual([s_2_4, s_3_4], g.inside(s_1_5))
        self.assertEqual([s_1_5], g.outside(s_2_4))
        self.assertEqual([s_1_5, s_2_4], g.outside(s_3_4))


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

    def add_non_edu(self, anno_id):
        self._add_fake_node(anno_id, 'not-an-EDU')

    def add_rel(self, anno_id, node1, node2):
        self._add_fake_edge(anno_id, 'rel', [str(node1), str(node2)])

    def add_cdu(self, anno_id, members):
        self._add_fake_edge(anno_id, 'CDU', list(map(str,members)))

class BasicGraphTest(unittest.TestCase):
    def test_cdu_members_trivial(self):
        "trivial CDU membership"
        gr = FakeGraph()
        gr.add_edus(1,2,3)
        gr.add_rel('a',1,2)
        gr.add_cdu('X',[1,2])

        members  = gr.cdu_members('X')
        expected = frozenset(['1','2'])
        self.assertEqual(expected, members)

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
        self.assertEqual(expected1, ns1)

        ns2       = frozenset(gr.neighbors('a2'))
        expected2 = frozenset(['a1'])
        self.assertEqual(expected2,ns2)

        ns3       = frozenset(gr.neighbors('b'))
        expected3 = frozenset([])
        self.assertEqual(expected3,ns3)

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
        self.assertEqual(xset2,              gr2.edus())
        self.assertEqual(set(['1.2','1.3']), gr2.relations())
        self.assertEqual(set(['X1', 'X2']),  gr2.cdus())
        self.assertEqual(gr.links('X2'),     gr2.links('X2'))

        # some nonsense copies
        xset3 = xset2 | set(['X1'])
        gr3   = gr.copy(nodeset=xset3)
        self.assertEqual(xset2, gr3.edus()) #not xset3

        # including CDU should also result in members being included
        xset4 = set(['X2'])
        gr4   = gr.copy(nodeset=xset4)
        self.assertEqual(xset2,             gr4.edus())
        self.assertEqual(set(['X1', 'X2']), gr4.cdus())
