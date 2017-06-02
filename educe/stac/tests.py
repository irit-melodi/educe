# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3
"""
Tests for educe.stac
"""

from __future__ import print_function

import codecs
import copy
import subprocess
import sys
import unittest

import educe.stac.graph as stac_gr
from educe import annotation, corpus, stac
from educe.corpus import FileId
from educe.stac import fake_graph
from educe.stac.rfc import BasicRfc, ThreadedRfc
from educe.stac.util.output import mk_parent_dirs


class FakeEDU(annotation.Unit):
    def __init__(self, unit_id, span=(0, 0), type='Segment'):
        sp = annotation.Span(*span)
        annotation.Unit.__init__(self, unit_id, sp, type, {}, {}, None)


class FakeRelInst(annotation.Relation):
    def __init__(self, rel_id, source, target, type='Comment'):
        rel_sp = annotation.RelSpan(source.local_id(), target.local_id())
        annotation.Relation.__init__(self, rel_id, rel_sp, type, {}, {})


class FakeCDU(annotation.Schema):
    def __init__(self, schema_id, members):
        edus = set(x.local_id() for x in members if isinstance(x, FakeEDU))
        cdus = set(x.local_id() for x in members if isinstance(x, FakeCDU))
        rels = set()
        annotation.Schema.__init__(self, schema_id, edus, rels, cdus,
                                   'Complex_discourse_unit', {}, {})


class FakeDocument(annotation.Document):
    def __init__(self, edus, rels, cdus, text=""):
        self.copies = {}
        edus2, rels2, cdus2 = copy.deepcopy((edus, rels, cdus))
        for k, v in zip(edus, edus2):
            self.copies[k] = v
        for k, v in zip(rels, rels2):
            self.copies[k] = v
        for k, v in zip(cdus, cdus2):
            self.copies[k] = v
        annotation.Document.__init__(self, edus2, rels2, cdus2, text)


class FakeKey(corpus.FileId):
    def __init__(self, k):
        corpus.FileId.__init__(self, k, None, None, None)


def graph_ids(gr):
    """
    local ids to graph ids (eg. c1 => e_foo_c1)

    This is motivated by two things. First, graphs have an
    internal identifier mechanism they use to identify nodes and
    that the API uses for operations on these nodes.

    Second, there are various pointer-updating shenanigans in
    the rest of the educe library.  If we associate some annotations
    with a document and flesh it out, the annotation is updated with
    an "origin".  This means you can't reuse the same annotation in
    a different document (because fleshing that out would clobber the
    origin of the original).  This is not a problem for slurped
    corpora because each document is associated with different
    annotations, but in the test suite it's convenient to re-use
    annotations, so internally we make deep copies.

    Upshot is that the simplest way to refer to an object is through
    the short it. Mutability is evil.
    """
    ids = {}
    for x in gr.nodes():
        local_id = gr.annotation(x).local_id()
        ids[local_id] = x
    for x in gr.hyperedges():
        local_id = gr.annotation(x).local_id()
        ids[local_id] = x
    return ids


def nodeform_graph_ids(gr):
    ids = {}
    for x in gr.nodes():
        local_id = gr.annotation(x).local_id()
        ids[local_id] = x
    for x in gr.hyperedges():
        local_id = gr.annotation(x).local_id()
        ids[local_id] = gr.mirror(x)
    return ids


class GraphTest(unittest.TestCase):
    def mk_graph(self, edus, rels, cdus):
        doc = FakeDocument(edus, rels, cdus)
        k = FakeKey('k')
        doc.fleshout(k)
        gr = stac_gr.Graph.from_doc({k: doc}, k)
        return gr, graph_ids(gr)

    def setUp(self):
        self.edu1_1 = FakeEDU('e1.1', span=(1, 2))
        self.edu1_2 = FakeEDU('e1.2', span=(2, 5))
        self.edu1_3 = FakeEDU('e1.3', span=(7, 8))
        self.edu2_1 = FakeEDU('e3', span=(16, 18))

        self.edus1 = [self.edu1_1, self.edu1_2, self.edu1_3,
                      self.edu2_1]

    def test_containing_cdu_trivial(self):
        c = FakeCDU('c', [self.edu1_1])
        gr, ids = self.mk_graph(self.edus1, [], [c])
        mark = ids[self.edu1_1.local_id()]
        expected = ids[c.local_id()]
        self.assertEqual(expected, gr.containing_cdu(mark))

    def test_containing_cdu_nested(self):
        cx = FakeCDU('cx', [self.edu1_1])
        cy = FakeCDU('cy', [cx, self.edu1_2])
        cz = FakeCDU('cz', [cy, self.edu1_3])
        gr, ids = self.mk_graph(self.edus1, [], [cx, cy, cz])

        mark = ids[cx.local_id()]
        expected = ids[cy.local_id()]
        self.assertEqual(expected, gr.containing_cdu(mark))

        mark = ids[cy.local_id()]
        expected = ids[cz.local_id()]
        self.assertEqual(expected, gr.containing_cdu(mark))

        mark = ids[cz.local_id()]
        expected = None
        self.assertEqual(expected, gr.containing_cdu(mark))

    def test_linked_non_edus_included(self):
        "non EDU connection"
        edu1 = self.edu1_1
        non_edu1 = FakeEDU('ne1', span=(3, 3), type='Preference')
        non_edu2 = FakeEDU('ne2', span=(1, 3), type='Preference')
        rel1 = FakeRelInst('r-ne1-e1', non_edu1, edu1)
        edus = [non_edu1, non_edu2, edu1]
        rels = [rel1]
        gr, ids = self.mk_graph(edus, rels, [])

        # this tests both that non_edu1 is included (by virtue of its
        # link to edu1; and also that non_edu2 is excluded)
        expected = frozenset(e.local_id() for e in rels + [non_edu1, edu1])
        self.assertEqual(expected, frozenset(ids.keys()))

    def test_unlinked_edus_included(self):
        "non EDU connection"
        edu1 = self.edu1_1
        edu2 = self.edu1_2
        edu3 = self.edu1_3
        rel1 = FakeRelInst('r-e1-e2', edu1, edu2)
        edus = [edu1, edu2, edu3]
        rels = [rel1]
        gr, ids = self.mk_graph(edus, rels, [])

        # this tests both that non_edu1 is included (by virtue of its
        # link to edu1; and also that non_edu2 is excluded)
        expected = frozenset(e.local_id() for e in edus + rels)
        self.assertEqual(expected, frozenset(ids.keys()))

# FIXME: the tests below should be shuffled into the fixture above

edu1 = FakeEDU('e1')
edu2 = FakeEDU('e2')
edu3 = FakeEDU('e3', span=(1, 1))
edu4 = FakeEDU('e4')

cdu1 = FakeCDU('c1', [edu1, edu2, edu3])
cdu2 = FakeCDU('c2', [cdu1, edu4])

rel1 = FakeRelInst('r-e1-e2', edu1, edu2)
rel2 = FakeRelInst('r-e2-e3', edu2, edu3)
rel3 = FakeRelInst('r-c1-e4', cdu1, edu4)


def test_fake_objs():
    assert stac.is_edu(edu1)
    assert stac.is_relation_instance(rel1)
    assert stac.is_cdu(cdu1)


def test_cdu_head_multiheaded():
    "trivial CDU membership"
    doc = FakeDocument([edu1, edu2, edu3],
                       [rel1],
                       [cdu1])
    k = FakeKey('cdu_head_test')
    doc.fleshout(k)
    gr = stac_gr.Graph.from_doc({k: doc}, k)
    ids = graph_ids(gr)

    # gr1 has a multi-headed cdu, should fail
    try:
        gr.cdu_head(ids['c1'])
        assert False  # should not get here
    except stac_gr.MultiheadedCduException as e:
        pass

    # but sloppy is ok
    assert gr.cdu_head(ids['c1'], sloppy=True) == ids['e1']


class CduHeadTest(unittest.TestCase):

    def test_cdu_head(self):
        "cdu[e1 -> e2 -> e2]"
        doc = FakeDocument([edu1, edu2, edu3],
                           [rel1, rel2],
                           [cdu1])
        k = FakeKey('cdu_head_test')
        doc.fleshout(k)
        gra = stac_gr.Graph.from_doc({k: doc}, k)
        ids = graph_ids(gra)
        self.assertEqual(gra.cdu_head(ids['c1']),
                         ids['e1'])

    def test_embedded_cdu_head(self):
        "cdu[cdu[e1 -> e2 -> e3] -> e4]"
        doc = FakeDocument([edu1, edu2, edu3, edu4],
                           [rel1, rel2, rel3],
                           [cdu1, cdu2])
        k = FakeKey('cdu_head_test')
        doc.fleshout(k)
        gra = stac_gr.Graph.from_doc({k: doc}, k)
        ids = graph_ids(gra)
        self.assertEqual(gra.cdu_head(ids['c2']),
                         ids['c1'])

        deep_heads = gra.recursive_cdu_heads()
        self.assertEqual(deep_heads[ids['c1']],
                         gra.cdu_head(ids['c1']))
        self.assertEqual(deep_heads[ids['c1']],
                         deep_heads[ids['c2']])


def test_first_outermost_dus_simple():
    edu1 = FakeEDU('e1', span=(1, 2))
    edu2 = FakeEDU('e2', span=(1, 3))
    edu3 = FakeEDU('e3', span=(2, 3))
    rel1 = FakeRelInst('r-e1-e2', edu1, edu2)
    rel2 = FakeRelInst('r-e2-e3', edu2, edu3)
    doc = FakeDocument([edu1, edu2, edu3],
                       [rel1, rel2],
                       [])
    k = corpus.FileId('moo', None, None, None)
    gr = stac_gr.Graph.from_doc({k: doc}, k)
    ids = nodeform_graph_ids(gr)
    assert gr.first_outermost_dus() == [ids[x] for x in ['e2', 'e1', 'e3']]


def test_first_outermost_dus():
    edu1 = FakeEDU('e1', span=(1, 2))
    edu2 = FakeEDU('e2', span=(1, 3))
    edu3 = FakeEDU('e3', span=(2, 3))

    edu4 = FakeEDU('e4', span=(4, 5))
    edu5 = FakeEDU('e5', span=(6, 8))

    rel1 = FakeRelInst('r-e1-e2', edu1, edu2)
    rel2 = FakeRelInst('r-e2-e3', edu2, edu3)
    rel3 = FakeRelInst('r-e4-e5', edu4, edu5)

    cdu1 = FakeCDU('c1', [edu1, edu2, edu3])
    cdu2 = FakeCDU('c2', [edu4, edu5])
    cdu3 = FakeCDU('c3', [cdu1, cdu2])

    doc = FakeDocument([edu1, edu2, edu3, edu4, edu5],
                       [rel1, rel2, rel3],
                       [cdu1, cdu2, cdu3])
    k = corpus.FileId('moo', None, None, None)
    gr = stac_gr.Graph.from_doc({k: doc}, k)
    ids = nodeform_graph_ids(gr)
    got = gr.first_outermost_dus()
    expected = ['c3', 'c1', 'e2', 'e1', 'e3', 'c2', 'e4', 'e5']
    assert got == [ids[x] for x in expected]


def mk_graphs(src, dump=None):
    """ Returns educe.fake_graph.LightGraph and educe.stac.Graph
        for given LightGraph source string
        (see LightGraph doc for specification)
    """
    # TODO : dump
    lg = fake_graph.LightGraph(src)
    doc = lg.get_doc()
    doc_id = FileId('test', '01', 'discourse', 'GOLD')
    doc.set_origin(doc_id)
    graph = stac_gr.Graph.from_doc({doc_id: doc}, doc_id)

    if dump is not None:
        dump_graph(dump, graph)

    return lg, graph


def dump_graph(dump_filename, graph):
    """
    Write a dot graph and possibly run graphviz on it
    """
    dot_graph = stac_gr.DotGraph(graph)
    dot_file = dump_filename + '.dot'
    svg_file = dump_filename + '.svg'
    mk_parent_dirs(dot_file)
    with codecs.open(dot_file, 'w', encoding='utf-8') as dotf:
        print(dot_graph.to_string(), file=dotf)
    print("Creating %s" % svg_file, file=sys.stderr)
    subprocess.call('dot -T svg -o %s %s' % (svg_file, dot_file), shell=True)


class BasicRfcTest(unittest.TestCase):

    def violations(self, graph):
        rfc = BasicRfc(graph)
        return list(graph.annotation(x) for x in rfc.violations())

    def assertNoViolations(self, graph):
        violations = self.violations(graph)
        self.assertEqual(violations, [])

    def test_trivial(self):
        """ a -> b (no violations) """
        lg, graph = mk_graphs('#Aab / Sab')
        # lg, graph = mk_graphs('#Aab / Sab',
        # dump = '/tmp/graph/trivial')
        self.assertNoViolations(graph)

    def test_trivial_violation(self):
        """ a -C> b -S> c (a-c is a violation)"""
        lg, graph = mk_graphs('#Aabc / CabSc Sac')
        violations = self.violations(graph)
        self.assertNotIn(lg.get_edge('a', 'b'), violations)
        self.assertNotIn(lg.get_edge('b', 'c'), violations)
        self.assertIn(lg.get_edge('a', 'c'), violations)

    def test_cdu_bridge(self):
        """ a -> [b-c] -> d (a-d is a violation unless b-c is in a CDU) """
        lg1, g1 = mk_graphs('#Aabcd / SabCc Sad')
        lg2, g2 = mk_graphs('#Aabcd / x(bc) / Sax Cbc Sad')
        violations1 = self.violations(g1)
        violations2 = self.violations(g2)
        self.assertIn(lg1.get_edge('a', 'd'), violations1)
        self.assertNotIn(lg2.get_edge('a', 'd'), violations2)

    def test_cdu_basic(self):
        """ a -> [b-c] -> d (no violation)"""
        lg, graph = mk_graphs('#Aabcd / x(bc) / Saxd Cbc')
        self.assertNoViolations(graph)

    def test_cdu_dangler(self):
        """ [c] / c -> d (no violation)"""
        lg, graph = mk_graphs('#Acd / x(c) / Scd')
        self.assertNoViolations(graph)

    def test_ambiguity_multiparent(self):
        """ a -> c / b -> c / a -> d / b -> d (a and b not linked)
        Both parents of c (a and b) must be accessible by d
        """
        lg, graph = mk_graphs('#Aabcd / Sac bc ad bd')
        violations = self.violations(graph)
        self.assertNotIn(lg.get_edge('a', 'd'), violations)
        self.assertNotIn(lg.get_edge('b', 'd'), violations)

    def test_ambiguity_cdu_and_parent(self):
        """ a -> b -> d / [b] -> d / a -> d (a and [b] not linked)
        Both x (enclosing CDU) and a (parent) must be accessible by d
        """
        lg, graph = mk_graphs('#Aabd / x(b) / Sabd xd ad')
        violations = self.violations(graph)
        self.assertNotIn(lg.get_edge('x', 'd'), violations)
        self.assertNotIn(lg.get_edge('a', 'd'), violations)
        self.assertNotIn(lg.get_edge('b', 'd'), violations)

    def test_backwards(self):
        """ a -> b -> c -> b and b -> c -> b
        Both are backwards link violations
        """
        l1, g1 = mk_graphs('#Aabc / Sabc Scb')
        l2, g2 = mk_graphs('#Abc / Sbcb')
        violations1 = self.violations(g1)
        violations2 = self.violations(g2)
        self.assertIn(l1.get_edge('c', 'b'), violations1)
        self.assertIn(l2.get_edge('c', 'b'), violations2)


class ThreadedRfcTest(BasicRfcTest):
    def violations(self, graph):
        rfc = ThreadedRfc(graph)
        return list(graph.annotation(x) for x in rfc.violations())

    def test_multi_basic(self):
        """ Aa -> Cc, Bb -> Cc (a-c only violation if basic RFC) """
        lg, graph = mk_graphs('#Aa Bb Cc / Sac bc')
        basic_violations = super(ThreadedRfcTest, self).violations(graph)
        self.assertNotIn(lg.get_edge('b', 'c'), basic_violations)
        self.assertIn(lg.get_edge('a', 'c'), basic_violations)

        multi_violations = self.violations(graph)
        self.assertNotIn(lg.get_edge('b', 'c'), multi_violations)
        self.assertNotIn(lg.get_edge('a', 'c'), multi_violations)
