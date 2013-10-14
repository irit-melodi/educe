# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
Tests for educe.stac
"""

import copy

import educe.tests
import educe.stac.graph as stac_gr
from educe import annotation, corpus, stac

import sys


class FakeEDU(annotation.Unit):
    def __init__(self, unit_id, span=(0,0), type='Segment'):
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
        annotation.Schema.__init__(self, schema_id,\
                                   edus, rels, cdus,\
                                   'Complex_discourse_unit',
                                   {}, {})

class FakeDocument(annotation.Document):
    def __init__(self, edus, rels, cdus, text=""):
        edus2, rels2, cdus2 = copy.deepcopy((edus,rels,cdus))
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

edu1 = FakeEDU('e1')
edu2 = FakeEDU('e2')
edu3 = FakeEDU('e3',span=(1,1))
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
    doc  = FakeDocument([edu1, edu2, edu3],
                        [rel1],
                        [cdu1])
    k   = FakeKey('cdu_head_test')
    doc.fleshout(k)
    gr  = stac_gr.Graph.from_doc({k:doc}, k)
    ids = graph_ids(gr)

    # gr1 has a multi-headed cdu, should fail
    try:
        gr.cdu_head(ids['c1'])
        assert False # should not get here
    except stac_gr.MultiheadedCduException as e:
        pass

    # but sloppy is ok
    assert gr.cdu_head(ids['c1'], sloppy=True) == ids['e1']

def test_cdu_head():
    doc  = FakeDocument([edu1, edu2, edu3],
                        [rel1, rel2],
                        [cdu1])
    k    = FakeKey('cdu_head_test')
    doc.fleshout(k)
    gr   = stac_gr.Graph.from_doc({k:doc}, k)
    ids  = graph_ids(gr)
    assert gr.cdu_head(ids['c1']) == ids['e1']

def test_embedded_cdu_head():
    doc  = FakeDocument([edu1, edu2, edu3, edu4],
                        [rel1, rel2, rel3],
                        [cdu1, cdu2])
    k    = FakeKey('cdu_head_test')
    doc.fleshout(k)
    gr   = stac_gr.Graph.from_doc({k:doc}, k)
    ids  = graph_ids(gr)
    assert gr.cdu_head(ids['c2']) == ids['c1']

    deep_heads = gr.recursive_cdu_heads()
    assert deep_heads[ids['c1']] == gr.cdu_head(ids['c1'])
    assert deep_heads[ids['c1']] == deep_heads[ids['c2']]
