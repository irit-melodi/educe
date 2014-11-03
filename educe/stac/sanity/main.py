#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: BSD3

"""
Check the corpus for any consistency problems
"""

from __future__ import print_function
from collections import defaultdict
from itertools import chain
import argparse
import codecs
import copy
import glob
import itertools
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

from educe import stac, annotation, graph
from educe.annotation import Schema
from educe.corpus import FileId
from educe.stac.corpus import METAL_REVIEWERS, METAL_STR
from educe.stac.util.annotate import schema_text
from educe.stac.util.context import Context
import educe.stac.corenlp as stac_corenlp
import educe.stac.graph as egr
import educe.util

from .report import *
from .checks import *


STAC_REVIEWERS = METAL_REVIEWERS
STAC_GLOBS = {"data/pilot": "pilot*",
              "data/socl-season1": "s1-league*-game*",
              "data/socl-season2": "s2-*"}


def first_or_none(xs):
    """
    Return the first element or None if there isn't one
    """
    l = list(itertools.islice(xs, 1))
    return l[0] if l else None

# ----------------------------------------------------------------------
# glozz errors
# ----------------------------------------------------------------------


class BadIdItem(ContextItem):
    def __init__(self, doc, contexts, anno, expected_id):
        self.anno = anno
        self.expected_id = expected_id
        ContextItem.__init__(self, doc, contexts)

    def text(self):
        about = summarise_anno(self.doc)(self.anno)
        local_id = self.anno.local_id()
        return ['%s (%s, expect %s)' % (about, local_id, self.expected_id)]


def bad_ids(inputs, k):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    bad = []
    for x in doc.annotations():
        actual_id = x.local_id()
        expected_id = x.metadata['author'] + '_' + x.metadata['creation-date']
        if actual_id != expected_id:
            bad.append(BadIdItem(doc, contexts, x, expected_id))
    return bad


class DuplicateItem(ContextItem):
    def __init__(self, doc, contexts, anno, others):
        self.anno = anno
        self.others = others
        ContextItem.__init__(self, doc, contexts)

    def text(self):
        d = self.anno
        others = self.others
        tgt_txt = summarise_anno(self.doc)
        id_str = str(d)
        id_padding = ' ' * len(id_str)
        variants = map(tgt_txt, others)
        lines = ["%s: %s" % (id_str, variants[0])]
        for v in variants[1:]:
            lines.append("%s  %s" % (id_padding, v))
        return lines


def duplicate_annotations(inputs, k):
    """
    Multiple annotations with the same local_id()
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    annos = defaultdict(list)
    for x in doc.annotations():
        annos[x.local_id()].append(x)
    return [DuplicateItem(doc, contexts, k,v)
            for k,v in annos.items() if len(v) > 1]


class OverlapItem(ContextItem):
    def __init__(self, doc, contexts, anno, overlaps):
        self.anno = anno
        self.overlaps = overlaps
        ContextItem.__init__(self, doc, contexts)

    def annotations(self):
        return [self.anno]

    def text(self):
        def id_and_span(d):
            return d.local_id() + ' ' + str(d.span)
        ty = self.anno.type
        info = id_and_span(self.anno)
        overlap_str = ', '.join(map(id_and_span, self.overlaps))
        return ['[%s] %s\tvs %s' % (ty, info, overlap_str)]

    def html(self):
        tgt_html = summarise_anno_html(self.doc, self.contexts)
        parent = ET.Element('span')
        html_anno_id(parent, self.anno)
        html_span(parent, ' ')
        tgt_html(parent, self.anno)
        html_span(parent, ' vs')
        for olap in self.overlaps:
            html_br(parent)
            olap_span = html_span(parent, attrib={'class': 'indented'})
            html_anno_id(olap_span, olap)
            html_span(olap_span, ' ')
            tgt_html(olap_span, olap)
        return parent


def is_type_match(type):
    def fn(anno):
        return anno.type == type
    return fn


def overlapping(inputs, k, is_overlap):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    annos = [x for x in doc.units if is_overlap(x)]
    annos2 = copy.copy(annos)
    all_overlaps = {}
    for d in sorted(annos, key=lambda x:x.span):
        overlaps = [d2 for d2 in annos2 if d2 is not d and d2.span.overlaps(d.span)]
        if overlaps:
            annos2.remove(d)
            all_overlaps[d] = overlaps
    return [OverlapItem(doc, contexts, d, os) for d,os in all_overlaps.items()]

def overlapping_structs(inputs, k):
    return list(chain.from_iterable(overlapping(inputs, k, is_type_match(ty))\
                                    for ty in stac.STRUCTURE_TYPES))

# ----------------------------------------------------------------------
# type errors
# ----------------------------------------------------------------------

def edu_link_item(doc, contexts, g):
    def anno(x):
        return g.annotation(x)
    def info(x):
        return UnitItem(doc, contexts, anno(x))
    return info

def rel_link_item(doc, contexts, g):
    def anno(x):
        return g.annotation(x)
    def info(r):
        links = g.links(r)
        if len(links) != 2:
            raise Exception("Confused: %s does not have exactly 2 links: %s" % (r, links))
        return RelationItem(doc, contexts, anno(r), [])
    return info

def cdu_link_item(doc, contexts, g):
    def anno(x):
        return g.annotation(x)
    def info(r):
        links = g.links(r)
        return SchemaItem(doc, contexts, anno(r), [])
    return info

def search_edus(inputs, k, g, pred):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    mk_item = edu_link_item(doc, contexts, g)
    sorted_edus = g.sorted_first_widest(g.edus())
    return [mk_item(x) for x in sorted_edus if pred(g,contexts,x)]

def search_relations(inputs, k, g, pred):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    mk_item = rel_link_item(doc, contexts, g)
    return [mk_item(r) for r in g.relations() if pred(g,contexts,r)]

def search_cdus(inputs, k, g, pred):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    mk_item = cdu_link_item(doc, contexts, g)
    return [mk_item(r) for r in g.cdus() if pred(g,contexts,r)]

# ----------------------------------------------------------------------
# low-level annotation errors (in glozz)
# ----------------------------------------------------------------------

class UnitItem(ContextItem):
    def __init__(self, doc, contexts, unit):
        self.unit = unit
        ContextItem.__init__(self, doc, contexts)

    def annotations(self):
        return [self.unit]

    def text(self):
        return [summarise_anno(self.doc)(self.unit)]

    def html(self):
        parent = ET.Element('span')
        html_anno_id(parent, self.unit)
        html_span(parent, ' ')
        summarise_anno_html(self.doc, self.contexts)(parent, self.unit)
        return parent

class RelationItem(ContextItem):
    def __init__(self, doc, contexts, rel, naughty):
        self.rel = rel
        self.naughty = naughty
        ContextItem.__init__(self, doc, contexts)

    def annotations(self):
        return [self.rel]

    def text(self):
        source = self.rel.source
        target = self.rel.target
        tgt_text = summarise_anno(self.doc)
        txt1 = tgt_text(source)
        txt2 = tgt_text(target)
        return ["[%s] %s %s -> %s" % (self.rel.type, self.rel.local_id(), txt1, txt2)]

    def html(self):
        tgt_html = summarise_anno_html(self.doc, self.contexts)
        source = self.rel.source
        target = self.rel.target

        parent = ET.Element('span')
        html_span(parent, self.rel.type + ' ')
        html_anno_id(parent, self.rel, bracket=True)
        html_br(parent)

        h_source = html_span(parent, attrib={'class':'indented'})
        tgt_html(h_source, source, naughty=source in self.naughty)
        html_span(h_source, u' ⟶')
        html_br(parent)
        h_target = html_span(parent, attrib={'class':'indented'})
        tgt_html(h_target, target, naughty=target in self.naughty)
        return parent

class SchemaItem(ContextItem):
    def __init__(self, doc, contexts, schema, naughty):
        self.schema = schema
        self.naughty = naughty
        ContextItem.__init__(self, doc, contexts)

    def annotation(self):
        return [self.schema]

    def text(self):
        s = self.schema
        prefix = "[%s] %s" % (s.type, s.local_id())
        tgt_txt = summarise_anno(self.doc)
        if self.naughty:
            txt0 = tgt_txt(self.naughty[0])
            return ["%s: %s..." % (prefix, txt0)]
        else:
            return [prefix]

    def html(self):
        tgt_html = summarise_anno_html(self.doc, self.contexts)
        parent = ET.Element('span')
        tgt_html(parent, self.schema)
        for n in self.naughty:
            html_br(parent)
            n_span = html_span(parent, attrib={'class':'indented'})
            tgt_html(n_span, n)
        return parent

class OffByOneItem(UnitItem):
    def __init__(self, doc, contexts, unit):
        UnitItem.__init__(self, doc, contexts, unit)

    def text(self):
        return [summarise_anno(self.doc)(self.unit)]

    def html(self):
        doc = self.doc
        contexts = self.contexts
        t = self.unit

        parent = ET.Element('span')
        html_anno_id(parent, self.unit)
        html_span(parent, " " + anno_code(t))
        type_span = html_span(parent, '[%s] ' % t.type)

        if t in contexts:
            turn = contexts[t].turn
            turn_info = stac.split_turn_text(doc.text(turn.span))[0]
            turn_splits = turn_info.split(":")
            if len(turn_splits) > 1:
                tid = ET.SubElement(parent, 'b')
                tid.text = turn_splits[0] + ":"
                trest = html_span(parent, ":".join(turn_splits[1:]))
            else:
                html_span(parent, turn_info)

        t_span = t.text_span()
        t_text = doc.text(t_span)
        if t_span.char_start > 0:
            before_idx = t_span.char_start - 1
            before_sp = html_span(parent, doc.text()[before_idx])
            before_sp.attrib['class'] = 'spillover'
        text_sp = html_span(parent, t_text)
        text_sp.attrib['class'] = 'snippet'
        if t_span.char_end < len(doc.text()):
            after_idx = t_span.char_end
            after_sp = html_span(parent, doc.text()[after_idx])
            after_sp.attrib['class'] = 'spillover'
        html_span(parent, ' %s' % t_span)
        return parent

class FeatureItem(ContextItem):
    def __init__(self, doc, contexts, anno, attrs, status='missing'):
        ContextItem.__init__(self, doc, contexts)
        self.anno = anno
        self.attrs = attrs
        self.status = status

    def annotations(self):
        return [self.anno]

    def text(self):
        summary = summarise_anno(self.doc)(self.anno)
        status = self.status.upper()
        attrs = ", ".join(sorted(self.attrs))
        return ["%s | %s %s" % (summary, status, attrs)]

    def html(self):
        tgt_html = summarise_anno_html(self.doc, self.contexts)
        parent = ET.Element('span')
        for attr in sorted(self.attrs):
            attr_span = html_span(parent, attrib={'class':'feature'})
            attr_span.text = attr
            if self.attrs[attr]:
                attr_span.text += " (" + self.attrs[attr] + ")"
            html_span(parent, " ")
        html_span(parent, "in ")
        tgt_html(parent, self.anno)
        return parent

def is_glozz_relation(r):
    return isinstance(r, educe.annotation.Relation)

def is_glozz_unit(r):
    return isinstance(r, educe.annotation.Unit)

def is_glozz_schema(x):
    return isinstance(x, educe.annotation.Schema)

def search_glozz_units(inputs, k, pred):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    return [UnitItem(doc, contexts, u) for u in doc.units if pred(u)]

def is_default(anno):
    return anno.type == 'default'

def is_non_resource(anno):
    return not stac.is_resource(anno)

def is_non_preference(anno):
    return anno.type != 'Preference'

def is_non_du(anno):
    return is_glozz_relation(anno) or\
            (is_glozz_unit(anno) and not stac.is_edu(anno))

def has_non_du_member(anno):
    """
    True if `anno` is a relation that points to another relation,
    or if it's a CDU that has relation members
    """
    if stac.is_relation_instance(anno):
        members = [anno.source, anno.target]
    elif stac.is_cdu(anno):
        members = anno.members
    else:
        return False

    return any(is_non_du(x) for x in members)

def is_blank_edu(anno):
    # ignore spans of len <= 3 because these are often emoticons
    # so we don't typically care if they're unannotated
    return anno.type == 'Segment'

def is_review_edu(anno):
    # ignore spans of len <= 3 because these are often emoticons
    # so we don't typically care if they're unannotated
    return anno.type[:5] == 'FIXME'

def search_anaphora(inputs, k, pred):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    def seek():
        for x in doc.relations:
            if x.type != 'Anaphora': continue
            about = [x.source, x.target]
            naughty = filter(pred, about)
            if naughty:
                yield RelationItem(doc, contexts, x, naughty)
    return list(seek())

def search_in_glozz_schema(inputs, k, type, pred, pred2=None):
    """
    Search for schema whose memmbers satisfy a condition.
    Not to be confused with `search_for_glozz_schema`
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    def seek():
        for x in doc.schemas:
            if x.type != type: continue
            if any(pred(a) for a in x.members):
                naughty = filter(pred2, x.members) if pred2 else []
                yield SchemaItem(doc, contexts, x, naughty)
    return list(seek())

def search_for_glozz_relations(inputs, k, pred, pred2=None):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    def seek():
        for x in filter(pred, doc.relations):
            naughty = filter(pred2, [x.source, x.target]) if pred2 else []
            yield RelationItem(doc, contexts, x, naughty)
    return list(seek())

def search_for_glozz_schema(inputs, k, pred, pred2=None):
    """
    Search for schema that satisfy a condition
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    def seek():
        for x in filter(pred, doc.schemas):
            naughty = filter(pred2, x.members) if pred2 else []
            yield SchemaItem(doc, contexts, x, naughty)
    return list(seek())

def search_resource_groups(inputs, k, pred):
    return search_in_glozz_schema(inputs, k, 'Several_resources', pred, pred)

def search_preferences(inputs, k, pred):
    return search_in_glozz_schema(inputs, k, 'Preferences', pred, pred)

def search_glozz_off_by_one(inputs, k):
    """
    EDUs which have non-whitespace (or boundary) characters
    either on their right or left
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    txt = doc.text()
    return [OffByOneItem(doc, contexts, u) for u in doc.units\
            if stac.is_edu(u) and is_maybe_off_by_one(txt, u)]

def search_for_missing_unit_features(inputs, k):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    missing = []
    for x in doc.units + doc.schemas:
        attrs = { k:None for k in missing_features(doc, x) }
        if attrs:
            missing.append(FeatureItem(doc, contexts, x, attrs))
    return missing

def search_for_missing_discourse_features(inputs, k):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    missing = []
    for x in doc.relations:
        attrs = { k:None for k in missing_features(doc, x) }
        if attrs:
            missing.append(FeatureItem(doc, contexts, x, attrs))
    return missing

def search_for_unexpected_features(inputs, k):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    unexpected = []
    for x in doc.annotations():
        attrs = unexpected_features(doc, x)
        if attrs:
            unexpected.append(FeatureItem(doc, contexts, x, attrs, status='unexpected'))
    return unexpected

def search_for_fixme_features(inputs, k):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    result = []
    for x in doc.annotations():
        attrs = fixme_features(doc, x)
        if attrs:
            result.append(FeatureItem(doc, contexts, x, attrs, status='fixme'))
    return result

# ----------------------------------------------------------------------
# graph errors
# ----------------------------------------------------------------------

class CduOverlapItem(ContextItem):
    def __init__(self, doc, contexts, anno, cdus):
        self.anno = anno
        self.cdus = cdus
        ContextItem.__init__(self, doc, contexts)

    def annotation(self):
        return [self.anno]

    def text(self):
        cdus_str = ', '.join(x.local_id() for x in self.cdus)
        about = summarise_anno(self.doc)(self.anno)
        return ["%s in %s" % (about, cdus_str)]

    def html(self):
        tgt_html = summarise_anno_html(self.doc, self.contexts)
        parent = ET.Element('span')
        tgt_html(parent, self.anno)
        html_span(parent, ' in ')
        html_anno_id(parent, self.cdus[0])
        for cdu in self.cdus[1:]:
            html_span(parent, ', ')
            html_anno_id(parent, cdu)
        return parent

def search_cdu_overlap(inputs, k, g):
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    containers = defaultdict(list)
    for cdu in g.cdus():
        cdu_anno = g.annotation(cdu)
        if not stac.is_cdu(cdu_anno): continue
        for m in g.cdu_members(cdu):
            edu_anno = g.annotation(m)
            containers[edu_anno].append(cdu_anno)
    return [CduOverlapItem(doc, contexts, k,v) for k,v in containers.items() if len(v) > 1]

def is_arrow_inversion(g,contexts,r):
    """
    Relation in a graph that traverse a CDU boundary
    """
    n1, n2 = g.links(r)
    is_rel = stac.is_relation_instance(g.annotation(r))
    span1 = g.annotation(n1).text_span()
    span2 = g.annotation(n2).text_span()
    return is_rel and span1 > span2

def is_weird_qap(g,contexts,r):
    """
    Relation in a graph that represent a question answer pair
    which either does not start with a question, or which ends
    in a question
    """
    n1, n2 = g.links(r)
    is_qap = g.annotation(r).type == 'Question-answer_pair'
    span1 = g.annotation(n1).text_span()
    span2 = g.annotation(n2).text_span()
    final1 = g.doc.text(span1)[-1]
    final2 = g.doc.text(span2)[-1]
    def is_punc(x):
        return x in [".","?"]
    is_weird1 = is_punc(final1) and final1 != "?"
    is_weird2 = final2 == "?"
    return is_qap and (is_weird1 or is_weird2)

def rfc_violations(inputs, k, g):
    """
    Repackage right frontier contraint violations in a somewhat
    friendlier way
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    violations = g.right_frontier_violations().items()
    return map(rel_link_item(doc, contexts, g),
               chain.from_iterable(ls for v,ls in violations))

def filter_matches(unit, other_units):
    ty1 = rough_type(unit)
    def is_me(unit2):
        # they won't have the same identifiers because the reconstructed
        # aa files are regenerated, and have different glozz identifiers
        return unit2.span == unit.span and rough_type(unit2) == ty1
    return filter(is_me, other_units)

# ----------------------------------------------------------------------
# cross-check errors
# ----------------------------------------------------------------------


class MissingDocumentException(Exception):
    def __init__(self, k):
        self.k = k

    def __str__(self):
        return repr(self.k)


class MissingItem(ReportItem):
    missing_status = 'DELETED'
    excess_status = 'ADDED'
    status_len = max(len(missing_status), len(excess_status))

    def __init__(self, status, doc1, contexts1, unit, doc2, contexts2, approx):
        self.status = status
        self.doc1 = doc1
        self.contexts1 = contexts1
        self.unit = unit
        self.doc2 = doc2
        self.contexts2 = contexts2
        self.approx = approx
        # things that can be inferred from context
        self.roughtype = rough_type(self.unit)
        self.type = self.unit.type
        self.span = self.unit.span
        self.txt = doc1.text(self.span)
        ReportItem.__init__(self)

    def text(self):
        descr = 'EDU %s [%s]' % (self.span, self.type) if self.type == 'EDU'\
            else '%s annotation %s' % (self.type, self.span)
        if self.status == self.missing_status and self.approx:
            related = ', but has ' + ', '.join(a.type for a in self.approx)
        else:
            related = ''
        status = self.status.ljust(self.status_len)
        return ['%s %s%s: {%s}' % (status, descr, related, self.txt)]

    def html(self):
        tgt_html = summarise_anno_html(self.doc1, self.contexts1)
        parent = ET.Element('span')
        status_sp = html_span(parent, self.status + ' ')
        status_sp.attrib['class'] = 'missing'\
                if self.status == self.missing_status else 'excess'
        rtype_sp = html_span(parent, self.roughtype + ' ')
        html_anno_id(parent, self.unit, bracket=True)
        html_br(parent)
        h_unit = html_span(parent, attrib={'class': 'indented'})
        tgt_html(h_unit, self.unit)
        if self.approx:
            html_br(parent)
            approx_str = 'NB: but we have: ' + ', '.join(a.type for a in self.approx)
            html_span(parent, approx_str)
        return parent


class IdMismatch(ContextItem):
    def __init__(self, doc, contexts, unit1, unit2):
        self.unit1 = unit1
        self.unit2 = unit2
        ContextItem.__init__(self, doc, contexts)

    def annotations(self):
        return [self.unit1, self.unit2]

    def text(self):
        ty = self.unit1.type
        span = self.unit1.span
        id1 = self.unit1.local_id()
        id2 = self.unit2.local_id()
        return ['%s %s %s vs %s' % (ty, span, id1, id2)]

    def html(self):
        tgt_html = summarise_anno_html(self.doc, self.contexts)
        parent = ET.Element('span')
        tgt_html(parent, self.unit1)
        html_br(parent)
        html_span(parent, 'expected: ')
        html_anno_id(parent, self.unit1)
        html_span(parent, ', got: ')
        html_anno_id(parent, self.unit2)
        return parent


def cross_check_units(inputs, k1, k2, status):
    """
    Return tuples for certain corpus[k1] units
    not present in corpus[k2]
    """
    corpus = inputs.corpus
    if k1 not in corpus:
        raise MissingDocumentException(k1)
    if k2 not in corpus:
        raise MissingDocumentException(k2)
    doc1 = corpus[k1]
    doc2 = corpus[k2]
    contexts1 = inputs.contexts[k1]
    contexts2 = inputs.contexts[k2]
    missing = defaultdict(list)
    for unit in doc1.units:
        if stac.is_structure(unit) or stac.is_edu(unit):
            if not filter_matches(unit, doc2.units):
                rtype = rough_type(unit)
                approx = [x for x in doc2.units if x.span == unit.span]
                missing[rtype].append(MissingItem(status, doc1, contexts1,
                                                  unit,
                                                  doc2, contexts2, approx))
    return missing


def check_unit_ids(inputs, k1, k2):
    corpus = inputs.corpus
    if k1 not in corpus:
        raise MissingDocumentException(k1)
    if k2 not in corpus:
        raise MissingDocumentException(k2)
    doc1 = corpus[k1]
    doc2 = corpus[k2]
    contexts1 = inputs.contexts[k1]
    mismatches = []
    for unit1 in doc1.units:
        id1 = unit1.local_id()
        matches = filter_matches(unit1, doc2.units)
        if len(matches) > 1:
            print("WARNING: More than one match in check_unit_ids",
                  k1, k2, unit1.local_id(), file=sys.stderr)
        mismatches.extend(IdMismatch(doc1, contexts1, unit1, unit2)\
                          for unit2 in matches if unit2.local_id() != id1)
    return mismatches

# ----------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------


def anno_code(t):
    """
    Short code providing a clue what the annotation is
    """
    if is_glozz_relation(t):
        return 'r'
    elif stac.is_edu(t):
        return 'e'
    elif is_glozz_unit(t):
        return 'u'
    elif is_glozz_schema(t):
        return 's'
    else:
        return '???'


def summarise_anno_html(doc, contexts):
    def tgt_html(grandparent, t, naughty=False):
        def tid(x):
            if x in contexts:
                tid_str = contexts[x].turn.features['Identifier']
                return int(tid_str) if tid_str else None
            else:
                return None

        parent = html_span(grandparent)
        html_span(parent, anno_code(t))
        type_span = html_span(parent, '[%s] ' % t.type)
        if naughty:
            type_span.attrib['class'] = 'naughty'

        if t in contexts:
            turn = contexts[t].turn
            turn_info = stac.split_turn_text(doc.text(turn.span))[0]
            turn_splits = turn_info.split(":")
            if len(turn_splits) > 1:
                tid = ET.SubElement(parent, 'b')
                tid.text = turn_splits[0] + ":"
                trest = html_span(parent, ":".join(turn_splits[1:]))
            else:
                html_span(parent, turn_info)

        if not stac.is_relation_instance(t):
            t_span = t.text_span()
            if t_span is None:
                t_text = "(NO CONTENT?)"
            elif isinstance(t, Schema):
                t_text = schema_text(doc, t)
            else:
                t_text = doc.text(t_span)
            if stac.is_cdu(t):
                tids = [x for x in map(tid, t.terminals()) if x]
                if tids:
                    tspan = ET.SubElement(parent, 'b')
                    min_tid = min(tids)
                    max_tid = max(tids)
                    if min_tid == max_tid:
                        tspan.text = "%d: " % min_tid
                    else:
                        tspan.text = "%d-%d: " % (min_tid, max_tid)
            text_sp = html_span(parent, snippet(t_text, 100))
            text_sp.attrib['class'] = 'snippet'
            html_span(parent, ' %s' % t_span)
        return parent
    return tgt_html


def snippet(txt, l=50):
    if len(txt) > l:
        return txt[:l] + "..."
    else:
        return txt


def summarise_anno(doc, light=False):
    def tgt_txt(t):

        tag = anno_code(t)

        if light:
            tagged_type = ''
        else:
            tagged_type = '%s[%s]' % (tag, t.type)

        if stac.is_relation_instance(t):
            return tagged_type
        else:
            sp = t.text_span()
            txt = doc.text(sp)
            return '%s {%s} %s' % (tagged_type, snippet(txt, 20), sp)
    return tgt_txt

# ---------------------------------------------------------------------
# error reporting
# ---------------------------------------------------------------------


def create_dirname(path):
    """
    Create the directory beneath a path if it does not exist
    """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def mk_microphone(report, k, err_type, severity):
    def f(header, xs, noisy=False):
        return report.report(k, err_type, severity, header, xs, noisy)
    return f

# ---------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------


def cross_check_against(inputs, k1, stage='unannotated'):
    corpus = inputs.corpus
    annotator = None if stage == 'unannotated' else k1.annotator
    k2 = FileId(doc=k1.doc, subdoc=k1.subdoc, stage=stage, annotator=annotator)
    try:
        missing = cross_check_units(inputs, k2, k1, MissingItem.missing_status)
        excess = cross_check_units(inputs, k1, k2, MissingItem.excess_status)
        mismatches = check_unit_ids(inputs, k1, k2)
        missing_excess = []
        for ty, vs in missing.items():
            missing_excess.extend(vs)
        for ty, vs in excess.items():
            missing_excess.extend(vs)

        def start_antiwidth(sp):
            # so we can sort first, widest first
            return (sp.char_start, 0 - sp.char_end)

        def first_widest(x):
            return start_antiwidth(x.span)

        return (sorted(missing_excess, key=first_widest), mismatches)
    except MissingDocumentException as e:
        print("ARGH! Can't cross-check ", e.k, sys.stderr)
        return ({}, {})


def check_glozz_errors(inputs, k):
    missing_excess, mismatches = cross_check_against(inputs, k)

    squawk = mk_microphone(inputs.report, k, 'GLOZZ', sv_ERROR)
    quibble = mk_microphone(inputs.report, k, 'GLOZZ', sv_WARNING)

    squawk('unexpected annotation ids found',
           bad_ids(inputs, k))

    squawk('duplicate annotation ids found',
           duplicate_annotations(inputs, k))

    squawk('overlapping structural elements',
           overlapping_structs(inputs, k))

    squawk('overlapping EDUs',
           overlapping(inputs, k, stac.is_edu))

    squawk('fixed-span items added/deleted/moved',
           missing_excess)

    squawk('id mismatches', mismatches)

    quibble('EDU boundary may be off by one',
            search_glozz_off_by_one(inputs, k))

    if k.stage == 'discourse':
        missing_excess, mismatches =\
            cross_check_against(inputs, k, stage='units')
        squawk('[DISCOURSE v. UNIT] fixed-span items added/deleted/moved',
               missing_excess)
        squawk('[DISCOURSE v. UNIT] id mismatches',
               mismatches)


def check_annotation_errors(inputs, k):
    squawk = mk_microphone(inputs.report, k, 'ANNOTATION', sv_ERROR)

    if k.stage == 'units':
        squawk('EDU missing annotations',
               search_glozz_units(inputs, k, is_blank_edu))

        squawk('EDU annotation needs review',
               search_glozz_units(inputs, k, is_review_edu))
        squawk('Missing features',
               search_for_missing_unit_features(inputs, k))

    if k.stage == 'discourse':
        contexts = inputs.contexts[k]
        squawk('CDU spanning dialogue boundaries',
               search_for_glozz_schema(inputs, k, is_cross_dialogue(contexts)))
        squawk('relation across dialogue boundaries',
               search_for_glozz_relations(inputs, k,
                                          is_cross_dialogue(contexts)))
        squawk('relation missing annotations',
               search_for_glozz_relations(inputs, k, is_default))
        squawk('schema missing annotations',
               search_for_glozz_schema(inputs, k, is_default))
        squawk('relation missing features',
               search_for_missing_discourse_features(inputs, k))

    squawk('Unexpected features',
           search_for_unexpected_features(inputs, k))

    squawk('Features need review',
           search_for_fixme_features(inputs, k))


def check_type_errors(inputs, k):
    squawk = mk_microphone(inputs.report, k, 'TYPE', sv_ERROR)

    squawk('relations with non-DU endpoints',
           search_for_glozz_relations(inputs, k, has_non_du_member, is_non_du))

    squawk('CDUs with non-DU members',
           search_for_glozz_schema(inputs, k, has_non_du_member, is_non_du))

    squawk('Anaphora with non-Resource endpoints',
           search_anaphora(inputs, k, is_non_resource))

    squawk('Resource group with non-Resource members',
           search_resource_groups(inputs, k, is_non_resource))

    squawk('Preference group with non-Preference members',
           search_preferences(inputs, k, is_non_preference))


def horrible_context_kludge(graph, simplified_graph, contexts):
    # FIXME: this is pretty horrible
    #
    # Problem is that simplified_graph is a deepcopy of
    # the original (see implementation in educe without_cdus),
    # which on the one hand is safer in some ways, but on the
    # other hand means that we can't look up annotations in the
    # original contexts dictionary.
    #
    # All this horribleness could be avoided if we had
    # persistent data structures everywhere :-(
    simplified_contexts = {}
    for n in simplified_graph.edus():
        s_anno = simplified_graph.annotation(n)
        o_anno = graph.annotation(n)
        if o_anno in contexts:
            simplified_contexts[s_anno] = contexts[o_anno]
    return simplified_contexts


def check_graph_errors(inputs, k):
    if k.stage != 'discourse':
        return

    doc = inputs.corpus[k]
    graph = egr.Graph.from_doc(inputs.corpus, k)
    contexts = inputs.contexts[k]

    violations = list(graph.right_frontier_violations())

    squawk = mk_microphone(inputs.report, k, 'GRAPH', sv_ERROR)
    quibble = mk_microphone(inputs.report, k, 'GRAPH', sv_WARNING)

    squawk('CDU punctures found',
           search_relations(inputs, k, graph, is_puncture))

    squawk('EDU in more than one CDU',
           search_cdu_overlap(inputs, k, graph))

    squawk('Speaker Acknowledgement to themself',
           search_relations(inputs, k, graph, is_weird_ack))

    quibble('weird QAP (non "? -> .")',
            search_relations(inputs, k, graph, is_weird_qap))

    quibble('possible arrow inversion',
            search_relations(inputs, k, graph, is_arrow_inversion),
            noisy=True)

    #quibble('possible Right Frontier Constraint violation',
    #       rfc_violations(inputs, k, graph),
    #       noisy=True)

    simplified_doc = copy.deepcopy(doc)
    simplified_inputs = copy.copy(inputs)
    simplified_inputs.corpus = {k: simplified_doc}
    simplified_graph = egr.Graph.from_doc(simplified_inputs.corpus, k)
    simplified_graph.strip_cdus(sloppy=True)
    simplified_inputs.contexts =\
        {k: horrible_context_kludge(graph, simplified_graph, contexts)}

    quibble('non dialogue-initial EDUs without incoming links',
            search_edus(simplified_inputs, k,
                        simplified_graph, is_disconnected))

# ---------------------------------------------------------------------
# running the checks
# ---------------------------------------------------------------------


def sanity_check_order(k):
    """
    We want to sort file id by order of

    1. doc
    2. subdoc
    3. annotator
    4. stage (unannotated < unit < discourse)

    The important bit here is the idea that we should maybe
    group unit and discourse for 1-3 together
    """

    def stage_num(s):
        if s == 'unannotated':
            return 0
        elif s == 'units':
            return 1
        elif s == 'discourse':
            return 2
        else:
            return 3

    return (k.doc, k.subdoc, k.annotator, stage_num(k.stage))


def run_checks(inputs, k):
    check_glozz_errors(inputs, k)
    check_annotation_errors(inputs, k)
    check_type_errors(inputs, k)
    check_graph_errors(inputs, k)

# ---------------------------------------------------------------------
# copy stanford parses and stylesheet
# (could be handy for a sort of global dashboard view)
# ---------------------------------------------------------------------


def copy_parses(settings):
    corpus = settings.corpus
    output_dir = settings.output_dir

    docs = set(k.doc for k in settings.corpus)
    for doc in docs:
        subdocs = set(k.subdoc for k in settings.corpus if k.doc == doc)
        if subdocs:
            k = FileId(doc=doc,
                       subdoc=list(subdocs)[0],
                       stage=None,
                       annotator=None)
            i_style_dir = os.path.dirname(stac_corenlp.parsed_file_name(k, settings.corpus_dir))
            o_style_dir = os.path.dirname(stac_corenlp.parsed_file_name(k, output_dir))
            i_style_file = os.path.join(i_style_dir, 'CoreNLP-to-HTML.xsl')
            if os.path.exists(i_style_file):
                if not os.path.exists(o_style_dir):
                    os.makedirs(o_style_dir)
                shutil.copy(i_style_file, o_style_dir)
        for subdoc in subdocs:
            k = FileId(doc=doc, subdoc=subdoc, stage=None, annotator=None)
            i_file = stac_corenlp.parsed_file_name(k, settings.corpus_dir)
            o_file = stac_corenlp.parsed_file_name(k, output_dir)
            o_dir = os.path.dirname(o_file)
            if os.path.exists(i_file):
                if not os.path.exists(o_dir):
                    os.makedirs(o_dir)
                shutil.copy(i_file, o_dir)

# ---------------------------------------------------------------------
# generate graphs
# ---------------------------------------------------------------------


def generate_graphs(settings):
    discourse_only = [k for k in settings.corpus if k.stage == 'discourse']
    report = settings.report

    # generate dot files
    for k in discourse_only:
        try:
            g = egr.DotGraph(egr.Graph.from_doc(settings.corpus, k))
            dot_file = report.subreport_path(k, '.dot')
            create_dirname(dot_file)
            if g.get_nodes():
                with codecs.open(dot_file, 'w', encoding='utf-8') as f:
                    print(g.to_string(), file=f)
        except graph.DuplicateIdException:
            warning = "Couldn't graph %s because it has duplicate annotation ids" % doc_file
            print(warning, file=sys.stderr)

    # attempt to graphviz them
    try:
        print("Generating graphs... (you can safely ^-C here)", file=sys.stderr)
        for k in discourse_only:
            dot_file = report.subreport_path(k, '.dot')
            svg_file = report.subreport_path(k, '.svg')
            if os.path.exists(dot_file) and settings.draw:
                os.system('dot -T svg -o %s %s' % (svg_file, dot_file))
    except Exception as e:
        print("Couldn't run graphviz. (%s)", file=sys.stderr)
        print("You should install it for easier sanity check debugging.", file=sys.stderr)

# ---------------------------------------------------------------------
# index
# ---------------------------------------------------------------------

def add_element(settings, k, html, sep, descr, mk_path):
    abs_p = mk_path(k, settings.output_dir)
    rel_p = mk_path(k, '.')
    if os.path.exists(abs_p):
        sp = ET.SubElement(html, 'span')
        sp.text = sep
        h_a = ET.SubElement(html, 'a', href=rel_p)
        h_a.text = descr

def issues_descr(report, k):
    contents = "issues" if report.has_errors(k) else "warnings"
    return "%s %s" % (k.stage, contents)

def write_index(settings):
    report = settings.report
    corpus = settings.corpus

    htree = ET.Element('html')

    h_general_hdr = ET.SubElement(htree, 'h2')
    h_general_hdr.text = 'general'
    h_report_a = ET.SubElement(htree, 'a', href='report.txt')
    h_report_a.text = 'full report'
    h_notabene = ET.SubElement(htree, 'div')
    h_notabene.text = "NB: Try Firefox if Google Chrome won't open the parses"

    annotators = set(k.annotator for k in corpus if k.annotator)
    for anno in sorted(annotators):
        h_anno_hdr = ET.SubElement(htree, 'h2')
        h_anno_hdr.text = anno
        h_list = ET.SubElement(htree, 'ul')
        anno_keys = set(k for k in corpus if k.annotator == anno)
        for doc in sorted(set(k.doc for k in anno_keys)):
            h_li = ET.SubElement(h_list, 'li')
            h_li.text = doc
            h_sublist = ET.SubElement(h_li, 'ul')
            for subdoc in sorted(set(k.subdoc for k in anno_keys if k.doc == doc)):
                k_review = FileId(doc = doc, subdoc = subdoc, annotator = anno, stage = 'review')
                k_units = copy.copy(k_review)
                k_discourse = copy.copy(k_review)
                k_units.stage = 'units'
                k_discourse.stage = 'discourse'
                h_sub_li = ET.SubElement(h_sublist, 'li')
                h_sub_li.text = ' (' + subdoc + ')'
                mk_report_path = lambda k, odir : report.mk_output_path(odir, k, '.report.html')
                mk_svg_path = lambda k, odir : report.mk_output_path(odir, k, '.svg')
                add_element(settings, k_units,     h_sub_li, ' | ', issues_descr(report, k_units), mk_report_path)
                add_element(settings, k_discourse, h_sub_li, ' | ', issues_descr(report, k_discourse), mk_report_path)
                add_element(settings, k_discourse, h_sub_li, ' | ', 'graph' , mk_svg_path)
                add_element(settings, k_review,    h_sub_li, ' | ', 'parses', stac_corenlp.parsed_file_name)

    with open(os.path.join(settings.output_dir, 'index.html'), 'w') as f:
        print(ET.tostring(htree), file=f)

# ---------------------------------------------------------------------
# put it all together
# ---------------------------------------------------------------------

class SanityChecker:
    def __init__(self, args):
        is_interesting = educe.util.mk_is_interesting(args)
        self.corpus_dir = args.corpus
        self.__init_read_corpus(is_interesting, self.corpus_dir)
        self.__init_set_output(args.output)
        self.report = CombinedReport(self.anno_files, self.output_dir, args.verbose)
        self.draw = args.draw

    def __init_read_corpus(self, is_interesting, corpus_dir):
        reader = stac.Reader(corpus_dir)
        all_files = reader.files()
        self.anno_files = reader.filter(all_files, is_interesting)
        interesting = self.anno_files.keys()
        for k1 in interesting:
            k2 = FileId(doc=k1.doc, subdoc=k1.subdoc, stage='unannotated', annotator=None)
            if k2 in all_files:
                self.anno_files[k2] = all_files[k2]
        self.corpus = reader.slurp(self.anno_files, verbose=True)
        self.contexts = { k:Context.for_edus(self.corpus[k]) for k in self.corpus }

    def __init_set_output(self, output):
        if output:
            if os.path.isfile(output):
                sys.exit("Sorry, %s already exists and is not a directory" % output)
            elif not os.path.isdir(output):
                os.makedirs(output)
            self.output_dir = output
            self._output_to_temp = False
        else:
            self.output_dir = tempfile.mkdtemp()
            self._output_to_temp = True

    def output_is_temp(self):
        return self._output_to_temp

    def go(self):
        for k in sorted(self.corpus, key=sanity_check_order):
            run_checks(self, k)
            create_dirname(self.report.subreport_path(k))
            self.report.flush_subreport(k)

        copy_parses(self)
        generate_graphs(self)
        write_index(self)

        output_dir = self.output_dir
        if self.output_is_temp():
            print("See temp directory: %s" % output_dir, file=sys.stderr)
            print("HINT: use --output if you want to specify "
                  "an output directory",
                  file=sys.stderr)
        else:
            print("Fancy results saved in %s" % output_dir, file=sys.stderr)

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def easy_settings(args):
    """
    Modify args to reflect user-friendly defaults.
    (args.doc must be set, everything else expected to be empty)
    """

    if not args.doc:
        raise Exception("no document specified for easy mode")
    args.output = os.path.join("/tmp", "sanity-" + args.doc)

    # figure out where this thing lives
    for sdir in STAC_GLOBS:
        if glob.glob(os.path.join(sdir, args.doc)):
            args.corpus = sdir
    if not args.corpus:
        if not any(os.path.isdir(x) for x in STAC_GLOBS):
            sys.exit("You don't appear in to be in the STAC root dir")
        else:
            sys.exit("I don't know about any document called " + args.doc)

    args.annotator = METAL_STR
    print("Guessing convenience settings:")
    print("stac-check %(corpus)s\"\
 --doc \"%(doc)s\"\
 --annotator \"%(annotator)s\"\
 --output \"%(output)s\"" % args.__dict__)


EASY_SETTINGS_HINT = "Try this: stac-check --doc pilot03"


def main():
    arg_parser = argparse.ArgumentParser(description='Check corpus for '
                                         'potential problems.')
    arg_parser.add_argument('corpus', metavar='DIR', nargs='?')
    arg_parser.add_argument('--output', '-o', metavar='DIR')
    arg_parser.add_argument('--verbose', '-v',
                            action='count',
                            default=0)
    arg_parser.add_argument('--no-draw', action='store_true',
                            dest='draw', default=True,
                            help='Do not draw relations graph')
    educe.util.add_corpus_filters(arg_parser)
    args = arg_parser.parse_args()

    if args.corpus and not os.path.exists(args.corpus):
        sys.exit(EASY_SETTINGS_HINT)
    if not args.corpus:
        if args.doc:
            easy_settings(args)
        else:
            sys.exit(EASY_SETTINGS_HINT)

    checker = SanityChecker(args)
    checker.go()
