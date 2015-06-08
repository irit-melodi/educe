# -*- coding: utf-8 -*-
"""
Functionality and report types common to sanity checker
"""

from educe import stac
from educe.stac.util.annotate import schema_text
import educe.annotation
from .html import ET
from .report import (ReportItem,
                     html_anno_id,
                     snippet)
from . import html as h

# pylint: disable=too-few-public-methods

# ---------------------------------------------------------------------
# error types
# ---------------------------------------------------------------------


class ContextItem(ReportItem):
    """
    Report item involving EDU contexts
    """
    def __init__(self, doc, contexts):
        self.doc = doc
        self.contexts = contexts
        super(ContextItem, self).__init__()


class UnitItem(ContextItem):
    """
    Errors which involve Glozz unit-level annotations
    """
    def __init__(self, doc, contexts, unit):
        self.unit = unit
        ContextItem.__init__(self, doc, contexts)

    def annotations(self):
        return [self.unit]

    def html(self):
        parent = ET.Element('span')
        html_anno_id(parent, self.unit)
        h.span(parent, ' ')
        summarise_anno_html(self.doc, self.contexts)(parent, self.unit)
        return parent


class RelationItem(ContextItem):
    """
    Errors which involve Glozz relation annotations
    """
    def __init__(self, doc, contexts, rel, naughty):
        self.rel = rel
        self.naughty = naughty
        ContextItem.__init__(self, doc, contexts)

    def annotations(self):
        return [self.rel]

    def html(self):
        tgt_html = summarise_anno_html(self.doc, self.contexts)
        source = self.rel.source
        target = self.rel.target

        parent = ET.Element('span')
        h.span(parent, self.rel.type + ' ')
        html_anno_id(parent, self.rel, bracket=True)
        h.br(parent)

        h_source = h.span(parent, attrib={'class': 'indented'})
        tgt_html(h_source, source, naughty=source in self.naughty)
        h.span(h_source, u' ⟶')
        h.br(parent)
        h_target = h.span(parent, attrib={'class': 'indented'})
        tgt_html(h_target, target, naughty=target in self.naughty)
        return parent


class SchemaItem(ContextItem):
    """
    Errors which involve Glozz schema annotations
    """
    def __init__(self, doc, contexts, schema, naughty):
        self.schema = schema
        self.naughty = naughty
        ContextItem.__init__(self, doc, contexts)

    def annotations(self):
        return [self.schema]

    def html(self):
        tgt_html = summarise_anno_html(self.doc, self.contexts)
        parent = ET.Element('span')
        tgt_html(parent, self.schema)
        for offender in self.naughty:
            h.br(parent)
            n_span = h.span(parent, attrib={'class': 'indented'})
            tgt_html(n_span, offender)
        return parent

# ---------------------------------------------------------------------
# displaying errors
# ---------------------------------------------------------------------


def anno_code(anno):
    """
    Short code providing a clue what the annotation is
    """
    if is_glozz_relation(anno):
        return 'r'
    elif stac.is_edu(anno):
        return 'e'
    elif is_glozz_unit(anno):
        return 'u'
    elif is_glozz_schema(anno):
        return 's'
    else:
        return '???'


def rough_type(anno):
    """
    Return either

        * "EDU"
        * "relation"
        * or the annotation type
    """
    if anno.type == 'Segment' or stac.is_edu(anno):
        return 'EDU'
    elif stac.is_relation_instance(anno):
        return 'relation'
    else:
        return anno.type


def summarise_anno_html(doc, contexts):
    """
    Return a function that creates HTML descriptions of
    an annotation given document and contexts
    """
    def turn_id(anno):
        "turn id if we know about this annotation"
        return stac.turn_id(anno) if anno in contexts else None

    def text(anno):
        "return text content for an annotation"
        t_span = anno.text_span()
        if t_span is None:
            t_text = "(NO CONTENT?)"
        elif is_glozz_schema(anno):
            t_text = schema_text(doc, anno)
        else:
            t_text = doc.text(t_span)
        return t_text

    def turn_range(anno):
        """
        given a CDU return a string representing the turns
        spanned by that CDU (or None if empty)
        """
        if not stac.is_cdu(anno):
            raise ValueError("not a CDU: " + anno)

        tids = [turn_id(y) for y in anno.terminals()]
        tids = [x for x in tids if x]
        if tids:
            min_tid = min(tids)
            max_tid = max(tids)
            if min_tid == max_tid:
                return "%d: " % min_tid
            else:
                return "%d-%d: " % (min_tid, max_tid)
        else:
            return None

    def tgt_html(grandparent, anno, naughty=False):
        """
        Describe the given annotation in HTML and append that
        description to the given HTML grandparent node.
        """
        parent = h.span(grandparent)
        h.span(parent, anno_code(anno))
        type_span = h.span(parent, '[%s] ' % anno.type)
        if naughty:
            type_span.attrib['class'] = 'naughty'

        if anno in contexts:
            turn = contexts[anno].turn
            turn_info = stac.split_turn_text(doc.text(turn.span))[0]
            turn_splits = turn_info.split(":")
            if len(turn_splits) > 1:
                tid = ET.SubElement(parent, 'b')
                tid.text = turn_splits[0] + ":"
                h.span(parent, ":".join(turn_splits[1:]))
            else:
                h.span(parent, turn_info)

        if not stac.is_relation_instance(anno):
            t_text = text(anno)
            if stac.is_cdu(anno):
                trange = turn_range(anno)
                if trange:
                    h.elem(parent, 'b', trange)
            h.span(parent,
                   text=snippet(t_text, 100),
                   attrib={'class': 'snippet'})
            h.span(parent, ' %s' % anno.text_span())
        return parent
    return tgt_html


def summarise_anno(doc, light=False):
    """
    Return a function that returns a short text summary of
    an annotation
    """
    def tgt_txt(anno):
        """
        Return a short text summary of the given annotation
        """
        tag = anno_code(anno)

        if light:
            tagged_type = ''
        else:
            tagged_type = '%s[%s]' % (tag, anno.type)

        if stac.is_relation_instance(anno):
            return tagged_type
        else:
            span = anno.text_span()
            txt = doc.text(span)
            return '%s {%s} %s' % (tagged_type, snippet(txt, 20), span)
    return tgt_txt

# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------


def is_glozz_unit(anno):
    """
    True if the annotation is a Glozz unit
    """
    return isinstance(anno, educe.annotation.Unit)


def is_glozz_relation(anno):
    """
    True if the annotation is a Glozz relation
    """
    return isinstance(anno, educe.annotation.Relation)


def is_glozz_schema(anno):
    """
    True if the annotation is a Glozz schema
    """
    return isinstance(anno, educe.annotation.Schema)


def search_glozz_units(inputs, k, pred):
    """
    Return an item for every unit-level annotation in the given
    document that satisfies some predicate

    :rtype: :py:class:`ReportItem`
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    return [UnitItem(doc, contexts, u) for u in doc.units if pred(u)]


def search_for_glozz_relations(inputs, k, pred,
                               endpoint_is_naughty=None):
    """
    Return a :py:class:`ReportItem` for any glozz relation
    that satisfies the given predicate.

    If `endpoint_is_naughty` is supplied, note which of the
    endpoints can be considered naughty
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]

    res = []
    for anno in doc.relations:
        if not pred(anno):
            continue
        if endpoint_is_naughty is None:
            naughty = []
        else:
            naughty = [x for x in [anno.source, anno.target]
                       if endpoint_is_naughty(x)]
        res.append(RelationItem(doc, contexts, anno, naughty))
    return res


def search_for_glozz_schema(inputs, k, pred,
                            member_is_naughty=None):
    """
    Search for schema that satisfy a condition
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    res = []
    for anno in doc.schemas:
        if not pred(anno):
            continue
        if member_is_naughty is None:
            naughty = []
        else:
            naughty = [x for x in anno.members if member_is_naughty(x)]
        res.append(SchemaItem(doc, contexts, anno, naughty))
    return res


def search_in_glozz_schema(inputs, k, stype, pred,
                           member_is_naughty=None):
    """
    Search for schema whose memmbers satisfy a condition.
    Not to be confused with `search_for_glozz_schema`
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    res = []
    for anno in doc.schemas:
        if anno.type != stype:
            continue
        if any(pred(x) for x in anno.members):
            if member_is_naughty is None:
                naughty = []
            else:
                naughty = [x for x in anno.members if member_is_naughty(x)]
            res.append(SchemaItem(doc, contexts, anno, naughty))
    return res


def is_default(anno):
    """
    True if the annotation has type 'default'
    """
    return anno.type == 'default'
