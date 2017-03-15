'''
Sanity checker: low-level Glozz errors
'''

from __future__ import print_function
from collections import defaultdict
import copy
import sys

from educe import stac
from educe.stac.context import (sorted_first_widest)
from educe.stac.corpus import (twin_key)
from educe.util import (concat_l)

from .. import html as h
from ..common import (ContextItem,
                      UnitItem,
                      anno_code,
                      rough_type,
                      summarise_anno,
                      summarise_anno_html)
from ..html import ET
from ..report import (mk_microphone,
                      html_anno_id,
                      ReportItem,
                      Severity)

# pylint: disable=too-few-public-methods


class MissingDocumentException(Exception):
    """
    A document we are trying to cross check does not have the
    expected twin
    """
    def __init__(self, k):
        super(MissingDocumentException, self).__init__(str(k))
        self.k = k

# ----------------------------------------------------------------------
# off by one
# ----------------------------------------------------------------------


class OffByOneItem(UnitItem):
    """
    An annotation whose boundaries might be off by one
    """
    def __init__(self, doc, contexts, unit):
        super(OffByOneItem, self).__init__(doc, contexts, unit)

    def html_turn_info(self, parent, turn):
        """
        Given a turn annotation, append a prettified HTML
        representation of the turn text (highlighting parts
        of it, such as the turn number)
        """
        turn_text = self.doc.text(turn.text_span())
        turn_info = stac.split_turn_text(turn_text)[0]
        turn_splits = turn_info.split(":")
        if len(turn_splits) > 1:
            tid = turn_splits[0]
            trest = turn_splits[1:]
            h.elem(parent, 'b', text=tid + ":")
            h.span(parent, text=":".join(trest))
        else:
            h.span(parent, turn_info)

    def html(self):
        doc = self.doc
        contexts = self.contexts
        anno = self.unit

        parent = ET.Element('span')
        html_anno_id(parent, self.unit)
        h.span(parent, " " + anno_code(anno))
        h.span(parent, '[%s] ' % anno.type)

        if anno in contexts:
            self.html_turn_info(parent, contexts[anno].turn)

        t_span = anno.text_span()
        t_text = doc.text(t_span)
        if t_span.char_start > 0:
            before_idx = t_span.char_start - 1
            before_sp = h.span(parent, doc.text()[before_idx])
            before_sp.attrib['class'] = 'spillover'
        h.span(parent, text=t_text, attrib={'class': 'snippet'})
        if t_span.char_end < len(doc.text()):
            after_idx = t_span.char_end
            after_sp = h.span(parent, doc.text()[after_idx])
            after_sp.attrib['class'] = 'spillover'
        h.span(parent, ' %s' % t_span)
        return parent


def is_maybe_off_by_one(text, anno):
    """
    True if an annotation has non-whitespace characters on its
    immediate left/right
    """
    span = anno.text_span()
    start = span.char_start
    end = span.char_end
    start_ok = start == 0 or text[start - 1].isspace()
    end_ok = end == len(text) or text[end].isspace()
    return not (start_ok and end_ok)


def search_glozz_off_by_one(inputs, k):
    """
    EDUs which have non-whitespace (or boundary) characters
    either on their right or left
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    txt = doc.text()
    return [OffByOneItem(doc, contexts, u) for u in doc.units
            if stac.is_edu(u) and is_maybe_off_by_one(txt, u)]


# ----------------------------------------------------------------------
# bad id
# ----------------------------------------------------------------------


class BadIdItem(ContextItem):
    """
    An annotation whose identifier does not match its metadata
    """
    def __init__(self, doc, contexts, anno, expected_id):
        self.anno = anno
        self.expected_id = expected_id
        super(BadIdItem, self).__init__(doc, contexts)

    def text(self):
        about = summarise_anno(self.doc)(self.anno)
        local_id = self.anno.local_id()
        return ['%s (%s, expect %s)' % (about, local_id, self.expected_id)]


def bad_ids(inputs, k):
    """
    Return annotations whose identifiers do not match their metadata
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    bad = []
    for anno in doc.annotations():
        actual_id = anno.local_id()
        expected_id = (anno.metadata['author'] + '_' +
                       anno.metadata['creation-date'])
        if actual_id != expected_id:
            bad.append(BadIdItem(doc, contexts, anno, expected_id))
    return bad


# ----------------------------------------------------------------------
# duplicate id
# ----------------------------------------------------------------------


class DuplicateItem(ContextItem):
    """
    An annotation which shares an id with another
    """
    def __init__(self, doc, contexts, anno, others):
        self.anno = anno
        self.others = others
        ContextItem.__init__(self, doc, contexts)

    def text(self):
        others = self.others
        tgt_txt = summarise_anno(self.doc)
        id_str = str(self.anno)
        id_padding = ' ' * len(id_str)
        variants = [tgt_txt(x) for x in others]
        lines = ["%s: %s" % (id_str, variants[0])]
        for other in variants[1:]:
            lines.append("%s  %s" % (id_padding, other))
        return lines


def duplicate_annotations(inputs, k):
    """
    Multiple annotations with the same local_id()
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    annos = defaultdict(list)
    for anno in doc.annotations():
        annos[anno.local_id()].append(anno)
    return [DuplicateItem(doc, contexts, k, v)
            for k, v in annos.items() if len(v) > 1]

# ----------------------------------------------------------------------
# overlaps
# ----------------------------------------------------------------------


class OverlapItem(ContextItem):
    """
    An annotation whose span overlaps with that of another
    """
    def __init__(self, doc, contexts, anno, overlaps):
        self.anno = anno
        self.overlaps = overlaps
        super(OverlapItem, self).__init__(doc, contexts)

    def annotations(self):
        return [self.anno]

    def html(self):
        tgt_html = summarise_anno_html(self.doc, self.contexts)
        parent = ET.Element('span')
        html_anno_id(parent, self.anno)
        h.span(parent, ' ')
        tgt_html(parent, self.anno)
        h.span(parent, ' vs')
        for olap in self.overlaps:
            h.br(parent)
            olap_span = h.span(parent, attrib={'class': 'indented'})
            html_anno_id(olap_span, olap)
            h.span(olap_span, ' ')
            tgt_html(olap_span, olap)
        return parent


def overlapping(inputs, k, is_overlap):
    """
    Return items for annotations that have overlaps
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    annos = [x for x in doc.units if is_overlap(x)]
    annos2 = copy.copy(annos)
    all_overlaps = {}
    for anno in sorted(annos, key=lambda x: x.text_span()):
        overlaps = [anno2 for anno2 in annos2
                    if anno2 is not anno
                    and anno2.span.overlaps(anno.span)]
        if overlaps:
            annos2.remove(anno)
            all_overlaps[anno] = overlaps
    return [OverlapItem(doc, contexts, anno, olaps)
            for anno, olaps in all_overlaps.items()]


def overlapping_structs(inputs, k):
    """
    Return items for structural annotations that have overlaps
    """
    return concat_l(overlapping(inputs, k,
                                lambda x, t=ty: x.type == t)
                    for ty in stac.STRUCTURE_TYPES)


# ----------------------------------------------------------------------
# cross-check errors
# ----------------------------------------------------------------------

# pylint: disable=too-many-arguments, too-many-instance-attributes
class MissingItem(ReportItem):
    """
    An annotation which is missing in some document twin
    (or which looks like it may have been unexpectedly added)
    """
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
        super(MissingItem, self).__init__()

    def text_span(self):
        """
        Return the span for the annotation in question
        """
        return self.unit.span

    def html(self):
        tgt_html = summarise_anno_html(self.doc1, self.contexts1)
        parent = ET.Element('span')
        status_sp = h.span(parent, self.status + ' ')
        status_sp.attrib['class'] = 'missing'\
            if self.status == self.missing_status else 'excess'
        h.span(parent, self.roughtype + ' ')
        html_anno_id(parent, self.unit, bracket=True)
        h.br(parent)
        h_unit = h.span(parent, attrib={'class': 'indented'})
        tgt_html(h_unit, self.unit)
        if self.approx:
            h.br(parent)
            approx_str = ('NB: but we have: ' +
                          ', '.join(a.type for a in self.approx))
            h.span(parent, approx_str)
        return parent
# pylint: enable=too-many-arguments, too-many-instance-attributes


class IdMismatch(ContextItem):
    """
    An annotation which seems to have an equivalent in some twin
    but with the wrong identifier
    """
    def __init__(self, doc, contexts, unit1, unit2):
        self.unit1 = unit1
        self.unit2 = unit2
        ContextItem.__init__(self, doc, contexts)

    def annotations(self):
        return [self.unit1, self.unit2]

    def html(self):
        tgt_html = summarise_anno_html(self.doc, self.contexts)
        parent = ET.Element('span')
        tgt_html(parent, self.unit1)
        h.br(parent)
        h.span(parent, 'expected: ')
        html_anno_id(parent, self.unit1)
        h.span(parent, ', got: ')
        html_anno_id(parent, self.unit2)
        return parent


def filter_matches(unit, other_units):
    """
    Return any unit-level annotations in `other_units` that look like
    they may be the same as the given annotation
    """
    ty1 = rough_type(unit)

    def is_me(unit2):
        "looks like the given annotation"
        # they won't have the same identifiers because the reconstructed
        # aa files are regenerated, and have different glozz identifiers
        return unit2.span == unit.span and rough_type(unit2) == ty1
    return [x for x in other_units if is_me(x)]


def check_unit_ids(inputs, key1, key2):
    """
    Return annotations that match in the two documents modulo
    identifiers. This might arise if somebody creates a duplicate
    annotation in place and annotates that
    """
    corpus = inputs.corpus
    if key1 not in corpus:
        raise MissingDocumentException(key1)
    if key2 not in corpus:
        raise MissingDocumentException(key2)
    doc1 = corpus[key1]
    doc2 = corpus[key2]
    contexts1 = inputs.contexts[key1]
    mismatches = []
    for unit1 in doc1.units:
        id1 = unit1.local_id()
        matches = filter_matches(unit1, doc2.units)
        if len(matches) > 1:
            print("WARNING: More than one match in check_unit_ids",
                  key1, key2, unit1.local_id(), file=sys.stderr)
        mismatches.extend(IdMismatch(doc1, contexts1, unit1, unit2)
                          for unit2 in matches if unit2.local_id() != id1)
    return mismatches


def cross_check_units(inputs, key1, key2, status):
    """
    Return tuples for certain corpus[key1] units
    not present in corpus[key2]
    """
    corpus = inputs.corpus
    if key1 not in corpus:
        raise MissingDocumentException(key1)
    if key2 not in corpus:
        raise MissingDocumentException(key2)
    doc1 = corpus[key1]
    doc2 = corpus[key2]
    contexts1 = inputs.contexts[key1]
    contexts2 = inputs.contexts[key2]
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


def cross_check_against(inputs, key1, stage='unannotated'):
    """
    Compare annotations with their equivalents on a twin document
    in the corpus
    """
    key2 = twin_key(key1, stage)
    try:
        missing = cross_check_units(inputs, key2, key1,
                                    MissingItem.missing_status)
        excess = cross_check_units(inputs, key1, key2,
                                   MissingItem.excess_status)
        mismatches = check_unit_ids(inputs, key1, key2)
        missing_excess = []
        for vals in missing.values():
            missing_excess.extend(vals)
        for vals in excess.values():
            missing_excess.extend(vals)

        return sorted_first_widest(missing_excess), mismatches
    except MissingDocumentException as oops:
        print("ARGH! Can't cross-check ", oops.k,
              file=sys.stderr)
        return ({}, {})

# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------


def run(inputs, k):
    """
    Add any glozz errors to the current report
    """
    missing_excess, mismatches = cross_check_against(inputs, k)

    squawk = mk_microphone(inputs.report, k, 'GLOZZ', Severity.error)
    quibble = mk_microphone(inputs.report, k, 'GLOZZ', Severity.warning)

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
        missing_excess, mismatches = cross_check_against(
            inputs, k, stage='units')
        squawk('[DISCOURSE v. UNIT] fixed-span items added/deleted/moved',
               missing_excess)
        squawk('[DISCOURSE v. UNIT] id mismatches',
               mismatches)
