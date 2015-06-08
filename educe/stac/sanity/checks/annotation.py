"""
STAC sanity-check: annotation oversights
"""

from educe import stac

from .. import html as h
from ..common import (ContextItem,
                      is_default,
                      rough_type,
                      search_glozz_units,
                      search_for_glozz_relations,
                      search_for_glozz_schema,
                      summarise_anno_html)
from ..html import ET
from ..report import (mk_microphone,
                      Severity)


# ---------------------------------------------------------------------
# features
# ---------------------------------------------------------------------


# pylint: disable=too-many-arguments
class FeatureItem(ContextItem):
    """
    Annotations that are missing some feature(s)
    """
    def __init__(self, doc, contexts, anno, attrs, status='missing'):
        super(FeatureItem, self).__init__(doc, contexts)
        self.anno = anno
        self.attrs = attrs
        self.status = status

    def annotations(self):
        return [self.anno]

    def html(self):
        tgt_html = summarise_anno_html(self.doc, self.contexts)
        parent = ET.Element('span')
        for attr in sorted(self.attrs):
            attr_span = h.span(parent, attrib={'class': 'feature'})
            attr_span.text = attr
            if self.attrs[attr]:
                attr_span.text += " (" + self.attrs[attr] + ")"
            h.span(parent, " ")
        h.span(parent, "in ")
        tgt_html(parent, self.anno)
        return parent
# pylint: enable=too-many-arguments

STAC_EXPECTED_FEATURES =\
    {'EDU': frozenset(['Addressee', 'Surface_act']),
     'relation': frozenset(['Argument_scope']),
     'Resource': frozenset(['Status', 'Quantity', 'Correctness', 'Kind']),
     'Preference': frozenset(),
     'Several_resources': frozenset(['Operator']),
     'Complex_discourse_unit': frozenset()}

# lowercase stripped; annotations that are allowed not to have features
STAC_MISSING_FEATURE_TXT_WHITELIST =\
    frozenset([":)",
               ":p",
               ":d",
               ":o",
               ":/",
               ":(",
               "^_^",
               "...",
               "lol"])


def missing_features(doc, anno):
    """
    Return set of attribute names for any expected features that
    may be missing for this annotation
    """
    rty = rough_type(anno)
    txt = doc.text(anno.text_span()).strip().lower()
    if rty == 'EDU' and txt in STAC_MISSING_FEATURE_TXT_WHITELIST:
        return frozenset()
    elif rty in STAC_EXPECTED_FEATURES:
        expected = STAC_EXPECTED_FEATURES[rty]
        present = frozenset(k for k, v in anno.features.items() if v)
        return expected - present
    else:
        return frozenset()


def unexpected_features(_, anno):
    """
    Return set of attribute names for any features that we were
    not expecting to see in the given annotations
    """
    rty = rough_type(anno)
    ignored = frozenset(['Comments', 'highlight'])

    if rty in STAC_EXPECTED_FEATURES:
        expected = STAC_EXPECTED_FEATURES[rty]
        present = frozenset(anno.features.keys())
        leftover = present - ignored - expected
        return {k: anno.features[k] for k in leftover}
    else:
        return {}


def is_fixme(feature_value):
    """
    True if a feature value has a fixme value
    """
    return feature_value and feature_value[:5] == "FIXME"


def search_for_fixme_features(inputs, k):
    """
    Return a ReportItem for any annotations in the document whose
    features have a fixme type
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    res = []
    for anno in doc.annotations():
        attrs = {k: v for k, v in anno.features.items() if is_fixme(v)}
        if attrs:
            res.append(FeatureItem(doc, contexts, anno, attrs, status='fixme'))
    return res


def search_for_missing_unit_feats(inputs, k):
    """
    Return ReportItems for any EDUs and CDUs that are
    missing expected features
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    res = []
    for anno in doc.units + doc.schemas:
        attrs = {k: None for k in missing_features(doc, anno)}
        if attrs:
            res.append(FeatureItem(doc, contexts, anno, attrs))
    return res


def search_for_missing_rel_feats(inputs, k):
    """
    Return ReportItems for any relations that are missing expected
    features
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    res = []
    for anno in doc.relations:
        attrs = {k: None for k in missing_features(doc, anno)}
        if attrs:
            res.append(FeatureItem(doc, contexts, anno, attrs))
    return res


def search_for_unexpected_feats(inputs, k):
    """
    Return ReportItems for any annotations that are have features
    we were not expecting them to have
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    res = []
    for anno in doc.annotations():
        attrs = unexpected_features(doc, anno)
        if attrs:
            res.append(FeatureItem(doc, contexts, anno, attrs,
                                   status='unexpected'))
    return res


# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------


def is_blank_edu(anno):
    """
    True if the annotation looks like it may be an unannotated EDU
    """
    return anno.type == 'Segment'


def is_review_edu(anno):
    """
    True if the annotation has a FIXME tagged type
    """
    return anno.type[:5] == 'FIXME'


def is_cross_dialogue(contexts):
    """
    The units connected by this relation (or cdu)
    do not inhabit the same dialogue.
    """
    def expect_dialogue(anno):
        "true if the annotation should live in a dialogue"
        return stac.is_edu(anno) or stac.is_cdu(anno)

    def dialogue(anno):
        "return the enclosing dialogue for an EDU/CDU"
        if stac.is_edu(anno):
            if anno not in contexts:
                return None
            else:
                return contexts[anno].dialogue
        elif stac.is_cdu(anno):
            dialogues = [dialogue(x) for x in anno.terminals()]
            if dialogues and all(d == dialogues[0] for d in dialogues[1:]):
                return dialogues[0]
            else:
                return None
        else:
            return None

    def is_bad(anno):
        "true if the annotation is crosses a dialogue boundary"
        if stac.is_relation_instance(anno):
            members = [anno.source, anno.target]
        elif stac.is_cdu(anno):
            members = list(anno.members)
        else:
            members = []

        # don't worry about members which are relations
        members = [x for x in members if expect_dialogue(x)]
        dialogues = frozenset(dialogue(x) for x in members)
        if members:
            return len(dialogues) > 1
        else:
            return False
    return is_bad


def run(inputs, k):
    """
    Add any annotation omission errors to the current report
    """
    squawk = mk_microphone(inputs.report, k, 'ANNOTATION', Severity.error)

    if k.stage == 'units':
        squawk('EDU missing annotations',
               search_glozz_units(inputs, k, is_blank_edu))

        squawk('EDU annotation needs review',
               search_glozz_units(inputs, k, is_review_edu))
        squawk('Missing features',
               search_for_missing_unit_feats(inputs, k))

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
               search_for_missing_rel_feats(inputs, k))

    squawk('Unexpected features',
           search_for_unexpected_feats(inputs, k))

    squawk('Features need review',
           search_for_fixme_features(inputs, k))
