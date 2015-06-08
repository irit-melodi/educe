"""
STAC sanity-check: type errors
"""

from educe import stac

from ..common import (RelationItem,
                      is_glozz_unit,
                      is_glozz_relation,
                      search_in_glozz_schema,
                      search_for_glozz_relations,
                      search_for_glozz_schema)
from ..report import (mk_microphone,
                      Severity)


def search_anaphora(inputs, k, pred):
    """
    Return a ReportItem for any anaphora annotation in which
    at least one member (not the annotation itself) is
    true with the given predicate
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    res = []
    for anno in doc.relations:
        if anno.type != 'Anaphora':
            continue
        naughty = [x for x in [anno.source, anno.target]
                   if pred(x)]
        if naughty:
            res.append(RelationItem(doc, contexts, anno, naughty))
    return res


def search_resource_groups(inputs, k, pred):
    """
    Return a ReportItem for any Several_resources schema which has
    at least one member for which the predicate is True
    """
    return search_in_glozz_schema(inputs, k, 'Several_resources', pred, pred)


def search_preferences(inputs, k, pred):
    """
    Return a ReportItem for any Preferences schema which has
    at least one member for which the predicate is True
    """
    return search_in_glozz_schema(inputs, k, 'Preferences', pred, pred)


# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------


def is_non_resource(anno):
    """
    True if the annotation is NOT a resource
    """
    return not stac.is_resource(anno)


def is_non_preference(anno):
    """
    True if the annotation is NOT a preference
    """
    return anno.type != 'Preference'


def is_non_du(anno):
    """
    True if the annotation is neither an EDU nor a CDU
    """
    return (is_glozz_relation(anno) or
            (is_glozz_unit(anno) and not stac.is_edu(anno)))


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


def run(inputs, k):
    """
    Add any annotation type errors to the current report
    """
    squawk = mk_microphone(inputs.report, k, 'TYPE', Severity.error)

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
