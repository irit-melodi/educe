'''
Sanity checker: fancy graph-based errors
'''

from __future__ import print_function
from collections import defaultdict
import copy

from educe import stac
from educe.stac.rfc import (BasicRfc)
import educe.stac.graph as egr

from .. import html as h
from ..common import (ContextItem,
                      UnitItem,
                      RelationItem,
                      SchemaItem,
                      summarise_anno_html)
from ..html import ET
from ..report import (mk_microphone,
                      html_anno_id,
                      Severity)

# pylint: disable=too-few-public-methods

BACKWARDS_WHITELIST = ["Conditional"]
"relations that are allowed to go backwards"


def rel_link_item(doc, contexts, gra, rel):
    "return ReportItem for a graph relation"
    links = gra.links(rel)
    if len(links) != 2:
        raise Exception(("Confused: %s does not have exactly 2 :"
                         "links %s") % (rel, links))
    return RelationItem(doc, contexts, gra.annotation(rel), [])


def search_graph_edus(inputs, k, gra, pred):
    """
    Return a ReportItem for any EDU within the graph for which some
    predicate is true
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    sorted_edus = gra.sorted_first_widest(gra.edus())
    return [UnitItem(doc, contexts, gra.annotation(x))
            for x in sorted_edus if pred(gra, contexts, x)]


def search_graph_relations(inputs, k, gra, pred):
    """
    Return a ReportItem for any relation instance within the graph
    for which some predicate is true
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    return [rel_link_item(doc, contexts, gra, x)
            for x in gra.relations() if pred(gra, contexts, x)]


def search_graph_cdus(inputs, k, gra, pred):
    """
    Return a ReportItem for any CDU in the graph for which
    the given predicate is True
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    return [SchemaItem(doc, contexts, gra.annotation(x), [])
            for x in gra.cdus() if pred(gra, contexts, x)]

# ----------------------------------------------------------------------
# graph errors
# ----------------------------------------------------------------------


class CduOverlapItem(ContextItem):
    """
    EDUs that appear in more than one CDU
    """
    def __init__(self, doc, contexts, anno, cdus):
        self.anno = anno
        self.cdus = cdus
        ContextItem.__init__(self, doc, contexts)

    def annotations(self):
        return [self.anno]

    def html(self):
        tgt_html = summarise_anno_html(self.doc, self.contexts)
        parent = ET.Element('span')
        tgt_html(parent, self.anno)
        h.span(parent, ' in ')
        html_anno_id(parent, self.cdus[0])
        for cdu in self.cdus[1:]:
            h.span(parent, ', ')
            html_anno_id(parent, cdu)
        return parent


def search_graph_cdu_overlap(inputs, k, gra):
    """
    Return a ReportItem for every EDU that appears in more
    than one CDU
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    containers = defaultdict(list)
    for cdu in gra.cdus():
        cdu_anno = gra.annotation(cdu)
        if not stac.is_cdu(cdu_anno):
            continue
        for mem in gra.cdu_members(cdu):
            edu_anno = gra.annotation(mem)
            containers[edu_anno].append(cdu_anno)
    return [CduOverlapItem(doc, contexts, k, v)
            for k, v in containers.items() if len(v) > 1]


def is_arrow_inversion(gra, _, rel):
    """
    Relation in a graph that goes from textual right to left
    (may not be a problem)
    """
    node1, node2 = gra.links(rel)
    is_rel = stac.is_relation_instance(gra.annotation(rel))
    span1 = gra.annotation(node1).text_span()
    span2 = gra.annotation(node2).text_span()
    return is_rel and span1 > span2


def is_weird_qap(gra, _, rel):
    """
    Relation in a graph that represent a question answer pair
    which either does not start with a question, or which ends
    in a question
    """
    node1, node2 = gra.links(rel)
    is_qap = gra.annotation(rel).type == 'Question-answer_pair'
    span1 = gra.annotation(node1).text_span()
    span2 = gra.annotation(node2).text_span()
    final1 = gra.doc.text(span1)[-1]
    final2 = gra.doc.text(span2)[-1]

    def is_punc(char):
        "true if a char is a punctuation char"
        return char in [".", "?"]

    is_weird1 = is_punc(final1) and final1 != "?"
    is_weird2 = final2 == "?"
    return is_qap and (is_weird1 or is_weird2)


def rfc_violations(inputs, k, gra):
    """
    Repackage right frontier contraint violations in a somewhat
    friendlier way
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    violations = BasicRfc(gra).violations()
    return [rel_link_item(doc, contexts, gra, v)
            for v in violations]


def containing_cdu_chain(gra, cdu):
    """
    Given an annotation, return a list which represents its
    containing CDU, the container's container, and forth.
    Return the empty list if no CDU contains this one.
    """
    res = []
    while cdu:
        node = gra.nodeform(cdu)
        res.append(node)
        cdu = gra.containing_cdu(node)
    return res[1:]  # drop the node itself


def is_puncture(gra, _, rel):
    """
    Relation in a graph that traverse a CDU boundary
    """
    if not stac.is_relation_instance(gra.annotation(rel)):
        return False
    n_from, n_to = gra.links(rel)
    cdus_from = containing_cdu_chain(gra, n_from)
    cdus_to = containing_cdu_chain(gra, n_to)
    prefix = len(cdus_from) - len(cdus_to)
    return prefix < 0 or cdus_from[prefix:] != cdus_to


def is_weird_ack(gra, contexts, rel):
    """
    Relation in a graph that represent a question answer pair
    which either does not start with a question, or which ends
    in a question.

    Note the detection process is a lot sloppier when one of
    the endpoints is a CDU. If all EDUs in the CDU are by
    the same speaker, we can check as usual; otherwise, all
    bets are off, so we ignore the relation.

    Note: slightly curried to accept contexts as an argument
    """
    def edu_speaker(anno):
        "return the speaker for an EDU"
        return contexts[anno].speaker() if anno in contexts else None

    def node_speaker(anno):
        "return the designated speaker for an EDU or CDU"
        if stac.is_edu(anno):
            return edu_speaker(anno)
        elif stac.is_cdu(anno):
            speakers = frozenset(edu_speaker(x) for x in anno.terminals())
            if len(speakers) == 1:
                return list(speakers)[0]
            else:
                return None
        else:
            return None

    node1, node2 = gra.links(rel)
    is_ty = gra.annotation(rel).type == 'Acknowledgement'
    anno1 = gra.annotation(node1)
    anno2 = gra.annotation(node2)
    speaker1 = node_speaker(anno1)
    speaker2 = node_speaker(anno2)
    is_talking_to_self = speaker1 and speaker2 and speaker1 == speaker2
    return is_ty and is_talking_to_self


def dialogue_graphs(k, doc, contexts):
    """
    Return a dict from dialogue annotations to subgraphs
    containing at least everything in that dialogue (and
    perhaps some connected items)
    """
    def in_dialogue(d_annos, anno):
        "if the given annotation is in the given dialogue"
        if stac.is_edu(anno):
            return anno in d_annos
        elif stac.is_relation_instance(anno):
            return anno.source in d_annos and anno.target in d_annos
        elif stac.is_cdu(anno):
            return all(t in d_annos for t in anno.terminals())
        else:
            return False

    dialogues = defaultdict(list)
    for anno, ctx in contexts.items():
        dialogues[ctx.dialogue].append(anno)
    graphs = {}
    for dia, annos in dialogues.items():
        keep = lambda x, d_annos=annos: in_dialogue(d_annos, x)
        graphs[dia] = egr.Graph.from_doc({k: doc}, k, pred=keep)
    return graphs


def is_disconnected(gra, contexts, node):
    """
    An EDU is considered disconnected unless:

    * it has an incoming link or
    * it has an outgoing Conditional link
    * it's at the beginning of a dialogue

    In principle we don't need to look at EDUs that are disconnected
    on the outgoing end because (1) it's can be legitimate for
    non-dialogue-ending EDUs to not have outgoing links and (2) such
    information would be redundant with the incoming anyway
    """
    def rel_type(rel):
        "relation type for a given link (string)"
        return gra.annotation(gra.mirror(rel)).type

    edu = gra.annotation(node)
    if edu not in contexts:
        return True
    else:
        ctx = contexts[edu]
        first_turn_span = ctx.dialogue_turns[0].text_span()
        first_turn_text = gra.doc.text(first_turn_span)
        first_turn_pref = stac.split_turn_text(first_turn_text)[0]
        first_turn_start = first_turn_span.char_start + len(first_turn_pref)
        rel_links = [x for x in gra.links(node) if gra.is_relation(x)]
        has_incoming = any(node == gra.links(x)[1] for x in rel_links)
        has_outgoing_whitelist = any(node == gra.links(r)[0] and
                                     rel_type(r) in BACKWARDS_WHITELIST
                                     for r in rel_links)
        is_at_start = edu.text_span().char_start == first_turn_start
        return not (has_incoming or has_outgoing_whitelist or is_at_start)

# ---------------------------------------------------------------------
# run
# ---------------------------------------------------------------------


def horrible_context_kludge(graph, simplified_graph, contexts):
    """
    Given a graph and its copy, and given a context dictionary,
    return a copy of the context dictionary that corresponds to
    the simplified graph.  Ugh
    """
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
    for node in simplified_graph.edus():
        s_anno = simplified_graph.annotation(node)
        o_anno = graph.annotation(node)
        if o_anno in contexts:
            simplified_contexts[s_anno] = contexts[o_anno]
    return simplified_contexts


def run(inputs, k):
    """
    Add any graph errors to the current report
    """
    if k.stage != 'discourse':
        return

    doc = inputs.corpus[k]
    graph = egr.Graph.from_doc(inputs.corpus, k)
    contexts = inputs.contexts[k]

    squawk = mk_microphone(inputs.report, k, 'GRAPH', Severity.error)
    quibble = mk_microphone(inputs.report, k, 'GRAPH', Severity.warning)

    squawk('CDU punctures found',
           search_graph_relations(inputs, k, graph, is_puncture))

    squawk('EDU in more than one CDU',
           search_graph_cdu_overlap(inputs, k, graph))

    squawk('Speaker Acknowledgement to self',
           search_graph_relations(inputs, k, graph, is_weird_ack))

    quibble('weird QAP (non "? -> .")',
            search_graph_relations(inputs, k, graph, is_weird_qap))

    quibble('possible arrow inversion',
            search_graph_relations(inputs, k, graph, is_arrow_inversion),
            noisy=True)

    quibble('possible Right Frontier Constraint violation',
            rfc_violations(inputs, k, graph),
            noisy=True)

    simplified_doc = copy.deepcopy(doc)
    simplified_inputs = copy.copy(inputs)
    simplified_inputs.corpus = {k: simplified_doc}
    simplified_graph = egr.Graph.from_doc(simplified_inputs.corpus, k)
    simplified_graph.strip_cdus(sloppy=True)
    simplified_inputs.contexts =\
        {k: horrible_context_kludge(graph, simplified_graph, contexts)}

    quibble('non dialogue-initial EDUs without incoming links',
            search_graph_edus(simplified_inputs, k,
                              simplified_graph, is_disconnected))
