'''
Sanity checker: fancy graph-based errors
'''

from __future__ import print_function
from collections import defaultdict
import copy
import itertools

from educe import stac
# from educe.stac.annotation import (COORDINATING_RELATIONS,
#                                    SUBORDINATING_RELATIONS)
from educe.stac.context import sorted_first_widest
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

PAIRS_WHITELIST = [
    # Julie's original list (2017-02-28)
    ('Contrast', 'Comment'),
    ('Narration', 'Result'),
    ('Narration', 'Continuation'),
    ('Parallel', 'Continuation'),
    ('Parallel', 'Background'),
    # additional pairs vetted by Nicholas (2017-03-01)
    ('Comment', 'Acknowledgement'),
    ('Parallel', 'Acknowledgement'),
    ('Question-answer_pair', 'Contrast'),
    ('Question-answer_pair', 'Parallel'),
]
"""pairs of relations that are explicitly allowed between the same
source/target DUs"""
# un-comment if you modify the above whitelist: this catches potential
# typos (tried and tested...)
# ALL_RELATIONS = set(SUBORDINATING_RELATIONS + COORDINATING_RELATIONS)
# assert all(x[0] in ALL_RELATIONS and x[1] in ALL_RELATIONS
#            for x in PAIRS_WHITELIST)

PAIRS_WHITEDICT = defaultdict(set)
for rel1, rel2 in PAIRS_WHITELIST:
    PAIRS_WHITEDICT[rel1].add(rel2)
    PAIRS_WHITEDICT[rel2].add(rel1)
"""Dict of pairwise compatible relations (more useful to check
membership)
"""


def rel_link_item(doc, contexts, gra, rel):
    "return ReportItem for a graph relation"
    return RelationItem(doc, contexts, gra.annotation(rel), [])


def search_graph_edus(inputs, k, gra, pred):
    """
    Return a ReportItem for any EDU within the graph for which some
    predicate is true
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    edu_names = {gra.annotation(name): name for name in gra.edus()}
    sorted_edus = sorted_first_widest(edu_names.keys())
    return [UnitItem(doc, contexts, x)
            for x in sorted_edus if pred(gra, contexts, edu_names[x])]


def search_graph_relations(inputs, k, gra, pred):
    """
    Return a ReportItem for any relation instance within the graph
    for which some predicate is true
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    return [rel_link_item(doc, contexts, gra, x)
            for x in gra.relations() if pred(gra, contexts, x)]


# 2017-03-02 whitelist certain pairs of relations
def search_graph_relations_same_dus(inputs, k, gra, pred):
    """Return a list of ReportItem (one per member of the set) for any
    set of relation instances within the graph for which some predicate
    is True.

    Parameters
    ----------
    inputs : educe.stac.sanity.main.SanityChecker
        SanityChecker, with attributes `corpus` and `contexts`.

    k : FileId
        Identifier of the desired Glozz document.

    gra : educe.stac.graph.Graph
        Graph that corresponds to the discourse structure (?).

    pred : function from (gra, contexts, rel_set) to boolean
        Predicate function.

    Returns
    -------
    report_items : list of ReportItem
        One ReportItem for each relation instance that belongs to a set
        of instances, on the same DUs, where pred is True.
    """
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]
    # group relations that have the same endpoints
    rel_sets = defaultdict(set)
    for rel in gra.relations():
        src, tgt = gra.links(rel)
        # store together relations on the *unordered pair* (src, tgt) ;
        # for each relation, we keep track of which element (src or tgt)
        # comes first in the unordered pair
        if src < tgt:
            upair = tuple([src, tgt])
            udir = 'src_tgt'
        else:
            upair = tuple([tgt, src])
            udir = 'tgt_src'
        rel_sets[upair].add((udir, rel))
    # select sets for which pred is true
    sel_sets = [rels for src_tgt, rels in rel_sets.items()
                if pred(gra, contexts, rels)]
    # generate one relation item for each relation instance in a selected
    # set
    res = [rel_link_item(doc, contexts, gra, x)
           for sel_set in sel_sets
           for udir, x in sel_set]
    return res


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
    return [CduOverlapItem(doc, contexts, ek, ev)
            for ek, ev in containers.items() if len(ev) > 1]


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


def is_dupe_rel(gra, _, rel):
    """
    Relation instance for which there are relation instances
    between the same source/target DUs (regardless of direction)
    """
    src, tgt = gra.links(rel)
    return any(x != rel and
               (gra.rel_links(x) == (src, tgt) or
                gra.rel_links(x) == (tgt, src))
               for x in gra.links(src)
               if stac.is_relation_instance(gra.annotation(x)))


# 2017-03-02 whitelisted pairs of relations
def is_whitelisted_relpair(gra, _, relset):
    """True if a pair of instance relations is in `PAIRS_WHITELIST`.

    Parameters
    ----------
    gra : Graph
        Graph for the discourse structure.

    contexts : TODO
        TODO

    relset : set of relation instances
        Set of relation instances on the same DUs ; each instance is a
        pair (udir, rel), where:
        udir is one of {'src_tgt', 'tgt_src'} and
        rel is the identifier of a relation.

    Returns
    -------
    res : boolean
        True if relset is a pair of relation instances with the same
        direction and the corresponding pair of relations is explicitly
        allowed in the whitelist.
    """
    # we currently do not whitelist sets of more than two relation
    # instances, plus they need to have the same direction
    if ((len(relset) != 2 or
         len(set(udir for udir, rel in relset)) != 1)):
        return False
    # PAIRS_WHITEDICT is symmetric:
    # rel_a in PAIRS_WHITEDICT[rel_b] iff rel_b in PAIRS_WHITEDICT[rel_a]
    rel_a, rel_b = list(gra.annotation(x).type for udir, x in relset)
    return rel_a in PAIRS_WHITEDICT[rel_b]


def is_bad_relset(gra, contexts, relset):
    """True if a set of relation instances has more than one member
    and it is not whitelisted.

    Parameters
    ----------
    gra : Graph
        Graph for the discourse structure.

    contexts : TODO
        TODO

    relset : set of relation instances
        Set of relation instances on the same DUs ; each instance is a
        pair (udir, rel), where:
        udir is one of {'src_tgt', 'tgt_src'} and
        rel is the identifier of a relation.

    Returns
    -------
    res : boolean
        True if relset contains more than one element and
        `is_whitelisted_relpair` returns False.
    """
    return (len(relset) > 1 and
            not is_whitelisted_relpair(gra, contexts, relset))
# end WIP whitelist


def is_non2sided_rel(gra, _, rel):
    """
    Relation instance which does not have exactly a source and
    target link in the graph

    How this can possibly happen is a mystery
    """
    anno = gra.annotation(rel)
    return (stac.is_relation_instance(anno) and
            len(gra.links(rel)) != 2)


def is_weird_qap(gra, contexts, rel):
    """Return True if rel is a weird Question-Answer Pair relation.

    Parameters
    ----------
    gra : TODO
        Graph?

    contexts  : TODO
        Surrounding context

    rel : TODO
        Relation.

    Returns
    -------
    res : boolean
        True if rel is a relation that represents a question answer pair
        which either does not start with a question, or which ends in a
        question.
    """
    node1, node2 = gra.links(rel)
    is_qap = gra.annotation(rel).type == 'Question-answer_pair'
    anno1 = gra.annotation(node1)
    anno2 = gra.annotation(node2)
    span1 = anno1.text_span()
    span2 = anno2.text_span()
    final1 = gra.doc.text(span1)[-1]
    final2 = gra.doc.text(span2)[-1]

    # 2017-03-30 trade offer
    # don't raise warning for QAP where the first term is a trade offer
    # with the bank or a port and the second is the Server message for
    # its success ;
    # we currently use a precise characterization of the messages
    # involved so as to avoid unintentional captures
    # DIRTY c/c of helpers from is_weird_ack ; a clean refactoring and
    # redefinition of the graph API is much needed
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
    # end DIRTY

    def is_trade_offer_bank_port(anno):
        anno_text = gra.doc.text(anno.text_span())
        return (node_speaker(anno) == 'UI' and
                'made an offer to trade' in anno_text and
                'from the bank or a port.' in anno_text)

    def is_offer_bank_port_accepted(anno):
        anno_text = gra.doc.text(anno.text_span())
        return (node_speaker(anno) == 'Server' and
                'traded' in anno_text and
                ('from the bank.' in anno_text or
                 'from a port.' in anno_text))

    is_trade_offer_bank_port_qap = (is_trade_offer_bank_port(anno1) and
                                    is_offer_bank_port_accepted(anno2))
    # end trade offer

    def is_punc(char):
        "true if a char is a punctuation char"
        return char in [".", "?"]

    is_weird1 = is_punc(final1) and final1 != "?"
    is_weird2 = final2 == "?"
    return (is_qap and (is_weird1 or is_weird2) and
            not is_trade_offer_bank_port_qap)


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


def is_puncture(gra, _, rel):
    """
    Relation in a graph that traverse a CDU boundary
    """
    if not stac.is_relation_instance(gra.annotation(rel)):
        return False
    n_from, n_to = gra.links(rel)
    cdus_from = gra.containing_cdu_chain(n_from)
    cdus_to = gra.containing_cdu_chain(n_to)
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
    """Return a dict from dialogue annotations to subgraphs
    containing at least everything in that dialogue (and
    perhaps some connected items).

    Parameters
    ----------
    k : FileId
        File identifier

    doc : TODO
        TODO

    contexts : dict(Annotation, Context)
        Context for each annotation.

    Returns
    -------
    graphs : dict(Dialogue, Graph)
        Graph for each dialogue.

    Notes
    -----
    MM: I could not find any caller for this function in either educe or
    irit-stac, as of 2017-03-17.
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
    """Return True if an EDU is disconnected from a discourse structure.

    An EDU is considered disconnected unless:

    * it has an incoming link or
    * it has an outgoing Conditional link or
    * it's at the beginning of a dialogue

    In principle we don't need to look at EDUs that are disconnected
    on the outgoing end because (1) it can be legitimate for
    non-dialogue-ending EDUs to not have outgoing links and (2) such
    information would be redundant with the incoming anyway.
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
        has_incoming = any(node == gra.rel_links(x)[1] for x in rel_links)
        has_outgoing_whitelist = any(node == gra.rel_links(r)[0] and
                                     rel_type(r) in BACKWARDS_WHITELIST
                                     for r in rel_links)
        is_at_start = edu.text_span().char_start == first_turn_start
        return not (has_incoming or has_outgoing_whitelist or is_at_start)


# 2017-03-17 check that each CDU has exactly one head DU
# NB: quick and dirty implementation that probably needs a rewrite
def are_single_headed_cdus(inputs, k, gra):
    """Check that each CDU has exactly one head DU.

    Parameters
    ----------
    gra : Graph
        Graph for the discourse structure.

    Returns
    -------
    report_items : list of ReportItem
        List of report items, one per faulty CDU.
    """
    report_items = []
    doc = inputs.corpus[k]
    contexts = inputs.contexts[k]

    # compute the transitive closure of DUs embedded under each CDU
    # * map each CDU to its member EDUs and CDUs, as two lists
    # keys are edge ids eg. 'e_pilot01_07_jhunter_1487683021582',
    # values are node ids eg. 'n_pilot01_07_stac_1464335440'
    cdu2mems = defaultdict(lambda: ([], []))
    for cdu_id in gra.cdus():
        cdu = gra.annotation(cdu_id)
        cdu_members = set(gra.cdu_members(cdu_id))
        cdu2mems[cdu_id] = (
            [x for x in cdu_members if stac.is_edu(gra.annotation(x))],
            [x for x in cdu_members if stac.is_cdu(gra.annotation(x))]
        )
    # * replace each nested CDU in the second list with its member DUs
    # (to first list), and mark CDUs for exploration (to second list) ;
    # repeat until fixpoint, ie. transitive closure complete for each CDU
    while any(v[1] for k, v in cdu2mems.items()):
        for cdu_id, (mem_edus, mem_cdus) in cdu2mems.items():
            for mem_cdu in mem_cdus:
                # switch between the edge and node representations of CDUs:
                # gra.mirror()
                nested_edus, nested_cdus = cdu2mems[gra.mirror(mem_cdu)]
                # add the nested CDU and its EDU members
                cdu2mems[cdu_id][0].append(mem_cdu)
                cdu2mems[cdu_id][0].extend(nested_edus)
                # store CDU members of the nested CDU for exploration
                cdu2mems[cdu_id][1].extend(nested_cdus)
                # delete current nested CDU from list of CDUs to be explored
                cdu2mems[cdu_id][1].remove(mem_cdu)
    # switch to simple dict, forget list of CDUs for exploration
    cdu2mems = {k: v[0] for k, v in cdu2mems.items()}
    # end transitive closure

    for cdu_id in gra.cdus():
        cdu = gra.annotation(cdu_id)
        cdu_mems = set(gra.cdu_members(cdu_id))
        cdu_rec_mems = set(cdu2mems[cdu_id])
        internal_head = dict()
        for cdu_mem in cdu_mems:
            for rel in gra.links(cdu_mem):
                if gra.is_relation(rel):
                    src, tgt = gra.rel_links(rel)
                    # src can be any DU under the current CDU, eg. even
                    # a member of a nested CDU ; this is probably too
                    # loose but we'll see later if we need to refine
                    if src in cdu_rec_mems and tgt in cdu_mems:
                        internal_head[tgt] = src
        unheaded_mems = cdu_mems - set(internal_head.keys())
        if len(unheaded_mems) > 1:
            report_items.append(
                SchemaItem(doc, contexts, cdu, []))
    return report_items
# end connectedness of CDU members

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

    # 2017-03-02 deprecate systematic errors for >1 relation instances
    # on the same DU pair, in favor of: (a) warnings for whitelisted
    # pairs, (b) errors for other configurations
    # squawk('multiple relation instances between the same DU pair',
    #        cand_dupes)
    cand_dupes = search_graph_relations(inputs, k, graph, is_dupe_rel)
    #
    cand_dupes_bad = search_graph_relations_same_dus(
        inputs, k, graph, is_bad_relset)
    squawk('multiple relation instances between the same DU pair',
           cand_dupes_bad)
    # end WIP whitelist

    squawk('Speaker Acknowledgement to self',
           search_graph_relations(inputs, k, graph, is_weird_ack))

    # 2017-03-02 emit a warning for instances of whitelisted pairs of
    # relations
    cand_dupes_good = search_graph_relations_same_dus(
        inputs, k, graph, is_whitelisted_relpair)
    quibble('whitelisted pairs of relation instances between the same DU pair',
            cand_dupes_good,
            noisy=True)
    # check that we don't lose any relation (temporary assertion, should be
    # removed once we are certain there is no hole in its two separate
    # replacements)
    assert (set(x.rel for x in cand_dupes) ==
            set(x.rel for x in cand_dupes_bad + cand_dupes_good))
    # end WIP whitelist

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

    squawk('bizarre relation instance (causes loop after CDUs stripped)',
           search_graph_relations(inputs, k, simplified_graph,
                                  is_non2sided_rel))

    quibble('non dialogue-initial EDUs without incoming links',
            search_graph_edus(simplified_inputs, k,
                              simplified_graph, is_disconnected))

    squawk('CDUs with more than one head',
           are_single_headed_cdus(inputs, k, graph))
