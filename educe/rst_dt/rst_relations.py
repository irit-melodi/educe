"""Structured inventory of relations used in the RST-DT.

This module provides a structured view of the relation labels used in the
RST-DT, using information from the reference manual [rst-dt-manual]_
and the initial instructions for annotators [rst-dt-instru]_.

References
----------
.. [rst-dt-manual] Carlson, L., & Marcu, D. (2001). Discourse tagging reference manual. ISI Technical Report ISI-TR-545, 54, 56.
.. [rst-dt-instru] Marcu, D. (1999). Instructions for manually annotating the discourse structures of texts. Unpublished manuscript, USC/ISI.
"""

from __future__ import absolute_import, print_function

import os

import nltk.tree

from educe.rst_dt.annotation import SimpleRSTTree
from educe.rst_dt.corpus import Reader
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.pseudo_relations import rewrite_pseudo_rels


# Inventory of classes of rhetorical relations, from subsection 4.1 of
# the [rst-dt-manual]_.
# It maps 18 classes to 54 "representative members" of the 78 (53 mononuclear,
# 25 multinuclear) used in the RST-DT ; 2 multinuclear relations are in fact
# pseudo-relations: "Same-Unit" and "TextualOrganization", each has its own
# class.
# For completeness, I have added 2 more classes: "span" for c-trees, "root"
# for d-trees.
# Note:
# * attribution-n is a proper relation related to, but distinct from,
# "attribution", the "-n" stands for "attribution-negative" rather than
# "attribution from the nucleus"
RELATION_CLASSES = {
    "attribution": ["attribution", "attribution-n"],
    "background": ["background", "circumstance"],
    "cause": ["cause", "consequence"],  # "result" grouped with "cause" below
    "comparison": ["comparison", "preference", "analogy", "proportion"],
    "condition": ["condition", "hypothetical", "contingency", "otherwise"],
    "contrast": ["contrast", "concession"],  # "antithesis" with "contrast"
    "elaboration": ["elaboration-additional", "elaboration-general-specific",
                    "elaboration-part-whole", "elaboration-process-step",
                    "elaboration-object-attribute", "elaboration-set-member",
                    "example", "definition"],
    "enablement": ["purpose", "enablement"],
    "evaluation": ["evaluation", "interpretation", "conclusion", "comment"],
    "explanation": ["evidence", "explanation-argumentative", "reason"],
    "joint": ["list", "disjunction"],
    "manner-means": ["manner", "means"],
    "topic-comment": ["problem-solution", "question-answer",
                      "statement-response", "topic-comment", "comment-topic",
                      "rhetorical-question"],
    "summary": ["summary", "restatement"],
    "temporal": ["temporal-before", "temporal-after", "temporal-same-time",
                 "sequence", "inverted-sequence"],
    "topic-change": ["topic-shift", "topic-drift"],
    # the 25 multinuclear relations include 2 pseudo-relations
    "same-unit": ["same-unit"],
    "textual": ["textualorganization"],
    # add label "span" for completeness, for c-trees
    "span": ["span"],
    # add label "root" for completeness, for d-trees
    "root": ["root"],
}

# groups of relation labels that differ only in the respective nuclearity
# of their arguments ;
# groups of relations are triples of:
# (Mononuclear-satellite, Mononuclear-nucleus, Multinuclear)
# where 0 to 2 slots can be empty (None)
RELATION_REPRESENTATIVES = {
    "analogy": ("analogy", None, "Analogy"),
    # "antithesis": ("antithesis", None, "Contrast")  # see "contrast"
    "attribution": ("attribution", None, None),
    "attribution-n": ("attribution-n", None, None),  # negative attribution
    "background": ("background", None, None),
    "cause": ("result", "cause", "Cause-Result"),  # "result" moved here
    "circumstance": ("circumstance", None, None),
    "comparison": ("comparison", None, "Comparison"),
    "comment": ("comment", None, None),
    "comment-topic": (None, None, "Comment-Topic"),  # linear order of args of "topic-comment" reversed
    "concession": ("concession", None, None),
    "conclusion": ("conclusion", None, "Conclusion"),
    "condition": ("condition", None, None),
    "consequence": ("consequence-s", "consequence-n", "Consequence"),
    "contingency": ("contingency", None, None),
    "contrast": ("antithesis", None, "Contrast"),
    "definition": ("definition", None, None),
    "disjunction": (None, None, "Disjunction"),
    "elaboration-additional": ("elaboration-additional", None, None),
    "elaboration-set-member": ("elaboration-set-member", None, None),
    "elaboration-part-whole": ("elaboration-part-whole", None, None),
    "elaboration-process-step": ("elaboration-process-step", None, None),
    "elaboration-object-attribute": ("elaboration-object-attribute", None, None),
    "elaboration-general-specific": ("elaboration-general-specific", None, None),
    "enablement": ("enablement", None, None),
    "evaluation": ("evaluation-s", "evaluation-n", "Evaluation"),
    "evidence": ("evidence", None, None),
    "example": ("example", None, None),
    "explanation-argumentative": ("explanation-argumentative", None, None),
    "hypothetical": ("hypothetical", None, None),
    "interpretation": ("interpretation-s", "interpretation-n", "Interpretation"),
    "inverted-sequence": (None, None, "Inverted-Sequence"),
    "list": (None, None, "List"),
    "manner": ("manner", None, None),
    "means": ("means", None, None),
    "otherwise": ("otherwise", None, "Otherwise"),
    "preference": ("preference", None, None),
    "problem-solution": ("problem-solution-s", "problem-solution-n", "Problem-Solution"),
    "proportion": (None, None, "Proportion"),
    "purpose": ("purpose", None, None),
    "question-answer": ("question-answer-s", "question-answer-n", "Question-Answer"),
    "reason": ("reason", None, "Reason"),
    "restatement": ("restatement", None, None),
    # "result": ("cause", "result", "Cause-Result")  # see "cause"
    "rhetorical-question": ("rhetorical-question", None, None),
    "same-unit": (None, None, "Same-Unit"),
    "sequence": (None, None, "Sequence"),
    "statement-response": ("statement-response-s", "statement-response-n", "Statement-Response"),
    "summary": ("summary-s", "summary-n", None),
    "temporal-before": (None, "temporal-before", None),
    "temporal-same-time": ("temporal-same-time", "temporal-same-time", "Temporal-Same-Time"),
    "temporal-after": (None, "temporal-after", None),
    "textualorganization": (None, None, "TextualOrganization"),
    "topic-comment": (None, None, "Topic-Comment"),
    "topic-drift": ("topic-drift", None, "Topic-Drift"),
    "topic-shift": ("topic-shift", None, "Topic-Shift"),
    # for completeness (maybe useless)
    "span": "span",
}

# other, less populated dimensions or similarities can link relations:
# * "antithesis" differs from "concession": the latter is characterized by a
#   violated expectation,
# * "attribution-n" is an "attribution" with a negation (negations like "not"
#    but also semantically negative verbs like "deny") in the source
#    (satellite)
# * "background" is weaker than "circumstance": often, events in "background"
#   happen at distinct times whereas events in "circumstance" are somewhat
#   co-temporal,
# * "cause-result" and "consequence" ("Consequence" ~ "Cause-Result",
#   "consequence-n" ~ "result" and "consequence-s" ~ "cause") are similar,
#   the former are for when the causality is perceived as more direct while
#   the latter are for more indirect causal relation.
# * "comment" could be confused with "evaluation" and "interpretation"
# * "Comment-Topic" and "Topic-Comment" are the same relation but the linear
#   order of their arguments is reversed (Comment then Topic or the other way
#   around)
# * "comparison" could be confused with "contrast", but the latter typically
#   contains a contrastive discourse cue (ex: but, however, while) while the
#   former does not
# * "consequence-n" is similar to "result", "consequence-s"
# * the satellite of "elaboration-process-step" is usually realized as a
#   multinuclear "Sequence" relation
# * the satellite of "elaboration-set-member" can be a multinuclear "List"
#   relation where each member elaborates on part of the nucleus
# * "example" should be chosen rather than "elaboration-set-member" if not
#   the other members of the set are not known or specified
# * "explanation-argumentative" differs from "evidence" in that the writer
#   has no intention to convince the reader of a point in the former, and
#   it differs from "reason" because the latter involves the will or
#   intentions of the agent (hence the agent must be animate)
# * "hypothetical" presents a more abstract scenario than "condition"
# * "inverted-sequence" is "sequence" with elements in reverse chronological
#   order
# * "List" is for situations where "comparison", "contrast", or other
#   multinuclear relations
# * "manner" is less "goal-oriented" than "means", describes more the style
#   of an action
# * "preference" compares two situations/acts/events/... and assigns a clear
#   preference for one of those
# * "purpose" differs from "result" in that the satellite is only putative
#   (yet to be achieved) in the former, factual (achieved) in the latter;
#    can be confused with "elaboration-object-attribute-e" but the latter
#    can modify a noun phrase as a relative
# * "restatement" just reiterates the info with slightly different wording,
#   as opposed to e.g. interpretation
# * "temporal-before" is for mononuclear relations, usually the satellite is
#   realized as a subordinate clause that follows the nucleus ;
#   if the second (in the linear order) event happens before the first but
#   the relation is multinuclear, use "Inverted-Sequence".
# * "temporal-after" is for mononuclear relations (see "temporal-before") ;
#   for multinuclear relations with e1 < e2 use "Sequence".
# * "topic-shift" differs from "topic-drift": in the latter, the same elements
#   are in focus whereas it is not the case in the former


# embedded relations explicitly present in the annotation guide
EMBEDDED_RELATIONS = [
    "elaboration-additional-e",
    "elaboration-object-attribute-e",
    "elaboration-set-member-e",
    "interpretation-s-e",
    "manner-e",
]


# mapping from fine-grained to coarse-grained relation labels
FINE_TO_COARSE = {
    "analogy": "comparison",
    "analogy-e": "comparison",
    "antithesis": "contrast",
    "antithesis-e": "contrast",
    "attribution": "attribution",
    "attribution-e": "attribution",
    "attribution-n": "attribution",  # stands for "attribution-negative" !!
    "background": "background",
    "background-e": "background",
    "cause": "cause",  # missing from prev version of mapping (!?)
    "cause-e": "cause",  # origin? corpus: NO, Joty's map: NO
    "cause-result": "cause",
    "circumstance": "background",
    "circumstance-e": "background",
    "comment": "evaluation",
    "comment-e": "evaluation",
    "comment-topic": "topic-comment",
    "comparison": "comparison",
    "comparison-e": "comparison",
    "concession": "contrast",
    "concession-e": "contrast",
    "conclusion": "evaluation",
    "condition": "condition",
    "condition-e": "condition",
    "consequence": "cause",
    "consequence-n": "cause",
    "consequence-n-e": "cause",
    "consequence-s": "cause",
    "consequence-s-e": "cause",
    "contingency": "condition",
    "contrast": "contrast",
    "definition": "elaboration",
    "definition-e": "elaboration",
    "disjunction": "joint",
    "elaboration-additional": "elaboration",
    "elaboration-additional-e": "elaboration",
    "elaboration-e": "elaboration",  # origin? corpus: NO, Joty's map: NO
    "elaboration-general-specific": "elaboration",
    "elaboration-general-specific-e": "elaboration",
    "elaboration-object-attribute": "elaboration",
    "elaboration-object-attribute-e": "elaboration",
    "elaboration-part-whole": "elaboration",
    "elaboration-part-whole-e": "elaboration",
    "elaboration-process-step": "elaboration",
    "elaboration-process-step-e": "elaboration",
    "elaboration-set-member": "elaboration",
    "elaboration-set-member-e": "elaboration",
    "enablement": "enablement",
    "enablement-e": "enablement",
    "evaluation": "evaluation",
    "evaluation-n": "evaluation",
    "evaluation-s": "evaluation",
    "evaluation-s-e": "evaluation",
    "evidence": "explanation",
    "evidence-e": "explanation",
    "example": "elaboration",
    "example-e": "elaboration",
    "explanation-argumentative": "explanation",
    "explanation-argumentative-e": "explanation",
    "hypothetical": "condition",
    "interpretation": "evaluation",
    "interpretation-n": "evaluation",
    "interpretation-s": "evaluation",
    "interpretation-s-e": "evaluation",
    "inverted-sequence": "temporal",
    "list": "joint",
    "manner": "manner-means",
    "manner-e": "manner-means",
    "means": "manner-means",
    "means-e": "manner-means",
    "otherwise": "condition",
    "preference": "comparison",
    "preference-e": "comparison",
    "problem-solution": "topic-comment",
    "problem-solution-n": "topic-comment",
    "problem-solution-s": "topic-comment",
    "proportion": "comparison",
    "purpose": "enablement",
    "purpose-e": "enablement",
    "question-answer": "topic-comment",
    "question-answer-n": "topic-comment",
    "question-answer-s": "topic-comment",
    "reason": "explanation",
    "reason-e": "explanation",
    "restatement": "summary",
    "restatement-e": "summary",
    "result": "cause",
    "result-e": "cause",
    "rhetorical-question": "topic-comment",
    "same-unit": "same-unit",  # pseudo-rel
    "sequence": "temporal",
    "statement-response": "topic-comment",
    "statement-response-n": "topic-comment",
    "statement-response-s": "topic-comment",
    "summary-n": "summary",
    "summary-s": "summary",
    "temporal-after": "temporal",
    "temporal-after-e": "temporal",
    "temporal-before": "temporal",
    "temporal-before-e": "temporal",
    "temporal-same-time": "temporal",
    "temporal-same-time-e": "temporal",
    "textualorganization": "textual",  # pseudo-rel
    "topic-comment": "topic-comment",
    "topic-comment-n": "topic-comment",  # origin? corpus: NO, Joty's map: NO
    "topic-comment-s": "topic-comment",  # origin? corpus: NO, Joty's map: NO
    "topic-drift": "topic-change",
    "topic-shift": "topic-change",
}

# TODO test that we have the same mapping here and in Joty's file
joty_map = dict()
with open('/home/mmorey/melodi/joty/parsing_eval_metrics/RelationClasses.txt') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        fields = line.split(':')
        if len(fields) != 2:
            print(line)
            raise ValueError('gni')
        coarse_lbl = fields[0].strip().lower()
        fine_lbls = [x.strip() for x in fields[1].split(', ')]
        for fine_lbl in fine_lbls:
            joty_map[fine_lbl] = coarse_lbl

print(sorted(set(FINE_TO_COARSE.items()) - set(joty_map.items())))
print(sorted(set(joty_map.items()) - set(FINE_TO_COARSE.items())))
# assert set(FINE_TO_COARSE.items()) == set(relmap.items())
# FIXME: comparison between our mapping and Joty's reveals 2 differences:
# * 1 major: "comment: evaluation" (ours) vs "comment: topic-comment" (Joty)
# * 1 minor: "textualorganization: textual" (ours) vs
# "textualorganization: textualorganization" (joty)

# Examples of TextualOrganization blocks:
# - dateline: wsj_1105: "CHICAGO -", wsj_1377: "SMYRNA, Ga. --"
# - byline: (lots of examples)

# add "root" label for dependency trees
FINE_TO_COARSE["root"] = "root"
# add "span" label (?) for constituency trees
FINE_TO_COARSE["span"] = "span"

RST_RELS_FINE = sorted(FINE_TO_COARSE.keys())
RST_RELS_COARSE = sorted(set(FINE_TO_COARSE.values()))


# WIP
# relative to the educe docs directory
# was: DATA_DIR = '/home/muller/Ressources/'
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    '..', '..',
    'data',  # alt: '..', '..', 'corpora'
)
RST_DIR = os.path.join(DATA_DIR, 'rst_discourse_treebank', 'data')
RST_CORPUS = {
    'train': os.path.join(RST_DIR, 'RSTtrees-WSJ-main-1.0', 'TRAINING'),
    'test': os.path.join(RST_DIR, 'RSTtrees-WSJ-main-1.0', 'TEST'),
    'double': os.path.join(RST_DIR, 'RSTtrees-WSJ-double-1.0'),
}

rst_corpus_dir = RST_CORPUS['train']
rst_reader = Reader(rst_corpus_dir)
rst_corpus = rst_reader.slurp(verbose=True)
ctrees = [doc for doc_key, doc in sorted(rst_corpus.items())]

for doc_key, ctree in sorted(rst_corpus.items()):
    rewrite_pseudo_rels(doc_key, ctree)

raise ValueError('WIP TextualOrganization and Topic-Shift')

# "chain" transform from ctree to dtree (via SimpleRSTTree)
dtrees = [RstDepTree.from_simple_rst_tree(SimpleRSTTree.from_rst_tree(doc))
          for doc_key, doc in sorted(rst_corpus.items())]


# get dependencies
def get_dependencies(dtree):
    """Get dependency triplets from a dependency tree"""
    return [(gov_idx, dep_idx, lbl)
            for dep_idx, (gov_idx, lbl) in enumerate(
                    zip(dtree.heads[1:], dtree.labels[1:]),
                    start=1)]


# examine Same-Unit relations: search for governors on the right
def check_su_right_gov(dtree):
    """TODO"""
    all_deps = get_dependencies(dtree)
    su_roots = set(gov_idx
                   for gov_idx, dep_idx, lbl
                   in all_deps
                   if lbl == 'Same-Unit')
    su_right_govs = [(gov_idx, dep_idx, lbl)
                     for gov_idx, dep_idx, lbl
                     in all_deps
                     if (dep_idx in su_roots and
                         gov_idx > dep_idx)]
    if su_roots:
        print('W: {}\t{} out of {} Same-Unit roots have a right governor'.format(
            dtree.origin.doc, len(su_right_govs), len(su_roots)))
        for gov_idx, dep_idx, lbl in su_right_govs:
            print('\t{}\t{}\t{}'.format(
                gov_idx, dep_idx, dtree.labels[dep_idx]))
    return su_right_govs


# get same-unit pairs, from dtree or ctree
def same_units_deps_from_dtree(dtree):
    """Get same unit dependencies from a dependency tree"""
    return [(gov_idx, dep_idx)
            for dep_idx, gov_idx, lbl in get_dependencies(dtree)
            if lbl == 'Same-Unit']


def same_units_deps_from_ctree(ctree):
    """Get same unit dependencies from a constituency tree"""
    su_pairs = []  # result

    tree_posits = ctree.treepositions()
    for tpos in tree_posits:
        node = ctree[tpos]
        if not isinstance(node, nltk.tree.Tree):  # skip leaf nodes
            continue
        same_units = [(i, x) for i, x in enumerate(node)
                      if (isinstance(x, nltk.tree.Tree) and
                          x.label().rel == 'Same-Unit')]
        lmost_leaves = []
        for i, x in same_units:
            # compare the leftmost leaf of the leftmost nucleus
            # with the recursive leftmost nucleus
            # * leftmost leaf of the leftmost nucleus
            if ((len(x) == 1 and
                 not isinstance(x[0], nltk.tree.Tree))):
                lmost_leaf = x[0]
            else:
                lmost_nuc = [y for y in x
                             if (isinstance(y, nltk.tree.Tree) and
                                 y.label().nuclearity == 'Nucleus')][0]
                lmost_leaf = lmost_nuc.leaves()[0]
            lmost_leaves.append(lmost_leaf.num)
        # generate dependencies according to the "chain" transform
        su_pairs.extend([(gov_idx, dep_idx) for gov_idx, dep_idx in
                         zip(lmost_leaves, lmost_leaves[1:])])
    return su_pairs

# (pseudo-)relations used to impose a tree structure:
# span, Same-Unit, TextualOrganization

# 1. span
# only in ctrees

# 2. TextualOrganization

# 3. Same-Unit
# * pseudo-relation
# * the intervening material is attached to one of the constituents,
# usually the first one, but might be the second one if it is more
# appropriate (e.g. for relation that links two events, when the event
# is in the second fragment)
#
# The problem of spurious Same-Unit is not particular to the RST-DT,
# it also shows in the Discourse Graphbank:
# http://www.aclweb.org/anthology/W10-4311
# Interestingly, this article reveals systematic deviations in the DG
# corpus compared to the definition of the Same relation, by looking
# at cases where the same text is also part of the RST corpus but
# the RST annotation uses another relation than "Same-Unit".
# While this comparison between DG and RST uses the RST treebank as
# a reference, we show that the RST corpus also contains
# inconsistencies.
#
# For another study on Same-Unit in the RST corpus:
# https://www.seas.upenn.edu/~pdtb/papers/BanikLeeLREC08.pdf

# Marcu 2000:
# * p. 167, on another, related corpus (30 MUC7 coref, 30 Brown-Learned,
# 30 WSJ):
# rhetorical relations + "two constituency relations that were ubiquitous
# in the corpora and that often subsumed complex rhetorical constituents,
# and one textual relation. The constituency relations were `attribution`,
# which was used to label the relation between a reporting and a reported
# clause, and `apposition`. The textual relation was `TextualOrganization`;
# it was used to connect in an RST-like manner the textual spans that
# corresponded to the title, author, and textual body of each document in
# the corpus."


def same_units_adjacent(dtree):
    """Same-Units where the fragments are adjacent"""
    res = []
    for dep_idx, (gov_idx, lbl) in enumerate(
            zip(dtree.heads[1:], dtree.labels[1:]),
            start=1):
        if lbl == 'Same-Unit' and dep_idx - gov_idx == 1:
            res.append((gov_idx, dep_idx))
            print('W:', gov_idx, dep_idx)
    return res


def same_units_both_inside_deps(dtree):
    """Same-Units where both fragments govern intervening material"""
    res = []
    for dep_idx, (gov_idx, lbl) in enumerate(
            zip(dtree.heads[1:], dtree.labels[1:]),
            start=1):
        if lbl == 'Same-Unit':
            gov_deps_i = [x for x in dtree.deps(gov_idx)
                          if x > gov_idx and x < dep_idx]
            dep_deps_i = [x for x in dtree.deps(dep_idx)
                          if x > gov_idx and x < dep_idx]
            if gov_deps_i and dep_deps_i:
                res.append((gov_idx, dep_idx))
    return res


def same_units_different_sentences(dtree):
    """Same-Units where the two fragments belong to different sentences

    (Not yet implemented)
    """
    # TODO
    return []


def same_units_second_has_inside_attribution(dtree):
    """Same-Units with intervening 'attribution' headed by frag2."""
    res = []
    for dep_idx, (gov_idx, lbl) in enumerate(
            zip(dtree.heads[1:], dtree.labels[1:]),
            start=1):
        if lbl == 'Same-Unit':
            # frag2 has intervening direct dependents "attribution"
            dep_deps_i = [x for x in dtree.deps(dep_idx)
                          if (x > gov_idx and x < dep_idx and
                              dtree.labels[x].startswith('attribution'))]
            # TODO extend to transitive dependents ?
            if dep_deps_i:
                res.append((gov_idx, dep_idx))
    return res


# note from Feng's PhD thesis:
# their model fails on Topic-Change, Textual-Organization, Topic-Comment,
# Evaluation, because they look "more abstractly defined" => candidate for
# post-proc? impact of WMD on these spans?

def check_same_units_ctree(ctree):
    """Check structural properties of "Same-Unit" fragments.
    """
    tree_posits = ctree.treepositions()
    for tpos in tree_posits:
        node = ctree[tpos]
        if not isinstance(node, nltk.tree.Tree):  # skip leaf nodes
            continue
        same_units = [(i, x) for i, x in enumerate(node)
                      if (isinstance(x, nltk.tree.Tree) and
                          x.label().rel == 'Same-Unit')]
        # weird same-units: n-ary (n>2) same-units
        if len(same_units) > 2:
            print(ctree.origin.doc, '\tn-ary same-unit\t',
                  [x[1].label() for x in same_units])
        # weird same-units: nucleus != span[0]
        for i, x in same_units:
            # compare the leftmost leaf of the leftmost nucleus
            # with the recursive leftmost nucleus
            # * leftmost leaf of the leftmost nucleus
            if ((len(x) == 1 and
                 not isinstance(x[0], nltk.tree.Tree))):
                lmost_leaf = x[0]
            else:
                lmost_nuc = [y for y in x
                             if (isinstance(y, nltk.tree.Tree) and
                                 y.label().nuclearity == 'Nucleus')][0]
                lmost_leaf = lmost_nuc.leaves()[0]

            # * (recursively found) nucleus of the leftmost nucleus
            nuc_cand = x
            while True:
                if ((len(nuc_cand) == 1 and
                     not isinstance(nuc_cand[0], nltk.tree.Tree))):
                    nuc_cand = nuc_cand[0]
                    break
                # else recurse
                nuc_cands = [y for y in nuc_cand
                             if (isinstance(y, nltk.tree.Tree) and
                                 y.label().nuclearity == 'Nucleus')]
                if len(nuc_cands) > 1:
                    print(ctree.origin.doc, '\t>1 nucleus\t', x.label())
                nuc_cand = nuc_cands[0]
            if lmost_leaf != nuc_cand:
                print(ctree.origin.doc, '\tlmost_leaf != nucleus\t', x.label())
    su_groups = []  # TODO
    return su_groups


# attribution
if False:
    for dt in dtrees:
        check_su_right_gov(dt)
    raise ValueError('Hop SU right govs')

# Same-Unit
if False:  # run check functions
    for x in ctrees:
        check_same_units_ctree(x)


if True:  # get same-unit deps from d and c trees
    same_units_nb = 0
    with open('/home/mmorey/melodi/rst_same_unit_suspects_clean1', 'wb') as f:
        for ct, dt in zip(ctrees, dtrees):
            # typical pathological cases:
            # * both gov and dep have deps inside span
            both_inside = set(same_units_both_inside_deps(dt))
            # * adjacent fragments
            len_one = set(same_units_adjacent(dt))
            # * intervening EDU headed by frag2
            intervening_dep2 = set(same_units_second_has_inside_attribution(dt))
            # * different sentences (not implemented yet)
            diff_sents = set(same_units_different_sentences(dt))
            # union of Same-Unit weirdos
            su_weirdos = sorted(both_inside | len_one | intervening_dep2 |
                                diff_sents)
            doc_name = ct.origin.doc
            if su_weirdos:
                print('\n'.join('{}\t{}\t{}'.format(
                    doc_name, x[0], x[1])
                                for x in su_weirdos),
                      file=f)
            # total number of Same-Unit dependencies
            all_deps = get_dependencies(dt)
            all_same_units = set((gov_idx, dep_idx, lbl)
                                 for gov_idx, dep_idx, lbl
                                 in all_deps
                                 if lbl == 'Same-Unit')
            same_units_nb += len(all_same_units)
    print('Total number of Same-Unit dependencies:', same_units_nb)

raise ValueError('Check me')
# end WIP


def merge_same_units(dtree):
    """Merge fragments of EDUs linked by the pseudo-relation Same-Unit.

    Parameters
    ----------
    dtree : RstDepTree
        Dependency tree.

    Returns
    -------
    dtree_merged : RstDepTree?
        Dependency tree with merged EDUs instead of Same-Unit.
    """
    raise NotImplementedError('TODO implement merge_same_units')

# SIGDIAL 2001:
# "In addition, three relations are used to impose structure on the tree:
# textual-organization, span, and same-unit (used to link parts of units
# separated by embedded units or spans)."

# hence, the following relation labels should or can be separated or
# discarded for evaluation:
# * "span" (in consituency trees; obvious),
# * "root" (in dependency trees; obvious),
# * "same-unit" is a pseudo-relation (pretty obvious),
# * "textual-organization" (? check with NA)

# Joty also has a relation label "dummy", I suspect they serve the same
# purpose as the "ROOT" label from dependency trees


# TODO
# 0. ENH function to merge same-unit
# 1. ENH knn classification on EDUs => try to associate a "semantic class" of event to each EDU, then look at the pair of cluster IDs to decide attachment or labels
# 2. FIX STAC features
# 3. ENH MLP
