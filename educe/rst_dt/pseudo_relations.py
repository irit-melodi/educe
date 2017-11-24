"""Specific functions for pseudo-relations in the RST corpus.

"""
from __future__ import print_function

import nltk.tree

from educe.rst_dt.annotation import EDU, NUC_N


# WIP TextualOrganization:
# 1. oO : "(~)summary" (title)
# 2. O(o)+ : no-rel or "Style-TextualOrganization" (footnotes, byline)
# 3. (o)+O(o)+ : id. (2)
# problem: the two bydates (CHICAGO, SMYRNA) are (top-level) oO

# WIP Topic-Shift (NN)
# 1. top-level: no-rel or "Style-Topic-Shift"
# 2. not top-level: "(~)List"

def _rewrite_textualorganization(doc_name, node):
    """Rewrite TextualOrganization labels in an RST (sub) c-tree.

    The subtree is modified inplace.

    Parameters
    ----------
    doc_name : str
        Document name.
    node: RSTTree
        Parent node whose children are all labelled with the relation
        "TextualOrganization".
    """
    # We distinguish 3 cases:
    # 1. oO : "(~)summary" (title)
    # 2. O(o)+ : no-rel or "Style-TextualOrganization"
    # (footnotes, byline)
    # 3. (o)+O(o)+ : id. (2)
    kid_spans = [kid.text_span() for kid in node]
    kid_lens = [x.length() for x in kid_spans]
    if len(kid_lens) == 2 and kid_lens[0] < kid_lens[1]:
        # case 1: oO: "(~summary)" (title)
        # * special processing for the two bydates (CHICAGO, SMYRNA),
        # that are (top-level) oO ; they should be handled like case (2)
        if ((doc_name == 'wsj_1377.out' and
             node[0].label().edu_span == (2, 2)) or
            (doc_name == 'wsj_1105.out' and
             node[0].label().edu_span == (1, 1))):
            # process as case 2
            for kid in node:
                kid.label().rel = 'Style-TextualOrganization'
                kid.label().nuclearity = 'Nucleus'
        else:
            # * "regular" case 1:
            # (a) rewrite to mononuclear 'summary-n'
            # TODO define "small" for o ?
            # node[0]: title
            node[0].label().rel = 'span'
            node[0].label().nuclearity = 'Nucleus'
            # node[1]: main text
            node[1].label().rel = 'summary-n'
            node[1].label().nuclearity = 'Satellite'
            # (b) ??
    else:
        # cases 2 and 3: footnotes, byline
        # but first, catch manually defined exceptions
        # TODO questions for NA:
        # * wsj_1341: bydate or title?
        # * wsj_1944: 44 title, 45-69 main text, 70 footnote ?
        # or annotation error, should 70 be footnote for whole doc?
        if ((doc_name == 'wsj_0687.out' and
             node[0].label().edu_span == (1, 38)) or
            (doc_name == 'wsj_1398.out' and
             node[0].label().edu_span == (1, 4)) or
            (doc_name == 'wsj_2366.out' and
             node[0].label().edu_span == (1, 33))):
            # TextualOrganization that should be top-level Topic-Shift:
            # rewrite as "Style-Topic-Shift" (see below)
            for kid in node:
                kid.label().rel = 'Style-Topic-Shift'
                kid.label().nuclearity = 'Nucleus'
        elif ((doc_name == 'wsj_1322.out' and
               node[0].label().edu_span == (64, 88)) or
              (doc_name == 'wsj_1999.out' and
               node[0].label().edu_span == (3, 21))):
            # TextualOrganization that should be Topic-Shift
            # that should be List
            # TODO ask NA for blessing
            for kid in node:
                kid.label().rel = 'List'
                kid.label().nuclearity = 'Nucleus'
        else:
            # regular processing for cases 2 and 3: byline, footnotes
            # main_txt = kid_lens.index(max(kid_lens))
            for kid in node:
                kid.label().rel = 'Style-TextualOrganization'
                kid.label().nuclearity = 'Nucleus'
        # print('TO', [kid.label() for kid in node])


def _rewrite_topic_shift(doc_name, node):
    """Rewrite Topic-Shift (NN) labels in an RST (sub) c-tree.

    The subtree is modified inplace.

    Parameters
    ----------
    doc_name : str
        Document name.
    node: RSTTree
        Parent node whose children are all labelled with the relation
        "Topic-Shift".
    """
    # 1. top-level: no-rel or "Style-Topic-Shift"
    # 2. not top-level: "(~)List"
    if node.label().rel == '---':
        # top-level Topic-Shift => Style-Topic-Shift
        for kid in node:
            kid.label().rel = 'Style-Topic-Shift'
            kid.label().nuclearity = 'Nucleus'
    else:
        # not top-level: "(~)List"
        for kid in node:
            kid.label().rel = 'List'
            kid.label().nuclearity = 'Nucleus'
    # print('TS', [kid.label() for kid in node])


def _rewrite_same_unit(doc_name, node):
    """Rewrite Same-Unit labels in an RST (sub) c-tree.

    A Same-Unit is considered fishy if, for any two consecutive
    fragments:
    - they are adjacent or
    - both have dependents in the gap.

    The subtree is modified inplace.

    Parameters
    ----------
    doc_name : str
        Document name.
    node: RSTTree
        Parent node whose children are all labelled with the relation
        "Topic-Shift".
    """
    # 1. SU_1 and SU_2 have inside deps => fishy
    # 2. otherwise, probably a regular Same-Unit

    # find the (recursively defined) nucleus of each kid
    rec_nucs = []
    for kid in node:
        rec_nuc = kid
        while isinstance(rec_nuc, nltk.tree.Tree):
            if all(isinstance(rec_kid, nltk.tree.Tree)
                   for rec_kid in rec_nuc):
                rec_nuc = [rec_kid for rec_kid in rec_nuc
                           if (rec_kid.label().nuclearity ==
                               'Nucleus')][0]
            else:
                # pre-terminal
                assert len(rec_nuc) == 1
                rec_nuc = rec_nuc[0]
        rec_nucs.append(rec_nuc)
    # close examination of this SU: search for fishiness
    is_fishy = False
    for kid_cur, kid_nxt, rnuc_cur, rnuc_nxt in zip(
            node[:-1], node[1:], rec_nucs[:-1], rec_nucs[1:]):
        # an SU is fishy if in any consecutive pair of SU fragments
        # (SU_i, SU_{i+1}):
        # * both fragments have inside dependents
        # * or both fragments are adjacent
        # TODO different sentences
        # TODO ? intervening attribution?
        if (((kid_cur.leaves()[-1] != rnuc_cur and
              kid_nxt.leaves()[0] != rnuc_nxt) or
             (rnuc_nxt.num - rnuc_cur.num == 1))):
            # both have inside dependents or they are adjacent
            is_fishy = True
            break
    if is_fishy:
        for kid in node:
            kid.label().rel = 'Suspicious-Same-Unit'
            kid.label().nuclearity = 'Nucleus'
        # print('Suspicious Same Unit', [kid.label() for kid in node])


def rewrite_pseudo_rels(doc_key, ctree, verbose=False):
    """Rewrite pseudo-relations in an RST tree.

    The RST corpus officially contains two pseudo-relations:
    * Same-Unit, whose erroneous instances we filter to rewrite their
    label with the multinuclear "Suspicious-Same-Unit" ;
    * TextualOrganization, whose instances we rewrite in one of two ways:
    (a) title main_text: the relation is replaced with the mononuclear
    "summary-n", which is a true discourse relation,
    (b) (bydate)* main_text (byline)? (footnote)*: the relation is
    replaced with the multinuclear "Style-TextualOrganization", that
    we clearly mark as a pseudo-relation.

    Following our investigations on the corpus, we also rewrite instances of
    the multinuclear "Topic-Shift" relation:
    (a) top-level instances have their label rewritten as the new
    pseudo-relation "Style-Topic-Shift",
    (b) other instances have their label rewritten as the multinuclear
    "List", a true relation.

    Arguments
    ---------
    doc_key : FileId
        Identifier of the document.
    ctree : RSTTree
        RST tree to be fixed.
    verbose : boolean, defaults to False
        Verbosity.

    Returns
    -------
    ctree : RSTTree
        RST tree with pseudo-relations corrected.
    """
    doc_name = doc_key.doc
    if verbose:
        print('Doc', doc_name)
    for tpos in ctree.treepositions():
        node = ctree[tpos]
        if ((not isinstance(node, nltk.tree.Tree) or
             any(not isinstance(kid, nltk.tree.Tree) for kid in node))):
            # leaf or pre-terminal
            continue
        kid_rels = [kid.label().rel for kid in node]
        if all(rel == 'TextualOrganization' for rel in kid_rels):
            _rewrite_textualorganization(doc_name, node)
        elif all(rel == 'Topic-Shift' for rel in kid_rels):
            _rewrite_topic_shift(doc_name, node)
        elif all(rel == 'Same-Unit' for rel in kid_rels):
            _rewrite_same_unit(doc_name, node)
    return ctree


# WIP 2017-06-20
def _merge_same_unit(doc_name, subtree):
    """Merge a subtree into a unique node.

    In fact, the subtree is replaced with a pre-terminal and a terminal.

    The subtree is modified inplace.

    Parameters
    ----------
    doc_name : str
        Document name.
    subtree : RSTTree
        Subtree to be merged.
    """
    node = subtree.label()
    edus = subtree.leaves()
    # prepare new attributes
    new_num = node.edu_span[0]
    new_span = node.span
    new_txt = subtree.text()
    subtree[:] = [EDU(new_num, new_span, new_txt, context=edus[0].context,
                      origin=edus[0].origin)]


def merge_same_units(doc_key, ctree, verbose=False):
    """Merge fragmented EDUs, including anything in between fragments.

    Parameters
    ----------
    doc_key :  FileId
        Identifier of the document.
    ctree : RSTTree
        RST tree to be fixed.
    verbose : boolean, defaults to False
        Verbosity.

    Returns
    -------
    ctree : RSTTree
        RST tree with fragmented EDUs merged.
    """
    doc_name = doc_key.doc
    if verbose:
        print('Doc', doc_name)
    del_tpos = set()  # tree positions of merge points
    for tpos in ctree.treepositions():
        # if this tree pos is under a merge point, skip
        get_out = False
        for del_tp in del_tpos:
            if tpos[:len(del_tp)] == del_tp:
                # skip subtrees below merged subtrees
                get_out = True
                break
        if get_out:
            continue
        # if kids are Same-Unit, merge them into a unique kid for the
        # (previously) fragmented EDU plus intervening material
        node = ctree[tpos]
        if ((not isinstance(node, nltk.tree.Tree) or
             any(not isinstance(kid, nltk.tree.Tree) for kid in node))):
            # leaf or pre-terminal
            continue
        kid_rels = [kid.label().rel for kid in node]
        if all(rel == 'Same-Unit' for rel in kid_rels):
            _merge_same_unit(doc_name, node)
            # the whole subtree is replaced with a pre-terminal (the root
            # of the subtree) plus one terminal, so treepositions below
            # don't exist anymore and should be skipped
            del_tpos.add(tpos)
    # rewrite EDU spans across the whole tree ; postorder traversal
    # guarantees that the EDU spans of children are already updated
    edu_idx = 0  # keep track of last EDU num
    for tpos in ctree.treepositions(order='postorder'):
        node = ctree[tpos]
        if isinstance(node, nltk.tree.Tree):
            if all(isinstance(kid, nltk.tree.Tree) for kid in node):
                node.label().edu_span = (node[0].label().edu_span[0],
                                         node[-1].label().edu_span[1])
                # set EDU head to the head of the leftmost nucleus child
                kids_nuclei = [i for i, kid in enumerate(node)
                               if kid.label().nuclearity == NUC_N]
                lnuc = node[kids_nuclei[0]]
                node.label().head = lnuc.label().head
            else:
                # pre-terminal: (edu_num, edu_num)
                assert len(node) == 1
                node.label().edu_span = (node[0].num, node[0].num)
                node.label().head = node[0].num
        else:
            # terminal: EDU
            node.num = edu_idx + 1
            edu_idx += 1
    # end rewrite EDU spans
    return ctree
