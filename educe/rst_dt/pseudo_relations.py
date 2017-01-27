"""Specific functions for pseudo-relations in the RST corpus.

"""
from __future__ import print_function

import nltk.tree


# WIP TextualOrganization:
# 1. oO : "(~)summary" (title)
# 2. O(o)+ : no-rel or "Style-TextualOrganization" (footnotes, byline)
# 3. (o)+O(o)+ : id. (2)
# problem: the two bydates (CHICAGO, SMYRNA) are (top-level) oO

# WIP Topic-Shift (NN)
# 1. top-level: no-rel or "Style-Topic-Shift"
# 2. not top-level: "(~)List"

def rewrite_pseudo_rels(doc_key, ctree):
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
    doc_key: FileId
        Identifier of the document.
    ctree: RSTTree
        RST tree to be fixed.

    Returns
    -------
    ctree: RSTTree
        RST tree with pseudo-relations corrected.
    """
    print('Doc', doc_key.doc)
    for tpos in ctree.treepositions():
        node = ctree[tpos]
        if ((not isinstance(node, nltk.tree.Tree) or
             any(not isinstance(kid, nltk.tree.Tree) for kid in node))):
            # leaf or pre-terminal
            continue
        kid_rels = [kid.label().rel for kid in node]
        if all(rel == 'TextualOrganization' for rel in kid_rels):
            # WIP TextualOrganization:
            # 1. oO : "(~)summary" (title)
            # 2. O(o)+ : no-rel or "Style-TextualOrganization"
            # (footnotes, byline)
            # 3. (o)+O(o)+ : id. (2)
            # problem: the two bydates (CHICAGO, SMYRNA) are (top-level) oO
            kid_spans = [kid.text_span() for kid in node]
            kid_lens = [x.length() for x in kid_spans]
            if ((len(kid_lens) == 2 and
                 kid_lens[0] < kid_lens[1])):
                # case 1: oO: "(~summary)" (title)
                # * datelines: manual exclusion, treat as case 2.
                if ((doc_key.doc == 'wsj_1377.out' and
                     node[0].label().edu_span == (2, 2)) or
                    (doc_key.doc == 'wsj_1105.out' and
                     node[0].label().edu_span == (1, 1))):
                    # TODO do as case 2
                    for kid in node:
                        kid.label().rel = 'Style-TextualOrganization'
                        kid.label().nuclearity = 'Nucleus'
                    continue

                # regular case 1:
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
                if ((doc_key.doc == 'wsj_0687.out' and
                     node[0].label().edu_span == (1, 38)) or
                    (doc_key.doc == 'wsj_1398.out' and
                     node[0].label().edu_span == (1, 4)) or
                    (doc_key.doc == 'wsj_2366.out' and
                     node[0].label().edu_span == (1, 33))):
                    # TextualOrganization that should be top-level
                    # Topic-Shift: manual exclusion,
                    # mark as "Style-Topic-Shift" (see below)
                    for kid in node:
                        kid.label().rel = 'Style-Topic-Shift'
                        kid.label().nuclearity = 'Nucleus'
                    # print('TO', [kid.label() for kid in node])
                    continue
                elif ((doc_key.doc == 'wsj_1322.out' and
                       node[0].label().edu_span == (64, 88)) or
                      (doc_key.doc == 'wsj_1999.out' and
                       node[0].label().edu_span == (3, 21))):
                    # TextualOrganization that should be Topic-Shift
                    # that should be List
                    # TODO ask NA for blessing
                    for kid in node:
                        kid.label().rel = 'List'
                        kid.label().nuclearity = 'Nucleus'
                    # print('TO', [kid.label() for kid in node])
                    continue

                # regular processing for cases 2 and 3: byline, footnotes
                main_txt = kid_lens.index(max(kid_lens))
                for i, kid in enumerate(node):
                    if i == main_txt:
                        kid.label().rel = 'Style-TextualOrganization'
                        kid.label().nuclearity = 'Nucleus'
                        continue
                    # otherwise: 'Style-TextualOrganization'
                    kid.label().rel = 'Style-TextualOrganization'
                    kid.label().nuclearity = 'Nucleus'
            # print('TO', [kid.label() for kid in node])
        elif all(rel == 'Topic-Shift' for rel in kid_rels):
            # WIP Topic-Shift (NN)
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
        elif all(rel == 'Same-Unit' for rel in kid_rels):
            # WIP Same-Unit
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
    return ctree
