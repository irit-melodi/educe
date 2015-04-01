"""This submodule provides several functions that find heads in trees.

It uses head rules as described in (Collins 1999), Appendix A.
See `http://www.cs.columbia.edu/~mcollins/papers/heads`,
Bikel's 2004 CL paper on the intricacies of Collins' parser
and the classes in (StanfordNLP) CoreNLP that inherit from
`AbstractCollinsHeadFinder.java` .
"""

from __future__ import print_function


import os

from nltk import Tree


HEAD_RULES_FILE = os.path.join(os.path.dirname(__file__),
                               'collins_head_rules')


def load_head_rules(f):
    """Load the head rules from file f.

    Return a dictionary from parent non-terminal to (direction,
    priority list).
    """
    rules = dict()
    with open(f) as f:
        for line in f:
            prnt_nt, drctn, prrty_lst = line.strip().split('\t')
            lbls = prrty_lst.split() if prrty_lst != '_' else []
            rules[prnt_nt] = (drctn, lbls)
    return rules


HEAD_RULES = load_head_rules(HEAD_RULES_FILE)


# helper functions
def _find_head_generic(direction, priority_list, cnt_hws):
    """Determine the head word of a phrase"""
    if direction == 'Left':  # scan children left to right
        cands = list(enumerate(cnt_hws))
    elif direction == 'Right':
        cands = list(reversed(list(enumerate(cnt_hws))))
    else:
        err_msg = 'Direction can obly be Left or Right, got {}'
        raise ValueError(err_msg.format(direction))

    # try to find each label in the priority list, in turn
    # if none were found, return the head word of the left-
    # (resp. right-) most child
    for lbl in priority_list:
        for c_idx, (cnt, hw) in cands:
            if cnt == lbl:
                return (c_idx, hw)
    else:
        c_idx, (cnt, hw) = cands[0]
        return (c_idx, hw)


def _find_head_np(cnt_hws):
    """Find head word in NP following specific rules"""
    cands = list(enumerate(cnt_hws))
    # return last word if tagged 'POS'
    c_idx, (cnt, hw) = cands[-1]
    if cnt == 'POS':
        return (c_idx, hw)
    # else: RL search for NN, NNP, NNPS, NNS, NX, POS or JJR
    lset = set(['NN', 'NNP', 'NNPS', 'NNS', 'NX', 'POS', 'JJR'])
    for c_idx, (cnt, hw) in reversed(cands):
        if cnt in lset:
            return (c_idx, hw)
    # else: LR search for NP
    for c_idx, (cnt, hw) in cands:
        if cnt == 'NP':
            return (c_idx, hw)
    # else: RL search for $, ADJP or PRN
    lset = set(['$', 'ADJP', 'PRN'])
    for c_idx, (cnt, hw) in reversed(cands):
        if cnt in lset:
            return (c_idx, hw)
    # else: RL search for CD
    for c_idx, (cnt, hw) in reversed(cands):
        if cnt == 'CD':
            return (c_idx, hw)
    # else RL search for JJ, JJS, RB, QP
    lset = set(['JJ', 'JJS', 'RB', 'QP'])
    for c_idx, (cnt, hw) in reversed(cands):
        if cnt in lset:
            return (c_idx, hw)
    # else return last word
    c_idx, (cnt, hw) = cands[-1]
    return (c_idx, hw)


def find_lexical_heads(tree):
    """Find the lexical head at each node of a PTB tree."""

    # result
    head_word = {}  # dict(treepos, word)

    # recursive helper
    def _find_lexical_head_rec(treepos):
        """Get the label and head word of subtree"""
        subtree = tree[treepos]

        if isinstance(subtree, Tree):
            p_nt = subtree.label()  # parent non-terminal
            # non-terminal and head word for each child
            c_tposs = [tuple(list(treepos) + [c_idx])
                       for c_idx, c in enumerate(subtree)]
            cnt_hws = [_find_lexical_head_rec(c_tpos) for c_tpos in c_tposs]

            # no head rule for unary productions
            if len(subtree) == 1:
                c_idx = 0
                cnt, hw = cnt_hws[0]
            else:
                # use the head rule to get the head word from the children
                if p_nt == 'NP':
                    c_idx, hw = _find_head_np(cnt_hws)
                else:
                    try:
                        drctn, prrty_lst = HEAD_RULES[p_nt]
                    except KeyError:
                        print(subtree)
                        raise
                    c_idx, hw = _find_head_generic(drctn, prrty_lst, cnt_hws)

                # apply special post-rule for coordinated phrases (if needed)
                # if h > 2 and Y_h-1 == 'CC': head = Y_h-2
                if (c_idx > 1 and cnt_hws[c_idx - 1] == 'CC'):
                    c_idx = c_idx - 2
                    hw = cnt_hws[c_idx - 2]

        else:  # must be a Token
            p_nt = subtree.tag
            hw = treepos

        # store at treepos and return (convenience for rec calls)
        head_word[treepos] = hw
        return (p_nt, hw)

    treepos_root = ()
    root_nt, hw = _find_lexical_head_rec(treepos_root)

    # return dict(treepos, word)
    return head_word
