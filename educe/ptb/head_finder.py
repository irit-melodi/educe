"""This submodule provides several functions that find heads in trees.

It uses head rules as described in (Collins 1999), Appendix A.
See `http://www.cs.columbia.edu/~mcollins/papers/heads`,
Bikel's 2004 CL paper on the intricacies of Collins' parser
and the classes in (StanfordNLP) CoreNLP that inherit from
`AbstractCollinsHeadFinder.java` .
"""

from __future__ import print_function

from collections import deque
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
    """Find the lexical head at each node of a constituency tree.

    The logic corresponds to Collins' head finding rules.

    This is typically used to find the lexical head of each node of a
    (clean) `educe.external.parser.ConstituencyTree` whose leaves are
    `educe.external.postag.Token`.

    Parameters
    ----------
    tree: `nltk.Tree` with `educe.external.postag.RawToken` leaves
        PTB tree whose lexical heads we want

    Returns
    -------
    head_word: dict(tuple(int), tuple(int))
        Map each node of the constituency tree to its lexical head. Both
        nodes are designated by their (NLTK) tree position (a.k.a. Gorn
        address).
    """
    head_word = {}  # result mapping

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
                if c_idx > 1 and cnt_hws[c_idx - 1] == 'CC':
                    c_idx = c_idx - 2
                    hw = cnt_hws[c_idx]

        else:  # must be an educe Token
            p_nt = subtree.tag
            hw = treepos

        # store at treepos and return (convenience for rec calls)
        head_word[treepos] = hw
        return (p_nt, hw)

    treepos_root = ()
    root_nt, hw = _find_lexical_head_rec(treepos_root)

    return head_word


def find_edu_head(tree, hwords, wanted):
    """Find the head word of a set of wanted nodes from a tree.

    The tree is traversed top-down, breadth first, until we reach a node
    headed by a word from `wanted`.

    Return a pair of treepositions (head node, head word), or None if
    no occurrence of any word in `wanted` was found.

    This function is typically called for each EDU, `wanted` being the
    set of tree positions of its tokens, after `find_lexical_heads` has
    been called on the entire `tree` (providing `hwords`).

    Parameters
    ----------
    tree: `nltk.Tree` with `educe.external.postag.RawToken` leaves
        PTB tree whose lexical heads we want.

    hwords: dict(tuple(int), tuple(int))
        Map from each node of the constituency tree to its lexical head.
        Both nodes are designated by their (NLTK) tree position (a.k.a.
        Gorn address).

    wanted: iterable of tuple(int)
        The tree positions of the tokens in the span of interest, e.g.
        in the EDU we are looking at.

    Returns
    -------
    cur_treepos: tuple(int)
        Tree position of the head node, i.e. the highest node headed by
        a word from wanted.

    cur_hw: tuple(int)
        Tree position of the head word.
    """
    # exclude punctuation from this search
    nohead_tags = set(['.', ',', "''", "``"])
    wanted = set([tp for tp in wanted
                  if tree[tp].tag not in nohead_tags])

    all_treepos = deque([()])  # init with root treepos: ()
    while all_treepos:
        cur_treepos = all_treepos.popleft()
        cur_tree = tree[cur_treepos]
        cur_hw = hwords[cur_treepos]
        if cur_hw in wanted:
            return (cur_treepos, cur_hw)
        elif isinstance(cur_tree, Tree):
            c_treeposs = [tuple(list(cur_treepos) + [c_idx])
                          for c_idx, c in enumerate(tree[cur_treepos])]
            all_treepos.extend(c_treeposs)
        else:  # don't try to recurse if the current subtree is a Token
            pass
    return None
