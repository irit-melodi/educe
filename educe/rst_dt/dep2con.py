"""Conversion between dependency and constituency trees.

TODO
----
* [ ] refactor strategies
* [ ] use label+nuclearity to help determine the order of attachment:
      greedily take successive modifiers that belong to the same
      underlying multinuclear relation
"""

from collections import namedtuple
import itertools

from .annotation import SimpleRSTTree, Node
from .deptree import RstDtException, NUC_N, NUC_S, NUC_R
from ..internalutil import treenode


class InsideOutAttachmentRanker(object):
    """Rank modifiers, from the inside out on either side.

    Given a dependency tree node and its children, return the list
    of children but *stably* sorted to fulfill an inside-out
    traversal on either side.

    Let's back up a little bit for some background on this criterion.
    We assume that dependency tree nodes can be characterised and
    ordered by their position in the text. If we did such a thing, the
    head node would sit somewhere between its children ::

        lX, .., l2, l1, h, r1, r2, .., rY

    An inside-out traversal is one in which moves strictly outward
    from the centre node. Note that this does not imply any other
    requirement on the order our traversal (particularly on left vs
    right), so `h l1 l2 .. lX r1 r2, etc` is as much a valid
    inside-outside traversal as `h l1 r1 l2 etc`; however,
    `h l2 l1 etc` is right out.

    The ranking is determined by a strategy. Currently implemented
    strategies are:
    - 'id': stable sort that keeps the original interleaving, as
    defined below,
    - 'lllrrr': all left then right dependents,
    - 'rrrlll': all right then left dependents,
    - 'lrlrlr': alternating directions, left first,
    - 'rlrlrl': alternating directions, right first.

    TRICKY SORTING! The current implementation of the 'id' strategy was
    reached through a bit of trial and error, so you may want to modify
    with caution.

    Most of the trickiness in the 'id' strategy is in making this a
    *stable* sort, ie. we want to preserve the original order of
    the targets as much as possible because this allows us to have
    round trip conversions from RST to DT and back. This essentially
    means preserving the interleaving of left/right nodes. The basic
    logic in the implementation is to traverse our target list as
    a series of LEFT or RIGHT slots, filling the slots in an
    inside-out order. So for example, if we saw a target list
    `l3 r1 r3 l2 l1 r2`, we would treat it as the slots `L R R L L R`
    and fill them out as `l1 r1 r2 l2 l3 r3`
    """

    def __init__(self, strategy='id'):
        if strategy not in ['id',
                            'lllrrr', 'rrrlll',
                            'lrlrlr', 'rlrlrl',
                            'closest-lr', 'closest-rl',
                            'closest-intra-lr-inter-lr',
                            'closest-intra-rl-inter-rl',
                            'closest-intra-rl-inter-lr']:
            raise ValueError('Unknown transformation strategy ',
                             '{stg}'.format(stg=strategy))
        self.strategy = strategy

    def fit(self, X, y):
        """Here a no-op."""
        return self

    def predict(self, X):
        """Produce a ranking.

        This keeps the alternation of left and right modifiers as it is
        in `targets` but possibly re-orders dependents on either side to
        guarantee inside-out traversal.

        Parameters
        ----------
        X: list of triples (RstDepTree, EDU, list of EDUs)
            Dependency tree, head EDU and its list of modifiers

        Returns
        -------
        sorted_mods: list of EDUs
            Sorted list of modifiers

        Notes
        -----
        The '*intra*' strategies need to map each EDU to its surrounding
        sentence. This is currently done by attaching to the RstDepTree
        this mapping as a list, `sent_idx`.
        Future versions should make this right.
        """
        strategy = self.strategy

        sorted_mods = []
        for (dtree, head, targets) in X:
            sorted_nodes = sorted([head] + targets,
                                  key=lambda x: dtree.edus[x].span.char_start)
            centre = sorted_nodes.index(head)
            # elements to the left and right of the node respectively
            # these are stacks (outside ... inside)
            left = sorted_nodes[:centre]
            right = list(reversed(sorted_nodes[centre+1:]))

            # special strategy: 'id' (we know the true targets)
            if strategy == 'id':
                result = [left.pop() if (tree in left) else right.pop()
                          for tree in targets]

            # strategies that try to guess the order of attachment
            else:
                if strategy == 'lllrrr':
                    result = [left.pop() if left else right.pop()
                              for _ in targets]

                elif strategy == 'rrrlll':
                    result = [right.pop() if right else left.pop()
                              for _ in targets]

                elif strategy == 'lrlrlr':
                    # reverse lists of left and right modifiers
                    # these are queues (inside ... outside)
                    left_io = list(reversed(left))
                    right_io = list(reversed(right))
                    lrlrlr_gen = itertools.chain.from_iterable(
                        itertools.izip_longest(left_io, right_io))
                    result = [x for x in lrlrlr_gen
                              if x is not None]

                elif strategy == 'rlrlrl':
                    # reverse lists of left and right modifiers
                    # these are queues (inside ... outside)
                    left_io = list(reversed(left))
                    right_io = list(reversed(right))
                    rlrlrl_gen = itertools.chain.from_iterable(
                        itertools.izip_longest(right_io, left_io))
                    result = [x for x in rlrlrl_gen
                              if x is not None]

                elif strategy == 'closest-rl':
                    # take closest dependents first, take right over left to
                    # break ties
                    head_idx = dtree.idx[head]
                    sort_key = lambda e: (abs(dtree.idx[e] - head_idx),
                                          1 if dtree.idx[e] > head_idx else 2)
                    result = sorted(targets, key=sort_key)

                elif strategy == 'closest-lr':
                    # take closest dependents first, take left over right to
                    # break ties
                    head_idx = dtree.idx[head]
                    sort_key = lambda e: (abs(dtree.idx[e] - head_idx),
                                          2 if dtree.idx[e] > head_idx else 1)
                    result = sorted(targets, key=sort_key)

                elif strategy == 'closest-intra-rl-inter-lr':
                    # take closest dependents first, take right over left to
                    # break ties
                    head_idx = dtree.idx[head]
                    sort_key = lambda e: (1 if dtree.sent_idx[dtree.idx[e]] == dtree.sent_idx[dtree.idx[head]] else 2,
                                          abs(dtree.idx[e] - head_idx),
                                          1 if ((dtree.idx[e] > head_idx and
                                                 dtree.sent_idx[dtree.idx[e]] == dtree.sent_idx[dtree.idx[head]]) or
                                                (dtree.idx[e] < head_idx and
                                                 dtree.sent_idx[dtree.idx[e]] != dtree.sent_idx[dtree.idx[head]])) else 2)
                    result = sorted(targets, key=sort_key)

                elif strategy == 'closest-intra-rl-inter-rl':
                    # take closest dependents first, take right over left to
                    # break ties
                    head_idx = dtree.idx[head]
                    sort_key = lambda e: (1 if dtree.sent_idx[dtree.idx[e]] == dtree.sent_idx[dtree.idx[head]] else 2,
                                          abs(dtree.idx[e] - head_idx),
                                          1 if dtree.idx[e] > head_idx else 2)
                    result = sorted(targets, key=sort_key)

                elif strategy == 'closest-intra-lr-inter-lr':
                    # take closest dependents first, take left over right to
                    # break ties
                    head_idx = dtree.idx[head]
                    sort_key = lambda e: (1 if dtree.sent_idx[dtree.idx[e]] == dtree.sent_idx[dtree.idx[head]] else 2,
                                          abs(dtree.idx[e] - head_idx),
                                          2 if dtree.idx[e] > head_idx else 1)
                    result = sorted(targets, key=sort_key)

                else:
                    raise RstDtException('Unknown transformation strategy ',
                                         '{stg}'.format(stg=strategy))

            sorted_mods.append(result)
        return sorted_mods


def deptree_to_simple_rst_tree(dtree, multinuclear, strategy='id'):
    r"""
    Given a dependency tree and a collection of relation labels,
    to be interpreted as being multinuclear, return a 'SimpleRSTTree'.
    Note that nodes are ordered by text span, so that `e1 -R-> e2` and
    `e2 -R-> e1` would both be interpreted as `R(e1, e2)`, the
    potential difference being which is treated as the nucleus and as
    the satellite.

    The conversion algorithm can be seen a two-pass descent/conversion,
    ascent/assembly one. First, we go down the dependency tree,
    annotating each node with a trivial RST tree (ie. a leaf node).
    Then, we walk back up the dependency tree, connecting RST treelets
    together, gradually replacing nodes with increasingly larger RST
    trees.

    The first tricky part of this conversion is to keep in mind
    that the conversion process involves some rotation, namely
    that this dependency structure ::

           R
        a ---> b

    (if we gloss over nuclearity and node order details) is rotated
    into something that looks like this ::


        R +-> a                                 R
          |          (or more conventionally,  / \ )
          +-> b                               a   b


    Where things get tricky is if a node in the dependency tree
    points has multiple children ::

            r1
        src +--> tgt1
            |
            |r2
            +--> tgt2
            |
            ..
            |
            |rN
            +--> tgtN


    In that case, having pivoted the `src(r1:tgt1)` link into something
    like `r1(src, tgt)`, it may not be so clear how to deal with
    subsequents targets `tgt2..tgtN`. We want to both reflect that `src`
    has a dependency link with all of these targets at the same time
    even though it's stuck as an RST child node. The trick is to see the
    process as more of a fold than a map. Keeping in mind that `src`,
    `tgt1..tgtN` are all RST treelets (`tgt1..tgtN` having been built up
    from recursive descent), the idea is to gradually replace the src
    node with an increasingly nested RST tree as we go along ::


        +----------+
        |   r1     |    r2
        |  /  \    + -+---> tgt2
        | src tgt1 |  |
        +----------+  | r3
                      +---> tgt3
                      |
                      ..
                      | rN
                      +---> tgtN



        +------------+
        |     r2     |
        |    /  \    |
        |   r1  tgt2 |    r3
        |  /  \      + -+---> tgt3
        | src tgt1   |  |
        +------------+  ..
                        | rN
                        +---> tgtN


    (and so on, until all we have left is a single RST tree).
    """

    attach_ranker = InsideOutAttachmentRanker(strategy)

    def tgt_nuclearity(rel):
        """
        The target of a dep tree link is normally the satellite
        unless the relation is marked multinuclear
        """
        return NUC_N if rel in multinuclear else NUC_S

    def mk_leaf(edu):
        """
        Trivial partial tree for use when processing dependency
        tree leaves
        """
        return TreeParts(edu=edu,
                         edu_span=(edu.num, edu.num),
                         span=edu.text_span(),
                         rel="leaf",
                         kids=[])

    def parts_to_tree(nuclearity, parts):
        """
        Combine root nuclearity information with a partial tree
        to form a full RST `SimpleTree`
        """
        node = Node(nuclearity,
                    parts.edu_span,
                    parts.span,
                    parts.rel)
        kids = parts.kids or [parts.edu]
        return SimpleRSTTree(node, kids)

    def connect_trees(src, tgt, rel):
        """
        Return a partial tree, assigning order and nuclearity to
        child trees
        """
        tgt_nuc = tgt_nuclearity(rel)

        if src.span.overlaps(tgt.span):
            raise RstDtException("Span %s overlaps with %s " %
                                 (src.span, tgt.span))
        elif src.span <= tgt.span:
            left = parts_to_tree(NUC_N, src)
            right = parts_to_tree(tgt_nuc, tgt)
        else:
            left = parts_to_tree(tgt_nuc, tgt)
            right = parts_to_tree(NUC_N, src)

        l_edu_span = treenode(left).edu_span
        r_edu_span = treenode(right).edu_span

        edu_span = (min(l_edu_span[0], r_edu_span[0]),
                    max(l_edu_span[1], r_edu_span[1]))
        res = TreeParts(edu=src.edu,
                        edu_span=edu_span,
                        span=src.span.merge(tgt.span),
                        rel=rel,
                        kids=[left, right])
        return res

    def walk(ancestor, subtree, strategy):
        """
        The basic descent/ascent driver of our conversion algorithm.
        Note that we are looking at three layers of the dependency
        tree at the same time.


                     r0       r1
            ancestor --> src +--> tgt1
                             |
                             |r2
                             +--> tgt2
                             |
                             ..
                             |
                             |rN
                             +--> tgtN

        The base case is if src is a leaf node (no children),
        whereupon we return a tiny tree connecting the two.

        If we do have children, we have to first obtain the
        full RST tree for src (through the folding process
        described in the docstring for the main function)
        before connecting it to its ancestor.
        """
        rel = dtree.labels[subtree]
        src = mk_leaf(dtree.edus[subtree])
        # descend into each child, but note that we are folding
        # rather than mapping, ie. we threading along a nested
        # RST tree as go from sibling to sibling
        targets = [t for _, t in dtree.deps[subtree]]
        ranked_targets = attach_ranker.predict([(dtree, subtree, targets)])[0]
        for tgt in ranked_targets:
            src = walk(src, tgt, strategy)
        # ancestor is None in the case of the root node
        return connect_trees(ancestor, src, rel) if ancestor else src

    roots = dtree.real_roots_idx()
    if len(roots) == 1:
        real_root = roots[0][1]  # roots is a list of (label, num)
        rparts = walk(None, real_root, strategy)
    else:
        msg = ('Cannot convert RstDepTree to SimpleRSTTree, ',
               'multiple roots: {}'.format(roots))
        raise RstDtException(msg)
    return parts_to_tree(NUC_R, rparts)


# pylint: disable=R0903, W0232
class TreeParts(namedtuple("TreeParts_", "edu edu_span span rel kids")):
    """
    Partially built RST tree when converting from dependency tree
    Kids here is nuclearity-annotated children
    """
    pass
# pylint: enable=R0903, W0232
