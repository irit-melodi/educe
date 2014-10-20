#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Eric Kow
# License: CeCILL-B (BSD3-like)

"""
Convert RST trees to dependency trees and back.
"""

from collections import defaultdict, namedtuple
import itertools

from nltk import Tree

from educe.rst_dt.annotation import SimpleRSTTree, Node
from ..internalutil import treenode

_N = "Nucleus"
_S = "Satellite"
_R = "Root"


class RstDtException(Exception):
    """
    Exceptions related to conversion between RST and DT trees.
    The general expectation is that we only raise these on bad
    input, but in practice, you may see them more in cases of
    implementation error somewhere in the conversion process.
    """
    def __init__(self, msg):
        super(RstDtException, self).__init__(msg)


# pylint: disable=R0903, W0232, W0105
class DepNode(namedtuple("DepNode_", "edu num")):
    """
    Dependency graph node
    """
    pass


class RelDepNode(object):
    """
    Dependency graph node annotated with relation label for
    incoming link (None for Root node)
    """

    def __init__(self, edu, num, rel):
        self.edu = edu
        "core content of this node"

        self.num = num
        "which EDU this was (for EDU span)"

        self.rel = rel
        "relation label"

    def __str__(self):
        relstr = (self.rel + ": ") if self.rel else ""
        return relstr + str(self.edu) + (" [%d]" % self.num)

    def __repr__(self):
        return str(self)  # only for debugging
# pylint: enable=R0903, W0232, W0105


def sort_inside_out(head, targets, strategy='id'):
    """
    Given a dependency tree node and its children, return the list
    of children but *stably* sorted to fulfill an inside-out
    traversal on either side.

    Let's back up a little bit for some background on this criterion.
    We assume that dependency tree nodes can be characterised and ordered
    by their position in the text. If we did such a thing, the head node
    would sit somewhere between its children ::

        lX, .., l2, l1, h, r1, r2, .., rY

    An inside-out traversal is one in which moves strictly outward
    from the centre node. Note that this does not imply any other
    requirement on the order our traversal (particularly on left vs
    right), so `h l1 l2 .. lX r1 r2, etc` is as much a valid
    inside-outside traversal as `h l1 r1 l2 etc`; however,
    `h l2 l1 etc` is right out.

    The following strategies are currently implemented:
    - 'lllrrr': all left then right dependents,
    - 'rrrlll': all right then left dependents,
    - 'id': stable sort that keeps the original interleaving,
    as defined below.

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
    def start(node):
        "start position"
        return node.edu.span.char_start

#        def debug(thing):
#            "debug text"
#            if isinstance(thing, RelDepNode):
#                return thing.edu.text
#            elif isinstance(thing, tuple):
#                return (thing[0].edu.text, thing[1])
#            else:
#                return treenode(thing).edu.text

    sorted_nodes = sorted([head] + [treenode(t) for t in targets],
                          key=start)
    centre = sorted_nodes.index(head)
    order = dict(zip(sorted_nodes,
                     itertools.count(start=0-centre)))

    def rel_order(tree):
        "the textual relative order of a node"
        return abs(order[treenode(tree)])

    def is_left(tree):
        "if a node is textually to the left of the head"
        return order[treenode(tree)] < 0

    # elements to the left and right of the node respectively
    # these are stacks
    left = sorted([t for t in targets if is_left(t)],
                  key=rel_order, reverse=True)
    right = sorted([t for t in targets if not is_left(t)],
                   key=rel_order, reverse=True)

    result = []
    # built result according to strategy
    if strategy == 'id':
        for tree in targets:
            if is_left(tree):
                result.append(left.pop())
            else:
                result.append(right.pop())
    elif strategy == 'lllrrr':
        for _ in targets:
            if left:
                result.append(left.pop())
            else:
                result.append(right.pop())
    elif strategy == 'rrrlll':
        for _ in targets:
            if right:
                result.append(right.pop())
            else:
                result.append(left.pop())
    else:
        raise RstDtException('Unknown transformation strategy ',
                             '{stg}'.format(stg=strategy))

    return result


def relaxed_nuclearity_to_deptree(rst_tree):
    """
    Converts a `SimpleRSTTree` to a dependency tree
    """

    dlinks = defaultdict(list)

    def walk(tree):
        """
        Walk down tree, adding dependency graph links to dlinks as we go.
        Return/percolate the head node found in our descent
        """
        if len(tree) == 1:  # pre-terminal
            edu = tree[0]
            edu_num = treenode(tree).edu_span[0]
            return DepNode(edu, edu_num)
        else:
            rel = treenode(tree).rel
            left = tree[0]
            right = tree[1]
            nscode = "".join(treenode(kid).nuclearity[0] for kid in tree)
            lhead = walk(left)
            rhead = walk(right)

            if nscode == "NS":
                head = lhead
                child = rhead
            elif nscode == "SN":
                head = rhead
                child = lhead
            elif nscode == "NN":
                head = lhead
                child = rhead
            else:
                raise RstDtException("Don't know how to handle %s trees" %
                                     nscode)
            dlinks[head].append((rel, child))
            return head

    def mk_tree(relnode):
        """
        Convert a link dictionary to dependency tree
        """
        rel, dnode = relnode
        rnode = RelDepNode(dnode.edu, dnode.num, rel)
        kids = [mk_tree(n) for n in dlinks.get(dnode, [])]
        return Tree(rnode, kids)

    head = walk(rst_tree)  # build dlinks
    return mk_tree((None, head))


# pylint: disable=R0903, W0232
class TreeParts(namedtuple("TreeParts_", "edu edu_span span rel kids")):
    """
    Partially built RST tree when converting from dependency tree
    Kids here is nuclearity-annotated children
    """
    pass
# pylint: enable=R0903, W0232


def relaxed_nuclearity_from_deptree(dtree, multinuclear, strategy='id'):
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

    def tgt_nuclearity(rel):
        """
        The target of a dep tree link is normally the satellite
        unless the relation is marked multinuclear
        """
        return _N if rel in multinuclear else _S

    def mk_leaf(dnode):
        """
        Trivial partial tree for use when processing dependency
        tree leaves
        """
        return TreeParts(edu=dnode.edu,
                         edu_span=(dnode.num, dnode.num),
                         span=dnode.edu.text_span(),
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
            left = parts_to_tree(_N, src)
            right = parts_to_tree(tgt_nuc, tgt)
        else:
            left = parts_to_tree(tgt_nuc, tgt)
            right = parts_to_tree(_N, src)

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
        rel = treenode(subtree).rel
        src = mk_leaf(treenode(subtree))
        # descend into each child, but note that we are folding
        # rather than mapping, ie. we threading along a nested
        # RST tree as go from sibling to sibling
        for tgt in sort_inside_out(treenode(subtree), list(subtree),
                                   strategy):
            src = walk(src, tgt, strategy)
        # ancestor is None in the case of the root node
        return connect_trees(ancestor, src, rel) if ancestor else src

    rparts = walk(None, dtree, strategy)
    return parts_to_tree(_R, rparts)
