#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Eric Kow
# License: CeCILL-B (BSD3-like)

"""
Convert RST trees to other structures, for example, binary RST trees
"""

import copy

from educe.annotation import Standoff
from educe.external.parser import SearchableTree
from educe.rst_dt.parse import RSTTree, Node, EDU


class RSTTreeException(Exception):
    """
    Exceptions related to RST trees not looking like we would
    expect them to
    """
    def __init__(self, msg):
        super(RSTTreeException, self).__init__(msg)


# pylint: disable=R0904
class SimpleRSTTree(SearchableTree, Standoff):
    """
    Possibly easier representation of RST trees to work with:

    * binary
    * relation labels on parent nodes instead of children

    Note that `RSTTree` and `SimpleRSTTree` share the same
    `Node` type but because of the subtle difference in
    interpretation you should be extremely careful not to
    mix and match.
    """

    def __init__(self, node, children, origin=None):
        """
        Note, you should use `SimpleRSTTree.from_RSTTree(tree)`
        to create this tree instead
        """
        SearchableTree.__init__(self, node, children)
        Standoff.__init__(self, origin)

    def set_origin(self, origin):
        """
        Recursively update the origin for this annotation, ie.
        a little link to the document metadata for this annotation
        """
        self.origin = origin
        for child in self:
            child.set_origin(origin)

    def text_span(self):
        return self.node.span

    def _members(self):
        return list(self)  # children

    @classmethod
    def from_rst_tree(cls, tree):
        """
        Build and return a `SimpleRSTTree` from an `RSTTree`
        """
        return cls._from_binary_rst_tree(binarize(tree))

    @classmethod
    def _from_binary_rst_tree(cls, tree):
        """
        Helper to from_rst_tree; hoist the relation from the
        satellite node to the parent. If there is no satellite
        (ie. we have a multinuclear relation), take it from the
        left node.
        """
        if len(tree) == 1:
            node = copy.copy(tree.node)
            node.rel = "leaf"
            return SimpleRSTTree(node, tree, tree.origin)
        else:
            left = tree[0]
            right = tree[1]
            node = copy.copy(tree.node)
            node.rel = right.node.rel if right.node.is_satellite()\
                else left.node.rel
            kids = [cls._from_binary_rst_tree(kid) for kid in tree]
            return SimpleRSTTree(node, kids, tree.origin)


def _chain_to_binary(rel, kids):
    """
    (binarize helper)

    Fold a list of RST trees into a single binary tree given a relation
    that is expected to hold over each consequenctive pair of subtrees.
    """

    def builder(right, left):
        "function to fold with"
        edu_span = (left.node.edu_span[0], right.node.edu_span[1])
        span = left.node.span.merge(right.node.span)
        newnode = Node('Nucleus', edu_span, span, rel)
        return RSTTree(newnode, [left, right])
    return reduce(builder, kids[::-1])


def is_binary(tree):
    """
    True if the given RST tree or SimpleRSTTree is indeed binary
    """
    if isinstance(tree, EDU):
        return True
    elif len(tree) > 2:
        print tree
        return False
    else:
        return all(map(is_binary, tree))


def binarize(tree):
    """
    Slightly rearrange an RST tree as a binary tree.  The non-trivial
    cases here are

    * `X(sns) => X(N(sn),s)` Given a hypotactic relation with exactly two
      satellites (left and right), lower the left most satellite-nucleus
      pair into a subtree with a nuclear head.  As an example, given
      `X(attribution:S1, N, explanation-argumentative:S2)`, we would
      return something like this:
      `X(span:N(attribution:S1, N), explanation-argumentative:S2)`,


    * `X(nnn...)` => X(n,N(n,N(...))) (multi-nuclear, 0 satellites)
      Straightforwardly build a chain of cons cells glued together
      by new Nuclear nodes.

      For example, given `X(List:N1, List:N2, List:N3)`, we would
      return `X(List:N1, List:N(List:N2, List:N3))`
    """
    if isinstance(tree, EDU):
        return tree
    elif len(tree) <= 2:
        return RSTTree(tree.node, map(binarize, tree))
    else:
        # convenient string representation of what the children look like
        # eg. NS, SN, NNNNN, SNS
        nscode = "".join(kid.node.nuclearity[0] for kid in tree)

        nuclei = [kid for kid in tree if kid.node.is_nucleus()]
        satellites = [kid for kid in tree if kid.node.is_satellite()]
        if len(nuclei) + len(satellites) != len(tree):
            raise Exception("Nodes that are neither Nuclei nor Satellites\n%s"
                            % tree)

        if len(nuclei) == 0:
            raise Exception("No nucleus:\n%s" % tree)
        elif len(nuclei) > 1:  # multi-nuclear chain
            if satellites:
                raise Exception("Multinuclear with satellites:\n%s" % tree)
            kids = map(binarize, tree)
            left = kids[0]
            right = _chain_to_binary(left.node.rel, kids[1:])
            return RSTTree(tree.node, [left, right])
        elif nscode == 'SNS':
            left = _chain_to_binary('span', tree[:2])
            right = binarize(tree[2])
            return RSTTree(tree.node, [left, right])
        else:
            raise RSTTreeException("Don't know how to handle %s trees", nscode)
