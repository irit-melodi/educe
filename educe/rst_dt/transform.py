#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Eric Kow
# License: CeCILL-B (BSD3-like)

"""
Convert RST trees to other structures, for example, binary RST trees
"""

from educe.rst_dt.parse import RSTTree, Node


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
    if len(tree) == 1:  # pre-terminal
        return tree
    elif len(tree) == 2:
        return RSTTree(tree.node, map(binarize, tree))
    else:
        # convenient string representation of what the children look like
        # eg. NS, SN, NNNNN, SNS
        nscode = "".join(kid.node.type[0] for kid in tree)

        nuclei = filter(lambda x: x.node.type == 'Nucleus', tree)
        satellites = filter(lambda x: x.node.type == 'Satellite', tree)
        if len(nuclei) + len(satellites) != len(tree):
            raise Exception("Nodes that are neither Nuclei nor Satellites\n%s"
                            % tree)

        if len(nuclei) == 0:
            raise Exception("No nucleus:\n%s" % tree)
        elif len(nuclei) > 1:  # multi-nuclear chain
            if satellites:
                raise Exception("Multinuclear with satellites:\n%s" % tree)
            left = tree[0]
            right = _chain_to_binary(left.node.rel, tree[1:])
            return RSTTree(tree.node, [left, right])
        elif nscode == 'SNS':
            left = _chain_to_binary('span', tree[:2])
            right = tree[2]
            return RSTTree(tree.node, [left, right])
        else:
            raise Exception("Don't know how to handle %s trees", nscode)
