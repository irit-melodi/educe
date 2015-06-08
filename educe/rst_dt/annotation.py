#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Philippe Muller, Eric Kow
# License: CeCILL-B (BSD-3 like)

# disable "pointless string" warning because we want attribute docstrings
# pylint: disable=W0105

"""
Educe-style representation for RST discourse treebank trees
"""

import copy
import functools

from educe.annotation import Standoff, Span
from educe.external.parser import SearchableTree
from ..internalutil import treenode


class RSTTreeException(Exception):
    """
    Exceptions related to RST trees not looking like we would
    expect them to
    """
    def __init__(self, msg):
        super(RSTTreeException, self).__init__(msg)


# pylint: disable=R0903
# I would just used a namedtuple here except that I also want
# to associate the fields with docstrings
class RSTContext(object):
    """
    Additional annotations or contextual information that could
    accompany a RST tree proper. The idea is to have each subtree
    pointing back to the same context object for easy retrieval.
    """
    def __init__(self, text, sentences, paragraphs):
        self._text = text
        "original text on which standoff annotations are based"

        self.sentences = sentences
        "sentence annotations pointing back to the text"

        self.paragraphs = paragraphs
        "Paragraph annotations pointing back to the text"

    def text(self, span=None):
        """
        Return the text associated with these annotations (or None),
        optionally limited to a span
        """
        if self._text is None:
            return None
        elif span is None:
            return self._text
        else:
            return self._text[span.char_start:span.char_end]
# pylint: enable=R0903


# pylint: disable=R0913
class EDU(Standoff):
    """
    An RST leaf node
    """
    _SUMMARY_LEN = 20

    def __init__(self, num, span, text,
                 context=None,
                 origin=None):
        super(EDU, self).__init__(origin)

        self.num = num
        "EDU number (as used in tree node `edu_span`)"

        self.span = span
        "text span"

        self.raw_text = text
        """
        text that was in the EDU annotation itself

        This is not the same as the text that was in the annotated
        document, on which all standoff annotations and spans
        are based.
        """

        self.context = context
        """
        See the `RSTContext` object
        """

    def set_origin(self, origin):
        """
        Update the origin of this annotation and any contained within
        """
        self.origin = origin

    def set_context(self, context):
        """
        Update the context of this annotation.
        """
        self.context = context

    def identifier(self):
        """
        A global identifier (assuming the origin can be used to
        uniquely identify an RST tree)
        """
        # idiosyncratic
        if self.is_left_padding():
            return 'ROOT'
        # end idiosyncratic
        if self.origin:
            return self.origin.mk_global_id(str(self.num))
        else:
            return str(self.num)

    def __repr__(self):
        txt = self.text()
        if len(txt) > self._SUMMARY_LEN + 3:
            txt = txt[:self._SUMMARY_LEN] + "..."
        return "EDU:[%s]" % txt

    def text(self):
        """
        Return the text associated with this EDU. We try to return
        the underlying annotated text if we have the necessary
        context; if we not, we just fall back to the raw EDU text
        """
        if self.context:
            return self.context.text(self.span)
        else:
            return self.raw_text

    # left padding EDUs
    _lpad_num = 0
    _lpad_span = Span(0, 0)
    _lpad_txt = ''

    @classmethod
    def left_padding(cls, context=None, origin=None):
        """Return a left padding EDU"""
        return cls(cls._lpad_num, cls._lpad_span, cls._lpad_txt,
                   context, origin)

    def is_left_padding(self):
        """Returns True for left padding EDUs"""
        return (self.num == self._lpad_num and
                self.span == self._lpad_span)
# pylint: enable=R0913


class Node(object):
    """
    A node in an `RSTTree` or `SimpleRSTTree`.
    """

    def __init__(self, nuclearity, edu_span, span, rel,
                 context=None):
        self.nuclearity = nuclearity
        "one of Nucleus, Satellite, Root"

        self.edu_span = edu_span
        "pair of integers denoting edu span by count"

        self.span = span
        "span"

        self.rel = rel
        """
        relation label (see `SimpleRSTTree` for a note on the different
        interpretation of `rel` with this and `RSTTree`)
        """

        self.context = context
        "See the `RSTContext` object"

    def __repr__(self):
        return "%s %s %s" % (self.nuclearity,
                             "%s-%s" % self.edu_span,
                             self.rel)

    def __eq__(self, other):
        return\
            self.nuclearity == other.nuclearity and\
            self.edu_span == other.edu_span and\
            self.span == other.span and\
            self.rel == other.rel

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_nucleus(self):
        """
        A node can either be a nucleus, a satellite, or a root node.
        It may be easier to work with SimpleRSTTree, in which nodes
        can only either be nucleus/satellite or much more rarely,
        root.
        """
        return self.nuclearity == 'Nucleus'

    def is_satellite(self):
        """
        A node can either be a nucleus, a satellite, or a root node.
        """
        return self.nuclearity == 'Satellite'


# pylint: disable=R0904, E1103
class RSTTree(SearchableTree, Standoff):
    """
    Representation of RST trees which sticks fairly closely to the
    raw RST discourse treebank one.
    """

    def __init__(self, node, children,
                 origin=None):
        """
        See `educe.rst_dt.parse` to build trees from strings
        """
        SearchableTree.__init__(self, node, children)
        Standoff.__init__(self, origin)

    def set_origin(self, origin):
        """
        Update the origin of this annotation and any contained within
        """
        self.origin = origin
        for child in self:
            child.set_origin(origin)

    def text_span(self):
        return treenode(self).span

    def _members(self):
        return list(self)  # children

    def __repr__(self):
        return self.pprint()

    def edu_span(self):
        """
        Return the span of the tree in terms of EDU count
        See `self.span` refers more to the character offsets
        """
        return treenode(self).edu_span

    def text(self):
        """
        Return the text corresponding to this RST subtree.
        If the context is set, we return the appropriate
        segment from the subset of the text.
        If not we just concatenate the raw text of all
        EDU leaves.
        """
        node = treenode(self)
        if node.context:
            return node.context.text(node.span)
        else:
            return " ".join(l.raw_text for l in self.leaves())


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
        return treenode(self).span

    def _members(self):
        return list(self)  # children

    @classmethod
    def from_rst_tree(cls, tree):
        """
        Build and return a `SimpleRSTTree` from an `RSTTree`
        """
        return cls._from_binary_rst_tree(_binarize(tree))

    @classmethod
    def _from_binary_rst_tree(cls, tree):
        """
        Helper to from_rst_tree; hoist the relation from the
        satellite node to the parent. If there is no satellite
        (ie. we have a multinuclear relation), take it from the
        left node.
        """
        if len(tree) == 1:
            node = copy.copy(treenode(tree))
            node.rel = "leaf"
            return SimpleRSTTree(node, tree, tree.origin)
        else:
            left = tree[0]
            right = tree[1]
            node = copy.copy(treenode(tree))
            lnode = treenode(left)
            rnode = treenode(right)
            node.rel = rnode.rel if rnode.is_satellite() else lnode.rel
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
        lnode = treenode(left)
        rnode = treenode(right)
        edu_span = (lnode.edu_span[0], rnode.edu_span[1])
        span = lnode.span.merge(rnode.span)
        newnode = Node('Nucleus', edu_span, span, rel)
        return RSTTree(newnode, [left, right])
    return functools.reduce(builder, kids[::-1])


def is_binary(tree):
    """
    True if the given RST tree or SimpleRSTTree is indeed binary
    """
    if isinstance(tree, EDU):
        return True
    elif len(tree) > 2:
        return False
    else:
        return all(map(is_binary, tree))


def _binarize(tree):
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
    elif len(tree) == 1 and not isinstance(tree[0], EDU):
        raise RSTTreeException("Ill-formed RST tree? Unary non-terminal: " +
                               str(tree))
    elif len(tree) <= 2:
        return RSTTree(treenode(tree), list(map(_binarize, tree)))
    else:
        # convenient string representation of what the children look like
        # eg. NS, SN, NNNNN, SNS
        nscode = "".join(treenode(kid).nuclearity[0] for kid in tree)

        nuclei = [kid for kid in tree if treenode(kid).is_nucleus()]
        satellites = [kid for kid in tree if treenode(kid).is_satellite()]
        if len(nuclei) + len(satellites) != len(tree):
            raise Exception("Nodes that are neither Nuclei nor Satellites\n%s"
                            % tree)

        if len(nuclei) == 0:
            raise Exception("No nucleus:\n%s" % tree)
        elif len(nuclei) > 1:  # multi-nuclear chain
            if satellites:
                raise Exception("Multinuclear with satellites:\n%s" % tree)
            kids = [_binarize(x) for x in tree]
            left = kids[0]
            right = _chain_to_binary(treenode(left).rel, kids[1:])
            return RSTTree(treenode(tree), [left, right])
        elif nscode == 'SNS':
            left = _chain_to_binary('span', tree[:2])
            right = _binarize(tree[2])
            return RSTTree(treenode(tree), [left, right])
        else:
            raise RSTTreeException("Don't know how to handle %s trees", nscode)
