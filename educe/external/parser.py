#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
Syntactic parser output into educe standoff annotations
(at least as emitted by Stanford's CoreNLP_ pipeline

This currently builds off the NLTK Tree class, but if the
NLTK dependency proves too heavy, we could consider doing
without.

.. _CoreNLP:       http://nlp.stanford.edu/software/corenlp.shtml
"""

import collections
from   itertools import chain

import nltk.tree

from   educe.annotation import Span, Standoff

class SearchableTree(nltk.Tree):
    """
    A tree with helper search functions
    """
    def __init__(self, node, children):
        nltk.Tree.__init__(self, node, children)

    def topdown(self, pred, prunable=None):
        """
        Searching from the top down, return the biggest subtrees for which the
        predicate is True.  The optional prunable function can be used to
        throw out subtrees for more efficient search (note that pred always
        overrides prunable though).  Note that leaf nodes are ignored.
        """
        if pred(self):
            return [self]
        elif prunable and prunable(self):
            return []
        else:
            return chain.from_iterable(x.topdown(pred) for x in self
                                       if isinstance(x,SearchableTree))

class ConstituencyTree(SearchableTree, Standoff):
    """
    A variant of the NLTK Tree data structure which can be
    treated as an educe Standoff annotation.

    This can be useful for representing syntactic parse trees
    in a way that can be later queried on the basis of Span
    enclosure.

    Note that all children must have a `span` member of type
    `Span`

    The `subtrees()` function can useful here.
    """
    def __init__(self, node, children, origin=None):
        SearchableTree.__init__(self, node, children)
        Standoff.__init__(self, origin)
        if not children:
            raise Exception("Can't create a tree with no children")
        self.children = children
        start = min(x.span.char_start for x in children)
        end   = max(x.span.char_end   for x in children)
        self.span = Span(start, end)

    def _members(self):
        return self.children

    def text_span(self):
        """
        Note: doc is ignored here
        """
        return self.span

    @classmethod
    def build(cls, tree, tokens):
        """
        Build an educe tree by combining an existing NLTK tree with
        some replacement leaves.

        The replacement leaves should correspond 1:1 to the leaves of the
        original tree (for example, they may contain features related to
        those words
        """
        toks = collections.deque(tokens)
        def step(t):
            if not isinstance(t, nltk.tree.Tree):
                if toks:
                    return toks.popleft()
                else:
                    raise Exception('Must have same number of input tokens as leaves in the tree')
            return cls(t.node, list(map(step, t)))
        return step(tree)


class DependencyTree(SearchableTree, Standoff):
    """
    A variant of the NLTK Tree data structure for the representation
    of dependency trees. The dependency tree is also considered a
    Standoff annotation but not quite in the same way that a
    constituency tree might be. The spans roughly indicate the range
    covered by the tokens in the subtree (this glosses over any gaps).
    They are mostly useful for determining if the tree (at its root
    node) pertains to any given sentence based on its offsets.

    Fields:

    * node is an some annotation of type `educe.annotation.Standoff`
    * link is a string representing the link label between this node
      and its governor; None for the root node
    """
    def __init__(self, node, children, link, origin=None):
        SearchableTree.__init__(self, node, children)
        Standoff.__init__(self, origin)
        nodes = children
        if not self.is_root():
            nodes.append(self.node)
        start = min(x.span.char_start for x in nodes)
        end = max(x.span.char_end for x in nodes)
        self.link = link
        self.span = Span(start, end)
        self.origin = origin

    def is_root(self):
        """
        This is a dependency tree root (has a special node)
        """
        return self.node == 'ROOT'

    @classmethod
    def build(cls, deps, nodes, k, link=None):
        """
        Given two dictionaries

        * mapping node ids to a list of (link label, child node id))
        * mapping node ids to some representation of those nodes

        and the id for the root node, build a tree representation
        of the dependency tree
        """
        if k in nodes:
            node = nodes[k]
        else:
            node = 'ROOT'

        if k in deps:
            children = [ cls.build(deps, nodes, k2, link2) for link2, k2 in deps[k] ]
            return cls(node, children, link)
        else:
            return node
