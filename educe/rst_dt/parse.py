#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Philippe Muller, Eric Kow
#
"""
From RST discourse treebank trees to Educe-style objects
(reading the format from Di Eugenio's corpus of instructional texts).

The main classes of interest are `RSTTree` and `EDU`.  `RSTTree` can be treated
as an NLTK Tree structure.  It is also an educe `Standoff` object, which means
that it points to other RST trees (their children) or to `EDU`.
"""

import re, sys
import codecs
from nltk import Tree

from educe.external.parser import SearchableTree
from educe.annotation      import Standoff, Span

# pre-processing leaves
text_re = re.compile(r"\(text (?P<text>.+(</EDU>|</s>|_!))\)")

# pre-processing heads
type_re      = r"\((?P<type>(Nucleus|Satellite))"
root_type_re = r"\((?P<type>(Root))"
span_re = r"\((?P<span>(leaf [0-9]+|span [0-9]+ [0-9]+))\)"
rel_re  = r"\((?P<rel>rel2par [\-A-Za-z0-9:]+)\)"
head_pattern = re.compile("%s %s %s" % (type_re,      span_re, rel_re))
root_pattern = re.compile("%s %s"    % (root_type_re, span_re))

# parsing
leaf_pattern = r"\[[^\]]+\]" # non-']' chars in square brackets

def _process_text(matchobj):
    text = matchobj.group("text")
    return "[%s]"%text#.replace(" ","¤")

def _process_head(matchobj):
    """
    Helper function, ignore
    """
    ntype = matchobj.group("type")
    span  = matchobj.group("span").replace(" ","-")
    rel   = "---" if ntype == "Root" else matchobj.group("rel").split()[1]
    return "(%s|%s|%s"%(ntype,span,rel)

def _mark_leaves(str):
    """
    Helper function, ignore
    """
    return text_re.sub(_process_text,str)

def _mark_heads(str):
    """
    Helper function, ignore
    """
    s = root_pattern.sub(_process_head,str)
    return head_pattern.sub(_process_head,s)

class EDU(Standoff):
    def __init__(self, descr, start=0, origin=None):
        s = descr.strip()
        if s.startswith("<s>"):
            self._sentstart = True
            s = s[3:]
        else:
            self._sentstart = False
        if s.endswith("</s>"):
            self._sentend = True
            s = s[:-4]
        else:
            self._sentend = False

        if s.startswith("<EDU>") and s.endswith("</EDU>"):
            self.text = s[5:-6] # remove <EDU></EDU> mark
        elif s.startswith("_!") and s.endswith("_!"):
            self.text = s[2:-2]
        else:
            self.text = s
        end         = start + len(self.text)
        self.span   = Span(start, end) # text-span (not the same as EDU span)
        Standoff.__init__(self, origin)

    def set_origin(self, origin):
        self.origin = origin

    def __repr__(self):
        return self.text


class Node(object):
    """
    Fields of interest:

        * nuclearity: Nucleus, Satellite, Root
        * edu_span: pair of integers denoting edu span by count
        * span
        * rel
    """

    def __init__(self, nuclearity, edu_span, span, rel):
        self.nuclearity = nuclearity
        self.edu_span = edu_span
        self.span = span
        self.rel = rel

    @classmethod
    def from_treebank(cls, descr, span):
        """
        Build a node from a piece of an RST DT tree
        """
        nuclearity, edu_span, rel = descr.split("|")
        _edu_span = edu_span.split("-")[1:]
        if len(_edu_span) == 1:
            _edu_span.append(_edu_span[0])
        edu_span = (int(_edu_span[0]),
                    int(_edu_span[1]))
        return cls(nuclearity, edu_span, span, rel)

    def __repr__(self):
        return "%s %s %s" % (self.nuclearity, "%s-%s" % self.edu_span, self.rel)

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


class RSTTree(SearchableTree, Standoff):
    def __init__(self,node,children,origin=None):
        """
        Note, you should use `RSTTree.build(str)` to create this tree
        instead
        """
        SearchableTree.__init__(self, node, children)
        Standoff.__init__(self, origin)

    def set_origin(self, origin):
        self.origin = origin
        for child in self:
            child.set_origin(origin)

    def text_span(self):
        return self.node.span

    def _members(self):
        return list(self) # children

    def __repr__(self):
        return self.pprint()

    def edu_span(self):
        """
        Return the span of the tree in terms of EDU count
        See `self.span` refers more to the character offsets
        """
        return self.node.edu_span

    def text(self):
        """
        Return the text corresponding to this RST tree
        (traverses and concatenates leaf node text)

        Note that this (along with the standoff offsets)
        behave as though there were a single space
        between each EDU
        """
        return " ".join(l.text for l in self.leaves())

    @classmethod
    def build(cls, str):
        tstr = cls._preprocess(str)
        t_   = Tree.parse(tstr, leaf_pattern=leaf_pattern)
        return cls._postprocess(t_)

    @classmethod
    def _preprocess(cls, str):
        """
        Helper function: Given a raw RST treebank string, return a massaged
        representation for easier parsing, along with its first/last EDU number.
        """
        res = str.strip()
        res = _mark_leaves(res)
        res = re.sub(r"\(\s+","(",res)#.replace("\n",""))
        res = re.sub(r"\s+\)",")",res)
        res = re.sub(r"\s\s+"," ",res)
        res = _mark_heads(res)
        return res

    @classmethod
    def _postprocess(cls, tree, start=0):
        """
        Helper function: Convert the NLTK-parsed representation of an RST tree
        to one using educe-style Standoff objects
        """
        if isinstance(tree,Tree):
            children = []
            position = start - 1 # compensate for virtual whitespace added below
            for c in tree:
                child    = cls._postprocess(c, position + 1) # +1 to add virtual whitespace
                                                             # between each EDU
                children.append(child)
                child_sp = child.node.span if isinstance(child,Tree) else child.span
                position = child_sp.char_end

            span = Span(start, position)
            return cls(Node.from_treebank(tree.node, span), children)
        else:
            if tree.startswith("["):
                return EDU(tree[1:-1], start)
            else:
                raise Exception( "ERROR in rst tree format for leaf : ", child)


def read_annotation_file(anno_filename):
    """
    Read a single RST tree
    """
    t = None
    with codecs.open(anno_filename, 'r', 'utf-8') as tf:
        t = RSTTree.build(tf.read())
    return t
