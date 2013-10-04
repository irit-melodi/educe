#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author Philippe Muller
#
"""
RST basic API, 
reading the format from Di Eugenio's corpus of instructional texts

encodes RST trees as nltk Tree structure

TODO: 

- translation to predicate argument -> EDU api
- translation to EDU only via nuclearity principle
- import external processing: postag, parsing, etc



"""

test="""
( Root (span 1 9)
  ( Nucleus (leaf 1) (rel2par textualOrganization) (text <s><EDU> ORGANIZING YOUR MATERIALS </EDU></s>) )
  ( Satellite (span 2 9) (rel2par textualOrganization)
    ( Satellite (span 2 4) (rel2par general:specific)
      ( Nucleus (span 2 3) (rel2par preparation:act)
        ( Satellite (leaf 2) (rel2par preparation:act) (text <s><EDU> Once you've decided on the kind of paneling you want to install --- and the pattern ---</EDU>) )
        ( Nucleus (leaf 3) (rel2par preparation:act) (text <EDU>some preliminary steps remain</EDU>) )
      )
      ( Satellite (leaf 4) (rel2par preparation:act) (text <EDU>before you climb into your working clothes. </EDU></s>) )
    )
    ( Nucleus (span 5 9) (rel2par general:specific)
      ( Nucleus (span 5 8) (rel2par preparation:act)
        ( Nucleus (span 5 7) (rel2par step1:step2)
          ( Nucleus (span 5 6) (rel2par preparation:act)
            ( Satellite (leaf 5) (rel2par act:goal) (text <s><EDU> You'll need to measure the wall or room to be paneled,</EDU>) )
            ( Nucleus (leaf 6) (rel2par act:goal) (text <EDU>estimate the amount of paneling you'll need,</EDU>) )
          )
          ( Nucleus (leaf 7) (rel2par preparation:act) (text <EDU>buy the paneling,</EDU>) )
        )
        ( Nucleus (leaf 8) (rel2par step1:step2) (text <EDU>gather the necessary tools and equipment (see illustration on page 87),</EDU>) )
      )
      ( Nucleus (leaf 9) (rel2par preparation:act) (text <EDU>and even condition certain types of paneling before installation. </EDU></s>) )
    )
  )
)
"""

test_text="".join(\
        [" ORGANIZING YOUR MATERIALS ",
         " Once you've decided on the kind of paneling you want to install --- and the pattern ---",
         "some preliminary steps remain",
         "before you climb into your working clothes. ",
         " You'll need to measure the wall or room to be paneled,",
         "estimate the amount of paneling you'll need,",
         "buy the paneling,",
         "gather the necessary tools and equipment (see illustration on page 87),",
         "and even condition certain types of paneling before installation. "
         ])

test0="""
( Root (span 5 6)
  ( Satellite (leaf 5) (rel2par act:goal) (text <EDU>x</EDU>) )
  ( Nucleus   (leaf 6) (rel2par act:goal) (text <EDU>y</EDU>) )
)
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

def process_text(matchobj):
    text = matchobj.group("text")
    return "[%s]"%text#.replace(" ","¤")

def process_head(matchobj):
    ntype = matchobj.group("type")
    span  = matchobj.group("span").replace(" ","-")
    rel   = "---" if ntype == "Root" else matchobj.group("rel").split()[1]
    return "(%s|%s|%s"%(ntype,span,rel)

def mark_leaves(str):
    return text_re.sub(process_text,str)

def mark_heads(str):
    s = root_pattern.sub(process_head,str)
    return head_pattern.sub(process_head,s)

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

        self.text   = s[5:-6] # remove <EDU></EDU> mark
        end         = start + len(self.text)
        self.span   = Span(start, end) # text-span (not the same as EDU span)
        Standoff.__init__(self, origin)

    def __repr__(self):
        return self.text

class Node:
    def __init__(self,descr,span, origin=None):
        self.type, self.edu_span, self.rel = descr.split("|")
        self.edu_span = self.edu_span.split("-")
        if len(self.edu_span)==2:
            self.edu_span.append(self.edu_span[1])
        self.edu_span[1] = int(self.edu_span[1] )
        self.edu_span[2] = int(self.edu_span[2] )
        # Standoff
        self.span = span

    def __repr__(self):
        return "%s %s %s"%(self.type,"%s-%s"%tuple(self.edu_span[1:3]),self.rel)

class RSTTree(SearchableTree, Standoff):
    def __init__(self,node,children,origin=None):
        """
        Note, you should use `RSTTree.build(str)` to create this tree
        instead
        """
        SearchableTree.__init__(self, node, children)
        Standoff.__init__(self, origin)

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
        minedu = self.node.edu_span[1]
        maxedu = self.node.edu_span[2]
        return minedu,maxedu

    def text(self):
        """
        Return the text corresponding to this RST tree
        (traverses and concatenates leaf node text)
        """
        return "".join(l.text for l in self.leaves())

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
        res = mark_leaves(res)
        res = re.sub(r"\(\s+","(",res)#.replace("\n",""))
        res = re.sub(r"\s+\)",")",res)
        res = re.sub(r"\s\s+"," ",res)
        res = mark_heads(res)
        return res

    @classmethod
    def _postprocess(cls, tree, start=0):
        """
        Helper function: Convert the NLTK-parsed representation of an RST tree
        to one using educe-style Standoff objects
        """
        if isinstance(tree,Tree):
            children = []
            position = start
            for c in tree:
                child    = cls._postprocess(c, position)
                children.append(child)
                child_sp = child.node.span if isinstance(child,Tree) else child.span
                position = child_sp.char_end

            span = Span(start, position)
            return RSTTree(Node(tree.node, span),children)
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
