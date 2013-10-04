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

test_text=\
        [" ORGANIZING YOUR MATERIALS ",
         " Once you've decided on the kind of paneling you want to install --- and the pattern ---",
         "some preliminary steps remain",
         "before you climb into your working clothes. ",
         " You'll need to measure the wall or room to be paneled,",
         "estimate the amount of paneling you'll need,",
         "buy the paneling,",
         "gather the necessary tools and equipment (see illustration on page 87),",
         "and even condition certain types of paneling before installation. "
         ]

test0="""
( Root (span 5 6)
  ( Satellite (leaf 5) (rel2par act:goal) (text <EDU>x</EDU>) )
  ( Nucleus   (leaf 6) (rel2par act:goal) (text <EDU>y</EDU>) )
)
"""


import re, sys
import codecs
from nltk import Tree

# pre-processing leaves
text_re = re.compile(r"\(text (?P<text>.+(</EDU>|</s>|_!))\)")

# pre-processing heads
type_re = r"\((?P<type>(Nucleus|Satellite))"
span_re = r"\((?P<span>(leaf [0-9]+|span [0-9]+ [0-9]+))\)"
rel_re  = r"\((?P<rel>rel2par [\-A-Za-z0-9:]+)\)"
head_pattern = re.compile("%s %s %s"%(type_re,span_re,rel_re))

# parsing
leaf_pattern = r"\[[^\]]+\]" # non-']' chars in square brackets

def process_text(matchobj):
    text = matchobj.group("text")
    return "[%s]"%text#.replace(" ","¤")

def process_head(matchobj):
    ntype = matchobj.group("type")
    span = matchobj.group("span").replace(" ","-")
    rel = matchobj.group("rel").split()[1]
    return "(%s|%s|%s"%(ntype,span,rel)

def mark_leaves(str):
    return text_re.sub(process_text,str)

def mark_heads(str):
   return head_pattern.sub(process_head,str)

def preprocess(str):
    """
    Given a raw RST treebank string, return a massaged representation for easier
    parsing, along with its first/last EDU number.

    The string is assumed to take the form "(Root (span X Y) Nodes.. )", where X
    and Y are integers and Nodes are RST tree string nodes
    """
    s_     = preprocess_helper(str)
    root, span, minedu_, maxedu_, s = s_.split(" ",4)
    tstr   = "%s %s" % (root, s)
    minedu = int(minedu_)
    maxedu = int(maxedu_[:-1]) # trailing right bracket
    return tstr, minedu, maxedu

def preprocess_helper(str):
    """
    Convert a raw RST treebank tree string to something which is a bit easier
    for us to parse
    """
    res = str.strip()
    res = mark_leaves(res)
    res = re.sub(r"\(\s+","(",res)#.replace("\n",""))
    res = re.sub(r"\s+\)",")",res)
    res = re.sub(r"\s\s+"," ",res)
    res = mark_heads(res)
    return res

def postprocess(tree):
    if isinstance(tree,Tree):
        children = [postprocess(child) for child in tree] 
        return Tree(Node(tree.node),children)
    else:
        if tree.startswith("["):
            return EDU(tree[1:-1])
        else:
            raise Exception( "ERROR in rst tree format for leaf : ", child)

class EDU:
    def __init__(self,descr):
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
        # remove <EDU></EDU> mark
        self._text = s[5:-6]

    def __repr__(self):
        return self._text

class Node:
    def __init__(self,descr):
        if descr =="span" or descr=="Root":
            self.type = "root"
            self.span = "---"
            self.rel = None
        else:
            self.type, self.span, self.rel = descr.split("|")
            self.span = self.span.split("-")
            if len(self.span)==2:
                self.span.append(self.span[1])
            self.span[1] = int(self.span[1] )
            self.span[2] = int(self.span[2] )


    def __repr__(self):
        return "%s %s %s"%(self.type,"%s-%s"%tuple(self.span[1:3]),self.rel)

class RSTTree:

    def __init__(self,str):
        tstr, minedu, maxedu = preprocess(str)
        t       = Tree.parse(tstr, leaf_pattern=leaf_pattern)
        self._t = postprocess(t)
        self._minedu = minedu
        self._maxedu = maxedu

    def __repr__(self):
        return self._t.pprint()

    def tree(self):
        return self._t

    def span(self):
        return self._minedu,self._maxedu

    def latex(self):
        return self._t.pprint_latex_qtree()

def read_annotation_file(anno_filename):
    """
    Read a single RST tree
    """
    t = None
    with codecs.open(anno_filename, 'r', 'utf-8') as tf:
        t = RSTTree(tf.read())
    return t
