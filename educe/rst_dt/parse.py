#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

test0= """( Nucleus (span 5 6) (rel2par preparation:act)
            ( Satellite (leaf 5) (rel2par act:goal) (text <s><EDU> You'll need to measure the wall or room to be paneled,</EDU>) )
            ( Nucleus (leaf 6) (rel2par act:goal) (text <EDU>estimate the amount of paneling you'll need,</EDU>) )
          )
"""


import re, sys
from nltk import Tree


text_re = re.compile("\(text (?P<text>.+(</EDU>|</s>))\)")
#re.findall(text_re,test)
leaf_pattern = "\[[^\]]+\]"
# re.findall(leaf_pattern,s)
type_re = "\((?P<type>(Nucleus|Satellite))"
span_re = "\((?P<span>(leaf [0-9]+|span [0-9]+ [0-9]+))\)"
rel_re = "\((?P<rel>rel2par [A-Za-z0-9:]+)\)"
head_pattern = re.compile("%s %s %s"%(type_re,span_re,rel_re))
#re.findall("%s %s %s"%(type_re,span_re,rel_re),s)

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
    res = str.strip()
    res = mark_leaves(res)
    res = re.sub("\(\s+","(",res)#.replace("\n",""))
    res = re.sub("\s+\)",")",res)
    res = re.sub("\s\s+"," ",res)
    res = mark_heads(res)
    return res

#s= preprocess(test)
#t = Tree.parse(s,leaf_pattern=leaf_pattern)
  

def postprocess(tree):
    if isinstance(tree,Tree):
        children = [postprocess(child) for child in tree] 
        return Tree(Node(tree.node),children)
    else:
        if tree.startswith("["):
            return EDU(tree[1:-1])
        else:
            print >> sys.stderr, "ERROR in rst tree format for leaf : ", child
            sys.exit(0)

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
            #print descr
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
        s = preprocess(str)
        # starts with (Root (span 1 9) 
        root, span, minedu,maxedu, s = s.split(" ",4)
        t = Tree.parse("%s %s"%(root,s),leaf_pattern=leaf_pattern)
        self._t = postprocess(t)
        self._minedu = int(minedu)
        self._maxedu = int(maxedu[:-1])
        
    def __repr__(self):
        return self._t.pprint()

    def tree(self):
        return self._t

    def span(self):
        return self._minedu,self._maxedu

    def latex(self):
        return self._t.pprint_latex_qtree()


a = RSTTree(test)
