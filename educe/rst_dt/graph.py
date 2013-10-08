# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
Converter from RST Discourse Treebank trees to educe-style hypergraphs
"""

import copy
import collections
import itertools
import textwrap

from educe import corpus, stac
from educe.graph import *
import educe.graph
from pygraph.readwrite import dot
import pydot
import pygraph.classes.hypergraph as gr
import pygraph.classes.digraph    as dgr
from pygraph.algorithms import traversal
from pygraph.algorithms import accessibility

class Graph(educe.graph.Graph):
    def __init__(self):
        return educe.graph.Graph.__init__(self)

    @classmethod
    def from_doc(cls, corpus, doc_key):
        return super(Graph, cls).from_doc(corpus, doc_key)

class DotGraph(educe.graph.DotGraph):
    """
    A dot representation of this graph for visualisation.
    The `to_string()` method is most likely to be of interest here
    """

    def __init__(self, anno_graph):
        educe.graph.DotGraph.__init__(self, anno_graph)

    def _edu_label(self, anno):
        if callable(getattr(anno, "text_span", None)):
            span = ' ' + str(anno.text_span())
        else:
            span = ''
        text     = self.doc.text(anno.span)
        return "%s %s" % (text, span)

    def _add_edu(self, node):
        anno  = self.core.annotation(node)
        label = self._edu_label(anno)
        attrs = { 'label' : textwrap.fill(label, 30)
                , 'shape' : 'plaintext'
                }
        self.add_node(pydot.Node(node, **attrs))
