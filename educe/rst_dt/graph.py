# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
Converter from RST Discourse Treebank trees to educe-style hypergraphs
"""

import textwrap

from educe.graph import (Graph as EduceGraph,
                         DotGraph as EduceDotGraph)
import pydot


class Graph(EduceGraph):
    def __init__(self):
        EduceGraph.__init__(self)

    @classmethod
    def from_doc(cls, corpus, doc_key):
        return super(Graph, cls).from_doc(corpus, doc_key)


class DotGraph(EduceDotGraph):
    """
    A dot representation of this graph for visualisation.
    The `to_string()` method is most likely to be of interest here
    """

    def __init__(self, anno_graph):
        EduceDotGraph.__init__(self, anno_graph)

    def _edu_label(self, anno):
        if callable(getattr(anno, "text_span", None)):
            span = ' ' + str(anno.text_span())
        else:
            span = ''
        text = self.doc.text(anno.span)
        return "%s %s" % (text, span)

    def _add_edu(self, node):
        anno = self.core.annotation(node)
        label = self._edu_label(anno)
        attrs = {
            'label': textwrap.fill(label, 30),
            'shape': 'plaintext'
        }
        self.add_node(pydot.Node(node, **attrs))
