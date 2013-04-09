# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
Graph representation of discourse structure.

Currently a bit tied to `educe.stac`, but it may be possible in
time to tease the STAC-specific bits from it.

The core structure currently assumes a Glozz representation
(ie. CDUs as schemas)

The STAC bits are really just for visualisation in graphviz
"""

import copy
import textwrap

from educe import corpus, stac
from pygraph.readwrite import dot
import pydot
import pygraph.classes.hypergraph as gr

class DuplicateIdException(Exception):
    def __init__(self, duplicate):
        self.duplicate = duplicate
        Exception.__init__(self, "Duplicate node id: %s" % duplicate)


class Graph(gr.hypergraph):
    def __init__(self, corpus, doc_key, doc):
        self.corpus  = corpus
        self.doc_key = doc_key
        self.doc     = doc
        self.turns   = [ u for u in doc.units if u.type == 'Turn' ]

        gr.hypergraph.__init__(self)

        # objects that are pointed to by a relations or schemas
        pointed_to = []
        for x in doc.relations: pointed_to.extend([x.span.t1, x.span.t2])
        for x in doc.schemas:   pointed_to.extend(x.span)

        nodes = []
        edges = []

        edus  = [ x for x in doc.units   if x.local_id() in pointed_to ]
        rels  = doc.relations
        cdus  = [ s for s in doc.schemas if s.type != 'default' ]

        for x in edus: nodes.append(self._unit_node(x))
        for x in rels: nodes.append(self._rel_node(x))
        for x in cdus: nodes.append(self._schema_node(x))
        for x in rels: edges.append(self._rel_edge(x))
        for x in cdus: edges.append(self._schema_edge(x))

        for node, attrs in nodes:
            if not self.has_node(node):
                self.add_node(node)
                for x in attrs.items():
                    self.add_node_attribute(node,x)
            else:
                raise DuplicateIdException(node)

        for edge, attrs, links in edges:
            if not self.has_edge(edge):
                self.add_edge(edge)
                self.add_edge_attributes(edge, attrs.items())
                for l in links: self.link(l,edge)

    def _mk_guid(self, x):
        return self.doc_key.mk_global_id(x)

    def _mk_node(self, anno, type):
        anno_id     = anno.identifier()
        attrs = { 'type'       : type
                , 'annotation' : anno
                }
        return (anno_id, attrs)

    def _mk_edge(self, anno, type, members):
        anno_id = anno.identifier()
        attrs   = { 'type'       : type
                  , 'annotation' : anno
                  }
        links   = [ self._mk_guid(m) for m in members ]
        return (anno_id,attrs,links)

    def _unit_node(self, anno):
        return self._mk_node(anno, 'EDU')

    def _rel_node(self, anno):
        # by rights, there are no such things as nodes corresponding to
        # relations, but we do have relations pointing to relations
        # and python-graph reasonably enough gets confused if we try to
        # create edges to nodes that don't exist
        return self._mk_node(anno, 'rel')

    def _schema_node(self, anno):
        # see _rel_node comments
        return self._mk_node(anno, 'CDU')

    def _rel_edge(self, anno):
        members = [ anno.span.t1, anno.span.t2 ]
        return self._mk_edge(anno, 'rel', members)

    def _schema_edge(self, anno):
        return self._mk_edge(anno, 'CDU', anno.span)

    def _get_speaker(self, u):
        """
        Helper function for to_dot
        """
        enclosing_turns = [ t for t in self.turns if t.span.encloses(u.span) ]
        if len(enclosing_turns) > 0:
            return enclosing_turns[0].features['Emitter']
        else:
            return None

    def _get_speech_acts(self, anno):
        """
        Helper function for to_dot
        """
        # In discourse annotated part of the corpus, all segments have
        # type 'Other', which isn't too helpful. Try to recover the
        # speech act from the unit equivalent to this document
        anno_local_id  = anno.local_id()
        fallback       = stac.dialogue_act(anno)
        unit_key       = copy.copy(self.doc_key)
        unit_key.stage = 'units'
        if unit_key in self.corpus:
            udoc  = self.corpus[unit_key]
            doppelgangers = [ u for u in udoc.units if u.local_id() == anno_local_id ]
            if len(doppelgangers) > 0:
                return stac.dialogue_act(doppelgangers[0])
            else:
                return fallback
        else:
            return fallback

    def _edu_label(self, anno):
        """
        Helper function for to_dot
        """
        speech_acts = ", ".join(self._get_speech_acts(anno))
        speaker     = self._get_speaker(anno)
        if speaker is None:
            speaker_prefix = ''
        else:
            speaker_prefix = '(%s) ' % speaker
        return speaker_prefix + "%s [%s]" % (self.doc.text_for(anno), speech_acts)

    def to_dot(self):
        """
        Return a pydot representation of this graph for visualisation
        """
        dot_g = pydot.Dot(compound='true')
        dot_g.set_name('hypergraph')

        def gv_id(raw_id):
            attr_list = self.node_attributes(raw_id)
            if len(attr_list) == 0:
                attr_list = self.edge_attributes(raw_id)
            attrs = dict(attr_list)
            if attrs['type'] == 'CDU':
                return 'cluster_' + raw_id
            else:
                return raw_id

        # Add all of the nodes first
        for node in self.nodes():
            attrs   = dict(self.node_attributes(node))
            node_ty = attrs['type']
            if node_ty == 'EDU':
                anno  = attrs['annotation']
                label = self._edu_label(anno)
                attr_list = { 'label' : textwrap.fill(label, 30)
                            , 'shape' : 'plaintext'
                            }
                if not stac.is_dialogue_act(anno):
                    attr_list['fontcolor'] = 'red'
                dot_node = pydot.Node(str(node), **attr_list)
                dot_g.add_node(dot_node)

        for hyperedge in self.hyperedges():
            attrs   = dict(self.edge_attributes(hyperedge))
            edge_ty = attrs['type']
            if edge_ty == 'rel':
                # it's a bit more useful to represent relations as 'midpoint'
                # nodes rather than edges directly
                #
                # this allows for relations to point to relations for one thing
                # and for another, generally makes for nicer-looking output
                anno           = attrs['annotation']
                anno_id = hyperedge
                links   = self.links(hyperedge)
                link1_, link2_ = links
                midpoint_attrs =\
                    { 'label'      : anno.type
                    , 'shape'      : 'plaintext'
                    , 'fontcolor'  : 'blue'
                    }

                attrs1  = { 'arrowhead' : 'none'
                          }
                attrs2  = {
                          }
                clink1 = gv_id(link1_)
                clink2 = gv_id(link2_)
                if clink1 != link1_:
                    attrs1['ltail'] = clink1
                    link1 = self.links(link1_)[0]
                else:
                    link1 = link1_

                if clink2 != link2_:
                    attrs2['lhead'] = clink2
                    link2 = self.links(link2_)[0]
                else:
                    link2 = link2_

                midpoint = pydot.Node(hyperedge, **midpoint_attrs)
                edge1    = pydot.Edge(link1, hyperedge, **attrs1)
                edge2    = pydot.Edge(hyperedge, link2, **attrs2)
                dot_g.add_node(midpoint)
                dot_g.add_edge(edge1)
                dot_g.add_edge(edge2)

            elif edge_ty == 'CDU':
                attrs    = { 'color' : 'lightgrey'
                           }
                dot_subg = pydot.Subgraph(gv_id(hyperedge), **attrs)
                #point_attrs = { 'label' : '∙'
                #              , 'shape' : 'plaintext'
                #              }
                #point = pydot.Node(hyperedge, **point_attrs)
                #dot_subg.add_node(point)
                local_nodes = self.links(hyperedge)
                for node in local_nodes:
                    dot_subg.add_node(pydot.Node(node))
                    def is_enclosed(l):
                        return l != hyperedge and\
                               all( [x in local_nodes for x in self.links(l)] )

                    rlinks = [ l for l in self.links(node) if is_enclosed(l) ]
                    for rlink in rlinks: # relations
                        dot_rnode = pydot.Node(rlink)
                        dot_subg.add_node(dot_rnode)

                dot_g.add_subgraph(dot_subg)

        return dot_g
