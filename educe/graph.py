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
import collections
import textwrap

from educe import corpus, stac
from pygraph.readwrite import dot
import pydot
import pygraph.classes.hypergraph as gr
import pygraph.classes.digraph    as dgr
from pygraph.algorithms import traversal
from pygraph.algorithms import accessibility

class DuplicateIdException(Exception):
    def __init__(self, duplicate):
        self.duplicate = duplicate
        Exception.__init__(self, "Duplicate node id: %s" % duplicate)

class AttrsMixin():
    """
    Attributes common to both the hypergraph and directed graph
    representation of discourse structure
    """
    def __init__(self):
        pass

    def node_attributes_dict(self, x):
        return dict(self.node_attributes(x))

    def edge_attributes_dict(self, x):
        return dict(self.edge_attributes(x))

    def _attrs(self, x):
        """
        (abstract) should be implemented
        """
        pass

    def type(self, x):
        """
        Return if a node/edge is of type 'EDU', 'rel', or 'CDU'
        """
        return self._attrs(x)['type']

    def is_cdu(self, x):
        return self.type(x) == 'CDU'

    def is_edu(self, x):
        return self.type(x) == 'EDU'

    def is_relation(self, x):
        return self.type(x) == 'rel'

    def annotation(self, x):
        """
        Return the annotation object corresponding to a node or edge
        """
        return self._attrs(x)['annotation']

    def mirror(self, x):
        """
        For objects (particularly, relations/CDUs) that have a mirror image,
        ie. an edge representation if it's a node or vice-versa, return the
        identifier for that image
        """
        return self._attrs(x)['mirror']

    def node(self, x):
        """
        Return the argument if it is a node id, or its mirror if it's an
        edge id

        (This is possible because every edge in the graph has a node that
        corresponds to it)
        """
        if self.has_node(x):
            return x
        else:
            return self.mirror(x)

class Graph(gr.hypergraph, AttrsMixin):
    """
    Hypergraph representation of discourse structure

        * a node for every elementary discourse unit
        * relations as two-node hyperedges
        * CDUs as both nodes and multi-node hyperedges

    Every node/hyperedge is represented as string unique
    within the graph. Given one of these identifiers `x`
    and a graph `g`:

        * `g.type(x)` returns one of the strings
          "EDU", "CDU", "rel"
        * `g.annotation(x)` returns an
          educe.annotation object
        * for relations and CDUs, if `x` is the edge representation
          of the relation/cdu, return the node representation, and
          vice-versa

    Pitfalls and TODOS:

        * Relations, in addition to being edges are also represented as nodes;
          this is because we can sometimes have relations pointing to other
          relations

        * There is some awkwardness in the fact that we systematically have
          both nodes and hyperedges for relations/cdus.  These nodes and
          edges are different objects with different names. To go from node
          to edge and vice versa, use the `mirror` function. By default,
          API functions will return the edge representation of something.

        * TODO: Currently we use educe.annotation objects to represent the
          EDUs, CDUs and relations, but this is likely a bit too low-level to
          be helpful. It may be nice to have higher-level EDU and CDU
          objects instead

    You most likely want to use `Graph.from_doc` instead of
    instantiating an instance directly
    """

    def __init__(self):
        AttrsMixin.__init__(self)
        gr.hypergraph.__init__(self)

    @classmethod
    def from_doc(cls, corpus, doc_key):
        """
        Return a graph representation of a document

        Parameters:

        * corpus  : educe corpus dictionary
        * doc_key : FileId key pointing to the document
        * nodeset : limit the graph to things with ids in this
                    set (no limits by default)

        """
        self         = cls()
        doc          = corpus[doc_key]
        self.corpus  = corpus
        self.doc_key = doc_key
        self.doc     = doc

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

        return self

    def copy(self, nodeset=None):
        """
        Return a copy of the graph, optionally restricted to a subset
        of EDUs and CDUs.

        Note that if you include a CDU, then anything contained by that
        CDU will also be included.

        You don't specify (or otherwise have control over) what
        relations are copied.  The graph will include all
        hyperedges whose links are all
        (a) members of the subset or
        (b) (recursively) hyperedges included because of (a) and (b)

        Note that any non-EDUs you include in the copy set will be
        silently ignored.

        This is a shallow copy in the sense that the underlying
        layer of annotations and documents remains the same.
        """
        g=Graph()
        g.corpus  = self.corpus
        g.doc_key = self.doc_key
        g.doc     = self.doc

        if nodeset is None:
            nodes_wanted = set(self.nodes())
        else:
            nodes_wanted = set(nodeset)

        cdus = [ x for x in nodes_wanted if self.is_cdu(x) ]
        for x in cdus:
            nodes_wanted.update(self.cdu_members(x, deep=True))

        def is_wanted_edge(e):
            return all([l in nodes_wanted for l in self.links(e)])

        # keep expanding the copyable edge list until we've
        # covered everything that exclusively points
        # (indirectly or otherwise) to our copy set
        keep_growing    = True
        edges_remaining = self.hyperedges()
        edges_wanted    = set()
        while keep_growing:
            keep_growing = False
            for e in edges_remaining:
                if is_wanted_edge(e):
                    edges_wanted.add(e)
                    nodes_wanted.add(self.mirror(e)) # obligatory node mirror
                    edges_remaining.remove(e)
                    keep_growing = True

        for n in self.nodes():
            if n in nodes_wanted:
                g.add_node(n)
                for kv in self.node_attributes(n):
                    g.add_node_attribute(n,kv)

        for e in self.hyperedges():
            if e in edges_wanted:
                g.add_hyperedge(e)
                for kv in self.edge_attributes(e):
                    g.add_edge_attribute(e,kv)
                for l in self.links(e):
                    g.link(l,e)

        return g

    def connected_components(self):
        """
        Return a set of a connected components.

        Each connected component set can be passed to `self.subgraph()`
        to be selected as subgraph.

        This builds on python-graph's version of a function with the
        same name but also adds awareness of our conventions about there
        being both a node/edge for relations/CDUs.
        """
        ccs       = accessibility.connected_components(self)
        subgraphs = collections.defaultdict(list)
        for x,c in ccs.items():
            subgraphs[c].append(x)

        # the basic idea here: if any member of connected component is
        # one of our hybrid node/edge creatures, we need to help the
        # graph library recognise that that anything connected *via*
        # the edge should also be considered as connected *to* the
        # edge, so we merge the components
        eaten  = set ()
        merged = {}
        prior  = subgraphs.keys()
        rrr = 0
        while sorted(merged.keys()) != prior:
            rrr += 1
            prior     = sorted(merged.keys())
            merged    = {}
            for k in subgraphs:
                if k in eaten: continue
                cc = subgraphs[k]
                merged[k] = copy.copy(cc)
                for n in cc:
                    if self.has_edge(n):
                        links = set(self.links(n))
                        for k2 in subgraphs.keys():
                            links2 = set(subgraphs[k2])
                            if k2 != k and not links2.isdisjoint(links):
                                eaten.add(k2)
                                merged[k].extend(links2)
            subgraphs = merged

        ccs = frozenset([frozenset(v) for v in subgraphs.values()])
        return ccs

    def _attrs(self, x):
        if self.has_edge(x):
            return self.edge_attributes_dict(x)
        elif self.has_node(x):
            return self.node_attributes_dict(x)
        else:
            raise Exception('Tried to get attributes of non-existing object ' + x)

    def relations(self):
        """
        Set of relation edges representing the relations in the graph.
        By convention, the first link is considered the source and the
        the second is considered the target.
        """
        xs = [ e for e in self.hyperedges() if self.is_relation(e) ]
        return frozenset(xs)

    def edus(self):
        """
        Set of nodes representing elementary discourse units
        """
        xs = [ e for e in self.nodes() if self.is_edu(e) ]
        return frozenset(xs)

    def cdus(self):
        """
        Set of hyperedges representing complex discourse units.

        See also `cdu_members`
        """
        xs = [ e for e in self.hyperedges() if self.is_cdu(e) ]
        return frozenset(xs)

    def cdu_members(self, cdu, deep=False):
        """
        Return the set of EDUs, CDUs, and relations which can be considered as
        members of this CDU.

        This is shallow by default, in that we only return the immediate
        members of the CDU.  If `deep==True`, also return members of CDUs
        that are members of (members of ..) this CDU.
        """

        if deep:
            members = set()
            for m in self.cdu_members(cdu):
                members.add(m)
                if self.is_cdu(m):
                    members.update(self.cdu_members(m,deep))
            return frozenset(members)
        else:
            if self.has_node(cdu):
                hyperedge = self.mirror(cdu)
            else:
                hyperedge = cdu
            return frozenset(self.links(hyperedge))

    def _mk_guid(self, x):
        return self.doc_key.mk_global_id(x)

    def _mk_edge_id(self, x):
        return 'e_' + self._mk_guid(x)

    def _mk_node_id(self, x):
        return 'n_' + self._mk_guid(x)

    def _mk_node(self, anno, type, mirrored=False):
        # a node is mirrored if there is a also an edge
        # corresponding to the same object
        local_id = anno.local_id()
        node_id  = self._mk_node_id(local_id)
        edge_id  = self._mk_edge_id(local_id)
        attrs = { 'type'       : type
                , 'annotation' : anno
                }
        if mirrored:
            attrs['mirror'] = self._mk_edge_id(local_id)
        return (node_id, attrs)

    def _mk_edge(self, anno, type, members, mirrored=False):
        local_id = anno.local_id()
        node_id  = self._mk_node_id(local_id)
        edge_id  = self._mk_edge_id(local_id)
        attrs   = { 'type'       : type
                  , 'annotation' : anno
                  }
        if mirrored:
            attrs['mirror'] = self._mk_node_id(local_id)
        links   = [ self._mk_node_id(m) for m in members ]
        return (edge_id,attrs,links)

    def _unit_node(self, anno):
        return self._mk_node(anno, 'EDU')

    def _rel_node(self, anno):
        # by rights, there are no such things as nodes corresponding to
        # relations, but we do have relations pointing to relations
        # and python-graph reasonably enough gets confused if we try to
        # create edges to nodes that don't exist
        return self._mk_node(anno, 'rel', mirrored=True)

    def _schema_node(self, anno):
        # see _rel_node comments
        return self._mk_node(anno, 'CDU', mirrored=True)

    def _rel_edge(self, anno):
        members = [ anno.span.t1, anno.span.t2 ]
        return self._mk_edge(anno, 'rel', members, mirrored=True)

    def _schema_edge(self, anno):
        return self._mk_edge(anno, 'CDU', anno.span, mirrored=True)


# ---------------------------------------------------------------------
# flattening
# ---------------------------------------------------------------------

class FlatGraph(dgr.digraph, AttrsMixin):
    """
    A flattened structure is a straightforward directed graph
    representation of this discourse structure.

    * CDUs are represented as nodes that point to their contents

    * All relations are at least represented by an edge.  If the
      relation happens to have things pointing at it, it is also
      represented as a node.  Note that this node is just a
      placeholder; attributes are tied to the edge and not the node.
    """
    def __init__(self, gr):
        AttrsMixin.__init__(self)
        dgr.digraph.__init__(self)

        # a node for every EDU ...
        for n in gr.edus():
            self.add_node(n, gr.node_attributes(n))
        # ... and relation that is pointed to by another
        for n in gr.nodes():
            for n2 in gr.neighbors(n):
                if gr.is_relation(n2):
                    self.add_node(n2, gr.node_attributes(n2))
        # ... and CDU
        for e in gr.cdus():
            self.add_node(e, gr.edge_attributes(e))

        # an edge for every relation
        for e in gr.relations():
            links = gr.links(e)
            if len(links) != 2:
                msg = "Bug: relation %s has %d links (should be exactly 2)"\
                    % (e, len(links))
                raise Exception(msg)
            self.add_edge(tuple(links), gr.edge_attributes(e))

        # an edge for every CDU member
        for e in gr.cdus():
            for m in gr.links(e):
                self.add_edge((e,m))

    def _attrs(self, x):
        if self.has_node(x):
            return self.node_attributes_dict(x)
        else:
            raise Exception('Tried to get attributes of non-existing node %s' % x)


    def relations(self):
        """
        Set of relation tuples representing relations in the graph.
        By convention, the first item is considered the source
        and the the second is considered the target.
        """

        # this is a bit annoying; one difference with the hypergraph
        # representation is that we no longer use labels for edges,
        # because they are represented as tuples of of the nodes they
        # connect
        #
        # TODO: one tricky bit here is that we continue to represent
        # the relations that we have relations on as nodes, which is
        # fairly awkward
        def is_rel(e):
            return self.edge_attributes_dict(e)['type'] == 'rel'

        xs = [ e for e in self.edges() if is_rel(e) ]
        return frozenset(xs)

    def edus(self):
        """
        Set of nodes representing elementary discourse units
        """
        xs = [ e for e in self.nodes() if self.is_edu(e) ]
        return frozenset(xs)

    def cdus(self):
        """
        Set of nodes representing complex discourse units.

        See also `cdu_members`
        """
        xs = [ e for e in self.nodes() if self.is_cdu(e) ]
        return frozenset(xs)

    def cdu_members(self, cdu):
        """
        Return the set of EDUs and CDUs which can be considered as
        members of this CDU.

        TODO: For now, this is just straightforwardly the set of nodes that
        were explicitly included, but if there is a way to infer membership
        by some notion of transitivity.  I guess it depends on two things,

        1. whether you want to be able to point outside of the CDU
        2. whether you want to point from outside the CDU to individual
           members of the CDU

        If one of the above is true, I think all bets are off
        """
        return frozenset(self.neighbors(cdu))


# ---------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------

class DotGraph(pydot.Dot):
    """
    A dot representation of this graph for visualisation.
    The `to_string()` method is most likely to be of interest here
    """

    def _get_speaker(self, u):
        enclosing_turns = [ t for t in self.turns if t.span.encloses(u.span) ]
        if len(enclosing_turns) > 0:
            return enclosing_turns[0].features['Emitter']
        else:
            return None

    def _get_speech_acts(self, anno):
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
        speech_acts = ", ".join(self._get_speech_acts(anno))
        speaker     = self._get_speaker(anno)
        if speaker is None:
            speaker_prefix = ''
        else:
            speaker_prefix = '(%s) ' % speaker
        return speaker_prefix + "%s [%s]" % (self.doc.text_for(anno), speech_acts)

    def _has_rel_link(self, rel):
        """
        True if the relation points or is pointed to be another relation
        """
        neighbors = self.core.links(rel)
        return any([self.core.is_relation(n) for n in neighbors])

    def _dot_id(self, raw_id):
        """
        Basic story here is that in in graphviz, cluster names have
        to start with `cluster`, so if we have a CDU, prefix it
        accordingly.
        """
        node_id = self.core.node(raw_id)

        if self.core.is_cdu(node_id):
            edge_id = self.core.mirror(node_id)
            is_simple_cdu = edge_id not in self.complex_cdus
        else:
            is_simple_cdu = False

        if is_simple_cdu:
            return 'cluster_' + node_id
        else:
            return node_id

    def __point(self, logical_target, key):
        """
        Tricky graphviz'ery (helper for `_point_to` and `_point_from`)

        Point from/to a node. If it's a cluster, graphviz is a bit of
        a pain because we can't point directly to it.  Instead we have
        to point to an element within the cluster, and set an
        lhead/ltail attribute on the edge pointing to the cluster.

        So this gives us:

            * logical target  - what we are trying to point to
            * dot target      - the dot_id of the target (edge target)
            * proxy target    - what we have to point to (edge attribute)

        Return a tuple of (target, edge_attrs), the idea being
        that you set your graphviz edge to the target and update
        its attributes accordingly.  Notice we only handle one end
        of the connection.  If you link a potential CDU to another
        potential CDU, you'll need to call this for both ends.

        Crazy!
        """

        logical_target_node = self.core.node(logical_target)
        dot_target = self._dot_id(logical_target_node)

        if dot_target == logical_target_node:
            res = (logical_target_node, {})
        else:
            logical_target_edge = self.core.mirror(logical_target_node)
            proxies = self.core.links(logical_target_edge)
            proxy_target = proxies[0]
            if self.core.has_edge(proxy_target):
                proxy_target = self.core.mirror(proxy_target)
            res = (proxy_target, {key:dot_target})

        return res

    def _point_from(self, logical_target):
        """
        See `__point`
        """
        return self.__point(logical_target, 'ltail')

    def _point_to(self, logical_target):
        """
        See `__point`
        """
        return self.__point(logical_target, 'lhead')

    def _add_edu(self, node):
        anno  = self.core.annotation(node)
        label = self._edu_label(anno)
        attrs = { 'label' : textwrap.fill(label, 30)
                , 'shape' : 'plaintext'
                }
        if not stac.is_dialogue_act(anno):
            attrs['fontcolor'] = 'red'
        self.add_node(pydot.Node(node, **attrs))

    def _add_simple_rel(self, hyperedge):
        anno  = self.core.annotation(hyperedge)
        links = self.core.links(hyperedge)
        link1_, link2_ = links
        attrs =\
            { 'label'      : ' ' + anno.type
            , 'shape'      : 'plaintext'
            , 'fontcolor'  : 'blue'
            }

        link1, attrs1 = self._point_from(link1_)
        link2, attrs2 = self._point_to(link2_)

        attrs.update(attrs1)
        attrs.update(attrs2)
        self.add_edge(pydot.Edge(link1, link2, **attrs))

    def _add_complex_rel(self, hyperedge):
        anno  = self.core.annotation(hyperedge)
        links = self.core.links(hyperedge)
        link1_, link2_ = links
        midpoint_attrs =\
            { 'label'      : anno.type
            , 'style'      : 'dotted'
            , 'fontcolor'  : 'blue'
            }

        attrs1  = { 'arrowhead' : 'tee'
                  , 'arrowsize' : '0.5'
                  }
        attrs2  = {
                  }
        link1, attrs1_ = self._point_from(link1_)
        link2, attrs2_ = self._point_to(link2_)
        attrs1.update(attrs1_)
        attrs2.update(attrs2_)

        midpoint_id = self.core.node(hyperedge)
        midpoint = pydot.Node(midpoint_id, **midpoint_attrs)
        edge1    = pydot.Edge(link1, midpoint_id, **attrs1)
        edge2    = pydot.Edge(midpoint_id, link2, **attrs2)
        self.add_node(midpoint)
        self.add_edge(edge1)
        self.add_edge(edge2)

    def _add_simple_cdu(self, hyperedge):
        """
        Straightforward CDU that can be supported as a cluster.
        """
        attrs    = { 'color' : 'lightgrey'
                   }
        if len(self.complex_cdus) > 0:
            # complex CDUs have a CDU node, so I thought it might be
            # less confusing in those cases to also label the simple
            # CDUs so the user knows it's the same thing
            attrs['label'] = 'CDU'
        subg = pydot.Subgraph(self._dot_id(hyperedge), **attrs)
        local_nodes = self.core.links(hyperedge)
        for node in local_nodes:
            subg.add_node(pydot.Node(node))
            def is_enclosed(l):
                return l != hyperedge and\
                       l in self.complex_rels and\
                       all( [x in local_nodes for x in self.core.links(l)] )

            rlinks = [ l for l in self.core.links(node) if is_enclosed(l) ]
            for rlink in rlinks: # relations
                subg.add_node(pydot.Node(rlink))

        self.add_subgraph(subg)

    def _add_complex_cdu(self, hyperedge):
        """
        Yes, a complex "complex discourse unit".

        The idea is to to have a node representing a CDU and dotted lines
        pointing to its members.  It's actually simpler in implementation
        terms but more complex visually

        This is an artefact of graphviz 2.28's inability to
        work with nested subgraphs.
        """
        attrs    = { 'color' : 'grey'
                   , 'label' : 'CDU'
                   , 'shape' : 'rectangle'
                   }
        cdu_id   = self._dot_id(hyperedge)
        self.add_node(pydot.Node(cdu_id,  **attrs))
        for node in self.core.links(hyperedge):
            edge_attrs = { 'style' : 'dashed'
                         , 'color' : 'grey'
                         }
            dest, attrs_ = self._point_to(node)
            edge_attrs.update(attrs_)
            self.add_edge(pydot.Edge(cdu_id, dest, **edge_attrs))

    def __init__(self, anno_graph):
        """
        Params:

        * anno_graph - the abstract annotation graph
        """
        self.core       = anno_graph
        self.doc        = self.core.doc
        self.doc_key    = self.core.doc_key
        self.corpus     = self.core.corpus
        self.turns      = [ u for u in self.core.doc.units if u.type == 'Turn' ]
        pydot.Dot.__init__(self, compound='true')
        self.set_name('hypergraph')

        # rels which are the target of links
        self.complex_rels = set()
        for n in self.core.nodes():
            for n2 in self.core.neighbors(n):
                if self.core.is_relation(n2):
                    e2 = self.core.mirror(n2)
                    self.complex_rels.add(e2)

        # CDUs which are contained in other CDUs or which overlap other
        # CDUs
        #self.complex_cdus = self.core.cdus()
        self.complex_cdus = set()
        for e in self.core.cdus():
            members       = self.core.cdu_members(e)
            other_members = set()
            for e2 in self.core.cdus():
                if e != e2: other_members.update(self.core.cdu_members(e2))
            def is_complex(n):
                return self.core.is_cdu(n) or n in other_members
            if any([is_complex(n) for n in members]):
                self.complex_cdus.add(e)

        # Add all of the nodes first
        for node in self.core.edus():
            self._add_edu(node)

        for edge in self.core.relations():
            if edge in self.complex_rels:
                self._add_complex_rel(edge)
            else:
                self._add_simple_rel(edge)

        for edge in self.core.cdus():
            if edge in self.complex_cdus:
                self._add_complex_cdu(edge)
            else:
                self._add_simple_cdu(edge)


