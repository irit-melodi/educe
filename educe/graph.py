# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
Graph representation of discourse structure.
Classes of interest:

* Graph: the core structure, use the `Graph.from_doc` factory
  method to build one out of an `educe.annotation` document.

* DotGraph: visual representation, built from `Graph`.
  You probably want a project-specific variant to get more
  helpful graphs, see eg. `educe.stac.Graph.DotGraph`

.. _hypergraphs:

Educe hypergraphs
~~~~~~~~~~~~~~~~~
Somewhat tricky hypergraph representation of discourse structure.

    * a node  for every elementary discourse unit
    * a hyperedge for every relation instance [#]_
    * a hyperedge for every complex discourse unit
    * (the tricky bit) for every (hyper)edge `e_x` in the graph,
      introduce a "mirror node" `n_x` for that edge
      (this node also has `e_x` as its "mirror edge")

The tricky bit is a response to two issues that arise: (A) how do we point
to a CDU? Our hypergraph formalism and library doesn't have a notion of
pointing to hyperedges (only nodes) and (B) what do we do about
misannotations where we have relation instances pointing to relation
instances? *A* is the most important one to address (in principle, we could
just treat *B* as an error and raise an exception), but for now we decide to
model both scenarios, and the same "mirror" mechanism above.

The mirrors are a bit problematic because are not part of the formal graph
structure (think of them as extra labels). This could lead to some
seriously unintuitive consequences when traversing the graph. For example,
if you two DUs A and B connected by an Elab instance, and if that instance
is itself (bizarrely) connected to some other DU, you might intuitively
expect A, B, and C to all form one connected component ::

            A
            |
       Elab |
            o--------->C
            | Comment
            |
            v
            B

Alas, this is not so! The reality is a bit messier, with there being no
formal relationship between edge and mirror ::

            A
            |
       Elab |  n_ab
            |  o--------->C
            |    Comment
            |
            v
            B

The same goes for the connectedness of things pointing to CDUs
and with their members.  Looking at pictures, you might
intuitively think that if a discourse unit (A) were connected to
a CDU, it would also be connected to the discourse units within ::

            A
            |
       Elab |
            |
            v
            +-----+
            | B C |
            +-----+

The reality is messier for the same reasons above ::

            A
            |
       Elab |      +-----+ e_bc
            |      | B C |
            v      +-----+
            n_bc

.. [#] just a binary hyperedge, ie. like an edge in a regular graph. As
   these are undirected, we take the convention that the the first link is
   the tail (from) and the second link is the tail (to).

Classes
~~~~~~~
"""

from __future__ import print_function
import copy
import collections
import subprocess
import textwrap

import pydot
import pygraph.classes.hypergraph as gr
import pygraph.classes.digraph    as dgr
from pygraph.algorithms import accessibility

# pylint: disable=too-few-public-methods, star-args

class DuplicateIdException(Exception):
    '''Condition that arises in inconsistent corpora'''
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
        DEPRECATED (renamed 2013-11-19): use `self.nodeform(x)` instead
        """
        return self.nodeform(x)

    def nodeform(self, x):
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

    def edgeform(self, x):
        """
        Return the argument if it is an edge id, or its mirror if it's an
        edge id

        (This is possible because every edge in the graph has a node that
        corresponds to it)
        """
        if self.has_edge(x):
            return x
        else:
            return self.mirror(x)

class Graph(gr.hypergraph, AttrsMixin):
    """
    Hypergraph representation of discourse structure.
    See the section on Educe hypergraphs_

    You most likely want to use `Graph.from_doc` instead of
    instantiating an instance directly

    Every node/hyperedge is represented as string unique
    within the graph. Given one of these identifiers `x`
    and a graph `g`:

        * `g.type(x)` returns one of the strings
          "EDU", "CDU", "rel"
        * `g.annotation(x)` returns an
          educe.annotation object
        * for relations and CDUs, if `e_x` is the edge representation
          of the relation/cdu, `g.mirror(x)` will return its mirror
          node `n_x` and vice-versa

    TODOS:

        * TODO: Currently we use educe.annotation objects to represent the
          EDUs, CDUs and relations, but this is likely a bit too low-level to
          be helpful. It may be nice to have higher-level EDU and CDU
          objects instead


    """

    def __init__(self):
        AttrsMixin.__init__(self)
        gr.hypergraph.__init__(self)

    @classmethod
    def from_doc(cls, corpus, doc_key,
                 could_include=lambda x: False,
                 pred=lambda x: True):
        """
        Return a graph representation of a document

        Note: check the project layer for a version of this function
        which may be more appropriate to your project

        :param corpus: educe corpus dictionary
        :type  corpus: dict from `FileId` to documents

        :param doc_key: key pointing to the document
        :type  doc_key: `FileId`

        :param  could_include: predicate on unit level annotations that
            should be included regardless of whether or not we have links
            to them
        :type   could_include: annotation -> boolean

        :param  pred: predicate on annotations providing some requirement
            they must satisfy in order to be taken into account (you might
            say that `could_include` gives; and `pred` takes away)
        :type   pred: annotation -> boolean
        """
        grph = cls()
        doc = corpus[doc_key]
        grph.corpus = corpus
        grph.doc_key = doc_key
        grph.doc = doc

        # objects that are pointed to by a relations or schemas
        included = []
        included.extend(x.local_id() for x in doc.units
                        if could_include(x))
        for anno in doc.relations:
            if pred(anno):
                included.extend([anno.span.t1, anno.span.t2])
        for anno in doc.schemas:
            if pred(anno):
                included.extend(anno.span)

        nodes = []
        edges = []

        edus = [x for x in doc.units   if x.local_id() in included and pred(x)]
        rels = [x for x in doc.relations if pred(x)]
        cdus = [s for s in doc.schemas if pred(s)]

        nodes.extend(grph._unit_node(x) for x in edus)
        nodes.extend(grph._rel_node(x) for x in rels)
        nodes.extend(grph._schema_node(x)for x in cdus)
        edges.extend(grph._rel_edge(x) for x in rels)
        edges.extend(grph._schema_edge(x) for x in cdus)

        for node, attrs in nodes:
            if not grph.has_node(node):
                grph.add_node(node)
                for anno in attrs.items():
                    grph.add_node_attribute(node, anno)
            else:
                raise DuplicateIdException(node)

        for edge, attrs, links in edges:
            if not grph.has_edge(edge):
                grph.add_edge(edge)
                grph.add_edge_attributes(edge, attrs.items())
                for lnk in links:
                    grph.link(lnk, edge)

        return grph

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

        :param nodeset: only copy nodes with these names
        :type  nodeset: iterable of strings
        """
        g = self.__class__()
        g.corpus = self.corpus
        g.doc_key = self.doc_key
        g.doc = self.doc

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
        keep_growing = True
        edges_remaining = self.hyperedges()
        edges_wanted = set()
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

        Each connected component set can be passed to `self.copy()`
        to be copied as a subgraph.

        This builds on python-graph's version of a function with the
        same name but also adds awareness of our conventions about there
        being both a node/edge for relations/CDUs.
        """
        ccs = accessibility.connected_components(self)
        subgraphs = collections.defaultdict(set)
        for node, i in ccs.items():
            subgraphs[i].add(node)

        # the basic idea here: if any member of connected component is
        # one of our hybrid node/edge creatures, we need to help the
        # graph library recognise that that anything connected *via*
        # the edge should also be considered as connected *to* the
        # edge, so we merge the components
        eaten = set()
        merged = {}
        prior = subgraphs.keys()
        while sorted(merged.keys()) != prior:
            prior = sorted(merged.keys())
            merged = {}
            for k in subgraphs:
                if k in eaten:
                    continue
                subg = subgraphs[k]
                merged[k] = copy.copy(subg)
                for n in subg:
                    e = self.mirror(n)
                    if e is not None:
                        links = set(self.links(e))
                        for k2 in subgraphs.keys():
                            links2 = subgraphs[k2]
                            if k2 != k and not links2.isdisjoint(links):
                                eaten.add(k2)
                                merged[k] |= links2
            subgraphs = merged

        ccs = frozenset([frozenset(v) for v in subgraphs.values()])
        return ccs

    def _attrs(self, x):
        if self.has_edge(x):
            return self.edge_attributes_dict(x)
        elif self.has_node(x):
            return self.node_attributes_dict(x)
        else:
            raise Exception('Tried to get attributes of non-existing object ' + str(x))

    def relations(self):
        """
        Set of relation edges representing the relations in the graph.
        By convention, the first link is considered the source and the
        the second is considered the target.
        """
        return frozenset(e for e in self.hyperedges()
                         if self.is_relation(e))

    def edus(self):
        """
        Set of nodes representing elementary discourse units
        """
        return frozenset(e for e in self.nodes()
                         if self.is_edu(e))

    def cdus(self):
        """
        Set of hyperedges representing complex discourse units.

        See also `cdu_members`
        """
        return frozenset(e for e in self.hyperedges()
                         if self.is_cdu(e))

    def rel_links(self, edge):
        """
        Given an edge in the graph, return a tuple of its source and
        target nodes.

        If the edge has only a single link, we assume it's a loop and
        return the same value for both
        """
        links = self.links(edge)
        if len(links) == 2:
            return tuple(links)
        elif len(links) == 1:
            return links[0], links[0]
        else:
            raise Exception("confused by relation edge with 3+ links")

    def containing_cdu(self, node):
        """
        Given an EDU (or CDU, or relation instance), return immediate
        containing CDU (the hyperedge) if there is one or None otherwise.
        If there is more than one containing CDU, return one of them
        arbitrarily.
        """
        for node in self.links(self.nodeform(node)):
            if self.is_cdu(node):
                return node
        return None

    def containing_cdu_chain(self, node):
        """
        Given an annotation, return a list which represents its
        containing CDU, the container's container, and forth.
        Return the empty list if no CDU contains this one.
        """
        res = []
        while node:
            node = self.nodeform(node)
            res.append(node)
            node = self.containing_cdu(node)
        return res[1:]  # drop the node itself

    def cdu_members(self, cdu, deep=False):
        """
        Return the set of EDUs, CDUs, and relations which can be considered as
        members of this CDU.

        This is shallow by default, in that we only return the immediate
        members of the CDU.  If `deep==True`, also return members of CDUs
        that are members of (members of ..) this CDU.
        """

        hyperedge = self.edgeform(cdu)
        members = set(self.links(hyperedge))

        if deep:
            for m in list(members):
                if self.is_cdu(m):
                    members.update(self.cdu_members(m, deep=deep))

        return frozenset(members)

    def _mk_guid(self, x):
        return self.doc_key.mk_global_id(x)

    def _mk_edge_id(self, x):
        return 'e_' + self._mk_guid(x)

    def _mk_node_id(self, x):
        return 'n_' + self._mk_guid(x)

    def _mk_node(self, anno, ntype, mirrored=False):
        # a node is mirrored if there is a also an edge
        # corresponding to the same object
        local_id = anno.local_id()
        node_id = self._mk_node_id(local_id)
        attrs = {'type': ntype,
                 'annotation': anno}
        if mirrored:
            attrs['mirror'] = self._mk_edge_id(local_id)
        else:
            attrs['mirror'] = None
        return (node_id, attrs)

    def _mk_edge(self, anno, etype, members, mirrored=False):
        local_id = anno.local_id()
        edge_id = self._mk_edge_id(local_id)
        attrs = {'type': etype,
                 'annotation': anno}
        if mirrored:
            attrs['mirror'] = self._mk_node_id(local_id)
        else:
            attrs['mirror'] = None

        links = [self._mk_node_id(m) for m in members]
        return (edge_id, attrs, links)

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

    def _repr_dot_(self):
        """Ipython magic: show Graphviz dot representation of the graph

        Note that this does not ship with iPython but is used by our
        `_repr_svg_` implementation
        """
        return DotGraph(self).to_string()

    def _repr_svg_(self):
        """Ipython magic: show SVG representation of the graph"""
        dot_string = self._repr_dot_()
        try:
            process = subprocess.Popen(['dot', '-Tsvg'], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            raise Exception('Cannot find the dot binary from Graphviz package')
        out, err = process.communicate(dot_string)
        if err:
            raise Exception('Cannot create svg representation by running dot from string\n:%s' % dot_string)
        return out

# ---------------------------------------------------------------------
# visualisation
# ---------------------------------------------------------------------

class DotGraph(pydot.Dot):
    """
    A dot representation of this graph for visualisation.
    The `to_string()` method is most likely to be of interest here

    This is fairly abstract and unhelpful.  You probably want the
    project-layer extension instead, eg. `educe.stac.graph`
    """

    def _edu_label(self, anno):
        '''string to display for an EDU'''
        return anno.type

    def _rel_label(self, anno):
        '''string to display for a relation instance'''
        return anno.type

    def _simple_rel_attrs(self, anno):
        '''formatting options for a relation instance'''
        return\
            {'label': ' ' + self._rel_label(anno),
             'shape': 'plaintext',
             'fontcolor': 'blue'}

    def _complex_rel_attrs(self, anno):
        """
        Return attributes for
        (midpoint, to midpoint, from midpoint)
        """
        midpoint_attrs =\
            {'label': self._rel_label(anno),
             'style': 'dotted',
             'fontcolor': 'blue'}
        attrs1 = {'arrowhead' : 'tee',
                  'arrowsize' : '0.5'}
        attrs2 = {}
        return (midpoint_attrs, attrs1, attrs2)

    def _simple_cdu_attrs(self, anno):
        return {'color': 'lightgrey'}

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
            if self.core.is_cdu(proxy_target):
                proxy_target, _ = self.__point(proxy_target, key)
            elif self.core.has_edge(proxy_target):
                proxy_target = self.core.mirror(proxy_target)
            proxy_target = self._dot_id(proxy_target)
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
        anno = self.core.annotation(node)
        label = self._edu_label(anno)
        attrs = {'label': textwrap.fill(label, 30),
                 'shape': 'plaintext'}
        if not self._edu_label(anno):
            attrs['fontcolor'] = 'red'
        self.add_node(pydot.Node(node, **attrs))

    def _add_simple_rel(self, hyperedge):
        anno = self.core.annotation(hyperedge)
        links = self.core.links(hyperedge)
        attrs = self._simple_rel_attrs(anno)

        if len(links) == 2:
            link1_, link2_ = links
            link1, attrs1 = self._point_from(link1_)
            link2, attrs2 = self._point_to(link2_)
        elif len(links) == 1:
            link_ = links[0]
            link1, attrs1 = self._point_from(link_)
            link2, attrs2 = self._point_from(link_)
        else:
            raise Exception("confused by relation edge with 3+ links")

        attrs.update(attrs1)
        attrs.update(attrs2)
        self.add_edge(pydot.Edge(link1, link2, **attrs))

    def _add_complex_rel(self, hyperedge):
        anno = self.core.annotation(hyperedge)
        links = self.core.links(hyperedge)
        link1_, link2_ = links
        midpoint_attrs, attrs1, attrs2 = self._complex_rel_attrs(anno)
        link1, attrs1_ = self._point_from(link1_)
        link2, attrs2_ = self._point_to(link2_)
        attrs1.update(attrs1_)
        attrs2.update(attrs2_)

        midpoint_id = self.core.node(hyperedge)
        midpoint = pydot.Node(midpoint_id, **midpoint_attrs)
        edge1 = pydot.Edge(link1, midpoint_id, **attrs1)
        edge2 = pydot.Edge(midpoint_id, link2, **attrs2)
        self.add_node(midpoint)
        self.add_edge(edge1)
        self.add_edge(edge2)

    def _add_simple_cdu(self, hyperedge, parent=None):
        """
        Straightforward CDU that can be supported as a cluster.
        """
        anno = self.core.annotation(hyperedge)
        attrs = self._simple_cdu_attrs(anno)
        if len(self.complex_cdus) > 0 and 'label' not in attrs:
            # complex CDUs have a CDU node, so I thought it might be
            # less confusing in those cases to also label the simple
            # CDUs so the user knows it's the same thing
            attrs['label'] = 'CDU'
        subg = pydot.Subgraph(self._dot_id(hyperedge), **attrs)
        local_nodes = self.core.links(hyperedge)
        local_nodes2 = []

        # take into account links to relations (sigh)
        def is_enclosed(l):
            return (l != hyperedge and
                    l in self.complex_rels and
                    all(x in local_nodes for x in self.core.links(l)))
        for node in local_nodes:
            if self.core.is_relation(node):
                local_nodes2.append(node)
            else:
                local_nodes2.append(node)
                rlinks = [x for x in self.core.links(node)
                          if is_enclosed(x)]
                local_nodes2.extend(self.core.mirror(l) for l in rlinks)

        for node in local_nodes2:
            if self.core.is_cdu(node):
                self._add_simple_cdu(self.core.mirror(node), parent=subg)
            else:
                subg.add_node(pydot.Node(node))

        if parent:
            parent.add_subgraph(subg)
        else:
            self.add_subgraph(subg)

    def _add_complex_cdu(self, hyperedge):
        """
        Yes, a complex "complex discourse unit".

        The idea is to to have a node representing a CDU and dotted lines
        pointing to its members.  It's actually simpler in implementation
        terms but more complex visually

        This is to deal with weird CDUs that do not have a sensible
        CDU-box representation (for example if we have non-embedded
        CDUs that share items)
        """
        attrs = {'color': 'grey',
                 'label': 'CDU',
                 'shape': 'rectangle'}
        cdu_id = self._dot_id(hyperedge)
        self.add_node(pydot.Node(cdu_id, **attrs))
        for node in self.core.links(hyperedge):
            edge_attrs = {'style': 'dashed',
                          'color': 'grey'}
            dest, attrs_ = self._point_to(node)
            edge_attrs.update(attrs_)
            self.add_edge(pydot.Edge(cdu_id, dest, **edge_attrs))

    def __init__(self, anno_graph):
        """
        Args

            anno_graph (Graph):  abstract annotation graph
        """
        self.core = anno_graph
        self.doc = self.core.doc
        self.doc_key = self.core.doc_key
        self.corpus = self.core.corpus
        # 2017-03-17 add 'NonplayerTurn' ; this is a slippery slope but
        # 'Turn' and 'NonplayerTurn' are both rather STAC-specific anyway
        self.turns = [u for u in self.core.doc.units
                      if u.type in ('Turn', 'NonplayerTurn')]
        super(DotGraph, self).__init__(compound='true')
        self.set_name('hypergraph')

        # rels which are the target of links
        self.complex_rels = set()
        for n in self.core.nodes():
            for n2 in self.core.neighbors(n):
                if self.core.is_relation(n2):
                    e2 = self.core.mirror(n2)
                    self.complex_rels.add(e2)

        # CDUs which overlap other CDUs
        #self.complex_cdus = self.core.cdus()
        self.complex_cdus = set()
        for e in self.core.cdus():
            members = self.core.cdu_members(e)
            other_members = set()
            for e2 in self.core.cdus():
                if e != e2:
                    other_members.update(self.core.cdu_members(e2))
            if any(n in other_members for n in members):
                self.complex_cdus.add(e)

        # CDUs which are contained in another
        self.contained_cdus = set()
        for e in self.core.cdus():
            for n2 in self.core.cdu_members(e):
                e2 = self.core.mirror(n2)
                if self.core.is_cdu(n2) and e2 not in self.complex_cdus:
                    self.contained_cdus.add(e2)

        # Add all of the nodes first
        for node in sorted(self.core.edus(),
                           key=lambda x: self.core.annotation(x).span):
            self._add_edu(node)

        # Add nodes that have some sort of error condition or another
        for edge in (self.core.relations() | self.core.cdus()):
            for node in self.core.links(edge):
                if not (self.core.is_edu(node) or\
                        self.core.is_relation(node) or\
                        self.core.is_cdu(node)):
                    self._add_edu(node)

        for edge in self.core.relations():
            if edge in self.complex_rels:
                self._add_complex_rel(edge)
            else:
                self._add_simple_rel(edge)

        for edge in self.core.cdus():
            if edge in self.contained_cdus:
                continue
            elif edge in self.complex_cdus:
                self._add_complex_cdu(edge)
            else:
                self._add_simple_cdu(edge)

# ---------------------------------------------------------------------
# enclosure graphs
# ---------------------------------------------------------------------

class EnclosureGraph(dgr.digraph, AttrsMixin):
    """
    Caching mechanism for span enclosure. Given an iterable of Annotation,
    return a directed graph where nodes point to the largest nodes
    they enclose (i.e. not to nodes that are enclosed by intermediary
    nodes they point to).  As a slight twist, we also allow nodes to
    redundantly point to enclosed nodes of the same typ.

    This *should* give you a multipartite graph with each layer
    representing a different type of annotation, but no promises!  We can't
    guarantee that the graph will be nicely layered because the annotations
    may be buggy (either nodes wrongly typed, or nodes of the same type
    that wrongly enclose each other), so you should not rely on this
    property aside from treating it as an optimisation.

    Note: there is a corner case for nodes that have the same span.
    Technically a span encloses itself, so the graph could have a loop.
    If you supply a sort key that differentiates two nodes, we use it
    as a tie-breaker (first node encloses second). Otherwise, we
    simply exclude both links.

    NB: nodes are labelled by their annotation id

    Initialisation parameters

    * annotations - iterable of Annotation
    * key - disambiguation key for nodes with same span
            (annotation -> sort key)
    """
    def __init__(self, annotations, key=None):
        super(EnclosureGraph, self).__init__()
        AttrsMixin.__init__(self)
        self._build_enclosure_graph(annotations, key)

    def _build_enclosure_graph(self, annotations, key=None):
        # text spans can be expensive to compute if there
        # are nested elements; cache them to avoid
        # recomputation
        spans = {}
        for anno in annotations:
            spans[anno] = anno.text_span()

        def can_enclose(anno1, anno2):
            span1 = spans[anno1]
            span2 = spans[anno2]
            if anno1 == anno2:
                return False
            elif span1.encloses(span2):
                return span1 != span2 or\
                    (key and key(anno1) < key(anno2))
            else:
                return False

        def connect_to_enclosed(mega, mini):
            """
            Given a enclosing and a subgraph represented by a (candidate)
            enclosed node, walk down the subgraph trying to connect the
            enclosing node to the largest node we can find
            """
            if not spans[mega].overlaps(spans[mini]):
                return
            elif can_enclose(mega, mini):
                self._add_edge(mega, mini)
                id_mini = self._mk_node_id(mini)
                # yucky extra step to also enclose subnodes
                # of the same type (let these be on the same layer)
                for id_kid in self.neighbors(id_mini):
                    kid = self.annotation(id_kid)
                    if kid.type == mini.type:
                        connect_to_enclosed(mega, kid)
            else:
                id_mini = self._mk_node_id(mini)
                for id_kid in self.neighbors(id_mini):
                    kid = self.annotation(id_kid)
                    connect_to_enclosed(mega, kid)

        of_width = collections.defaultdict(list)
        for anno in annotations:
            node, attrs = self._mk_node(anno)
            self.add_node(node)
            for pair in attrs.items():
                self.add_node_attribute(node, pair)
            of_width[spans[anno].length()].append(anno)

        narrow = []
        for width in sorted(of_width):
            mk_hidden = []
            layer = of_width[width]
            narrow.extend(layer)
            for mega in layer:
                for mini in narrow:
                    connect_to_enclosed(mega, mini)
                    if can_enclose(mega, mini):
                        mk_hidden.append(mini)
            narrow = [x for x in narrow if x not in mk_hidden]

    def _mk_node_id(self, anno):
        return anno.local_id()

    def _mk_node(self, anno):
        # a node is mirrored if there is a also an edge
        # corresponding to the same object
        node_id  = self._mk_node_id(anno)
        attrs = {'type': anno.type,
                 'annotation' : anno}
        return (node_id, attrs)

    def _add_edge(self, anno1, anno2):
        id1 = self._mk_node_id(anno1)
        id2 = self._mk_node_id(anno2)
        if not self.has_edge((id1, id2)):
            self.add_edge((id1, id2))

    def _attrs(self, x):
        return self.node_attributes_dict(x)

    def inside(self, annotation):
        """
        Given an annotation, return all annotations that are
        directly within it.
        Results are returned in the order of their local id
        """
        n1 = self._mk_node_id(annotation)
        return sorted([self.annotation(n2) for n2 in self.neighbors(n1)],
                      key=lambda x: x.text_span())

    def outside(self, annotation):
        """
        Given an annotation, return all annotations it is
        directly enclosed in.  Results are returned in the
        order of their local id
        """
        n1 = self._mk_node_id(annotation)
        return sorted([self.annotation(n2) for n2 in self.incidents(n1)],
                      key=lambda x: x.text_span())


class EnclosureDotGraph(pydot.Dot):

    def _add_unit(self, node):
        anno  = self.core.annotation(node)
        label = self._unit_label(anno)
        attrs = {'label' : textwrap.fill(label, 30),
                 'shape' : 'plaintext'}
        self.add_node(pydot.Node(node, **attrs))

    def _add_edge(self, edge):
        (node1, node2) = edge
        attrs = {}
        self.add_edge(pydot.Edge(node1, node2, **attrs))

    def _unit_label(self, anno):
        return "%s %s" % (anno.type, anno.text_span())

    def __init__(self, enc_graph):
        super(EnclosureDotGraph, self).__init__()
        self.core = enc_graph

        def node_sort_key(node):
            return self.core.annotation(node).text_span()

        for n in sorted(self.core.nodes(), key=node_sort_key):
            self._add_unit(n)
        for e in self.core.edges():
            self._add_edge(e)
