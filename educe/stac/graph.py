# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
STAC-specific conventions related to graphs.
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

    def is_cdu(self, x):
        return super(Graph, self).is_cdu(x) and\
                stac.is_cdu(self.annotation(x))

    def is_edu(self, x):
        return super(Graph, self).is_edu(x) and\
                stac.is_edu(self.annotation(x))

    def is_relation(self, x):
        return super(Graph, self).is_relation(x) and\
                stac.is_relation_instance(self.annotation(x))

    # --------------------------------------------------
    # right frontier constraint
    # --------------------------------------------------

    def first_widest_dus(self):
        """
        Return discourse units in this graph, ordered by their starting point,
        and in case of a tie their inverse width (ie. widest first)
        """
        sp_edus = []
        sp_cdus = []

        def span(n):
            return self.annotation(n).text_span(self.doc)

        def from_span(sp):
            # negate the endpoint so that if we have a tie on the starting
            # point, the widest span comes first
            return (sp.char_start, 0 - sp.char_end)

        for n in self.nodes():
            if self.is_edu(n):
                sp       = span(n)
                position = from_span(sp)
                sp_edus.append((position,sp,n))
            elif self.is_cdu(n) and self.cdu_members(n):
                sp       = span(n)
                position = from_span(sp)
                sp_cdus.append((position,sp,n))

        return [x[2] for x in sorted(sp_edus + sp_cdus)]

    def _build_right_frontier(self, points, last):
        """
        Given a dictionary mapping each node to its closest
        right frontier node, generate a path up that frontier.
        """
        frontier = []
        current  = last
        while current in points:
            next    = points[current]
            yield current
            current = next

    def _is_on_right_frontier(self, points, last, node):
        """
        Return True if node is on the right frontier as
        represented by the pair points/last.

        This uses `build_frontier`
        """
        return any(fnode == node for fnode in
                   self._build_right_frontier(points, last))

    def _frontier_points(self, nodes):
        """
        Given an ordered sequence of nodes in this graph return a dictionary
        mapping each node to the nearest node (in the sequence) that either

        * points to it with a subordinating relation
        * includes it as a CDU member
        """
        points = {}
        def position(n):
            if n in nodes:
                return nodes.index(n)
            else:
                return -1

        for n1 in nodes:
            candidates = []

            def is_incoming_subordinate_rel(l):
                ns = self.links(l)
                return self.is_relation(l)\
                        and stac.is_subordinating(self.annotation(l))\
                        and len(ns) == 2 and ns[1] == n1

            def add_candidate(n2):
                candidates.append((n2,position(n2)))

            for l in self.links(n1):
                if is_incoming_subordinate_rel(l):
                    n2 = self.links(l)[0]
                    add_candidate(n2)
                elif self.is_cdu(l):
                    n2 = self.mirror(l)
                    add_candidate(n2)

            if candidates:
                best = max(candidates, key=lambda x:x[1])
                points[n1] = best[0]
            else:
                points[n1] = None

        return points

    def right_frontier_violations(self):
        nodes      = self.first_widest_dus()
        violations = collections.defaultdict(list)
        if len(nodes) < 2:
            return violations

        points = self._frontier_points(nodes)
        nexts  = itertools.islice(nodes, 1, None)
        for last,n1 in itertools.izip(nodes, nexts):
            def is_incoming(l):
                ns = self.links(l)
                return self.is_relation(l) and len(ns) == 2 and ns[1] == n1

            for l in self.links(n1):
                if not is_incoming(l): continue
                n2 = self.links(l)[0]
                if not self._is_on_right_frontier(points, last, n2):
                    violations[n2].append(l)
        return violations

class DotGraph(educe.graph.DotGraph):
    """
    A dot representation of this graph for visualisation.
    The `to_string()` method is most likely to be of interest here
    """

    def __init__(self, anno_graph):
        doc   = anno_graph.doc
        nodes = anno_graph.first_widest_dus()
        self.node_order = {}
        for i,n in enumerate(nodes):
            self.node_order[anno_graph.annotation(n)] = i
        educe.graph.DotGraph.__init__(self, anno_graph)

    def _get_turn_info(self, u):
        enclosing_turns = [ t for t in self.turns if t.span.encloses(u.span) ]
        if len(enclosing_turns) > 0:
            turn      = enclosing_turns[0]
            speaker   = turn.features['Emitter']
            turn_text = stac.split_turn_text(self.doc.text(turn.span))[0]
            turn_id   = turn_text.split(':')[0].strip()
            return speaker, turn_id
        else:
            return None, None

    def _get_speech_acts(self, anno):
        # In discourse annotated part of the corpus, all segments have
        # type 'Other', which isn't too helpful. Try to recover the
        # speech act from the unit equivalent to this document
        twin = stac.twin(self.corpus, anno)
        edu  = twin if twin is not None else anno
        return stac.dialogue_act(edu)

    def _get_addressee(self, anno):
        # In discourse annotated part of the corpus, all segments have
        # type 'Other', which isn't too helpful. Try to recover the
        # speech act from the unit equivalent to this document
        twin = stac.twin(self.corpus, anno)
        edu  = twin if twin is not None else anno
        return edu.features.get('Addressee', None)

    def _edu_label(self, anno):
        speech_acts  = ", ".join(self._get_speech_acts(anno))
        speaker, tid = self._get_turn_info(anno)
        addressee    = self._get_addressee(anno)

        if speaker is None:
            speaker_prefix = '(%s)'  % tid
        elif addressee is None:
            speaker_prefix = '(%s: %s) ' % (tid, speaker)
        else:
            speaker_prefix = '(%s: %s to %s) ' % (tid, speaker, addressee)

        if callable(getattr(anno, "text_span", None)):
            span = ' ' + str(anno.text_span())
        else:
            span = ''
        text     = self.doc.text(anno.span)
        return "%s%s [%s]%s" % (speaker_prefix, text, speech_acts, span)

    def _add_edu(self, node):
        anno  = self.core.annotation(node)
        label = self._edu_label(anno)
        attrs = { 'label' : textwrap.fill(label, 30)
                , 'shape' : 'plaintext'
                }
        if not self._edu_label(anno) or not stac.is_edu(anno):
            attrs['fontcolor'] = 'red'
        self.add_node(pydot.Node(node, **attrs))

    def _rel_label(self, anno):
        return anno.type

    def _simple_rel_attrs(self, anno):
        attrs = educe.graph.DotGraph._simple_rel_attrs(self, anno)
        if anno.type not in stac.subordinating_relations:
            attrs['fontcolor'] = 'dodgerblue4'
            attrs['color'    ] = 'gray13'
        return attrs

    def _complex_rel_attrs(self, anno):
        midpoint_attrs, attrs1, attrs2 =\
                educe.graph.DotGraph._complex_rel_attrs(self, anno)
        if anno.type not in stac.subordinating_relations:
            midpoint_attrs['fontcolor'] = 'dodgerblue4'
        return (midpoint_attrs, attrs1, attrs2)

    def _simple_cdu_attrs(self, anno):
        attrs = educe.graph.DotGraph._simple_cdu_attrs(self, anno)
        attrs['rank'] = 'same'
        if anno in self.node_order:
            attrs['label'] = '%d. CDU' % self.node_order[anno]
        return attrs

