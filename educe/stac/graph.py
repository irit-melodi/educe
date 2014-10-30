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
import re
import textwrap

from pygraph.readwrite import dot
import pydot
import pygraph.classes.hypergraph as gr
import pygraph.classes.digraph as dgr
from pygraph.algorithms import traversal
from pygraph.algorithms import accessibility

from educe.annotation import Annotation
from .. import corpus, stac, annotation
from ..graph import *
import educe.graph

# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------

class MultiheadedCduException(Exception):
    def __init__(self, cdu, *args, **kw):
        self.cdu = cdu
        Exception.__init__(self, *args, **kw)

class Graph(educe.graph.Graph):
    def __init__(self):
        return educe.graph.Graph.__init__(self)

    @classmethod
    def from_doc(cls, corpus, doc_key, pred=lambda x:True):
        return super(Graph, cls).from_doc(corpus, doc_key,
                                          could_include=stac.is_edu,
                                          pred=pred)

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
    # recursive head for CDU
    # --------------------------------------------------

    def cdu_head(self, cdu, sloppy=False):
        """
        Given a CDU, return its head, defined here as the only DU
        that is not pointed to by any other member of this CDU.

        This is meant to approximate the description in Muller 2012
        (/Constrained decoding for text-level discourse parsing/):

        1. in the highest DU in its subgraph in terms of suboordinate
           relations
        2. in case of a tie in #1, the leftmost in terms of coordinate
           relations

        Corner cases:

        * Return None if the CDU has no members (annotation error)
        * If the CDU contains more than one head (annotation error)
          and if sloppy is True, return the textually leftmost one;
          otherwise, raise a MultiheadedCduException
        """
        # pylint: disable=E1101
        # pylint seems confused by our use of inheritence
        if self.has_node(cdu):
            hyperedge = self.mirror(cdu)
        else:
            hyperedge = cdu
        # pylint: enable=E1101

        members    = self.cdu_members(cdu)
        candidates = []
        # pylint: disable=E1101
        # pylint seems confused by our use of inheritence
        for m in members:
            def points_to_me(l): # some other member of this CDU
                                 # points to me via this link
                return l != hyperedge\
                        and self.is_relation(l)\
                        and self.links(l)[1] == m\
                        and self.links(l)[0] in members
            pointed_to = any(points_to_me(l) for l in self.links(m))
            if not (self.is_relation(m) or pointed_to):
                candidates.append(m)
        # pylint: enable=E1101

        if sloppy and not candidates:
            # this can arise if the only members of the CDU form a loop
            for m in members:
                if not self.is_relation(m):
                    candidates.append(m)

        if len(candidates) == 0:
            return None
        elif len(candidates) == 1 or sloppy:
            c = self.sorted_first_widest(candidates)[0]
            if self.is_cdu(c):
                return self.mirror(c)
            else:
                return c
        else:
            raise MultiheadedCduException(cdu)

    def recursive_cdu_heads(self, sloppy=False):
        """
        A dictionary mapping each CDU to its recursive CDU
        head (see `cdu_head`)
        """
        cache = {}
        def get_head(c):
            if c in cache:
                return cache[c]
            else:
                hd = self.cdu_head(c, sloppy)
                if hd is None: return None
                if self.is_cdu(hd):
                    deep_hd = get_head(hd)
                else:
                    deep_hd = hd
                if deep_hd is None:
                    return None
                else:
                    cache[c] = deep_hd
                    return deep_hd
        for c in self.cdus():
            get_head(c)
        return cache

    def without_cdus(self, sloppy=False):
        """
        Return a deep copy of this graph with all CDUs removed.
        Links involving these CDUs will point instead from/to
        their deep heads

        We'll probably deprecate this function, since you could
        just as easily call deepcopy yourself
        """
        g2 = copy.deepcopy(self)
        g2.strip_cdus(sloppy)
        return g2

    def strip_cdus(self, sloppy=False):
        """
        Delete all CDUs in this graph.
        Links involving these CDUs will point instead from/to
        their deep heads
        """
        heads = self.recursive_cdu_heads(sloppy)
        anno_heads = {self.annotation(k): self.annotation(v)
                      for k, v in heads.items()}
        # replace all links to/from cdus with to/from their heads
        for e_edge in self.relations():
            links = self.links(e_edge)
            targets = [heads[self.mirror(l)] if self.is_cdu(l) else l
                       for l in links]
            attrs = self.edge_attributes(e_edge)
            if any(self.is_cdu(l) for l in links):
                # recreate the edge
                self.del_edge(e_edge)
                self.add_edge(e_edge)
                self.add_edge_attributes(e_edge, attrs)
                for l in links:
                    l2 = heads[self.mirror(l)] if self.is_cdu(l) else l
                    if e_edge in self.links(l2):
                        # rare case where we have something that is pointing
                        # to itself
                        continue
                    self.link(l2, e_edge)
        # now that we've pointed everything away, nuke the CDUs
        for e_cdu in self.cdus():
            self.del_node(self.mirror(e_cdu))
            self.del_edge(e_cdu)
        # to be on the safe side, we should also do similar link-rewriting
        # but on the underlying educe.annotation objects layer
        # (symptom of a yucky design) :-(
        for r in self.doc.relations:
            if stac.is_relation_instance(r):
                src = r.source
                tgt = r.target
                src2 = anno_heads.get(src, src)
                tgt2 = anno_heads.get(tgt, tgt)
                r.source = src2
                r.target = tgt2
                r.span = annotation.RelSpan(src2.local_id(), tgt2.local_id())
        # remove the actual CDU objects too
        self.doc.schemas = [s for s in self.doc.schemas if not stac.is_cdu(s)]

    # --------------------------------------------------
    # right frontier constraint
    # --------------------------------------------------

    def sorted_first_widest(self, xs):
        """
        Given a list of nodes, return the nodes ordered by their starting point,
        and in case of a tie their inverse width (ie. widest first).
        """
        def span(n):
            return self.annotation(n).text_span()

        def from_span(sp):
            # negate the endpoint so that if we have a tie on the starting
            # point, the widest span comes first
            return (sp.char_start, 0 - sp.char_end)

        tagged = sorted((from_span(span(x)),x) for x in xs)
        return [x for _,x in tagged]

    def first_widest_dus(self):
        """
        Return discourse units in this graph, ordered by their starting point,
        and in case of a tie their inverse width (ie. widest first)
        """
        def is_interesting_du(n):
            return self.is_edu(n) or\
                (self.is_cdu(n) and self.cdu_members(n))

        dus = list(filter(is_interesting_du,self.nodes()))
        return self.sorted_first_widest(dus)


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
        self.set_name(self._format_name())

    def _format_name(self):
        "graphviz-friendly version of our doc key"
        key = self.doc_key
        name = "_".join([key.doc,
                         key.subdoc,
                         key.stage,
                         key.annotator])
        return re.sub(r'[-\[\]\r\n\.]', "_", name)

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
        if 'highlight' in anno.features:
            attrs['fontcolor'] = anno.features['highlight']
        elif not self._edu_label(anno) or not stac.is_edu(anno):
            attrs['fontcolor'] = 'red'
        self.add_node(pydot.Node(node, **attrs))

    def _rel_label(self, anno):
        return anno.type

    def _simple_rel_attrs(self, anno):
        attrs = educe.graph.DotGraph._simple_rel_attrs(self, anno)

        if 'highlight' in anno.features:
            attrs['fontcolor'] = anno.features['highlight']
            attrs['color'] = anno.features['highlight']
        elif not stac.is_subordinating(anno):
            attrs['fontcolor'] = 'dodgerblue4'
            attrs['color'    ] = 'gray13'
        return attrs

    def _complex_rel_attrs(self, anno):
        midpoint_attrs, attrs1, attrs2 =\
                educe.graph.DotGraph._complex_rel_attrs(self, anno)
        if not stac.is_subordinating(anno):
            midpoint_attrs['fontcolor'] = 'dodgerblue4'
        return (midpoint_attrs, attrs1, attrs2)

    def _simple_cdu_attrs(self, anno):
        attrs = educe.graph.DotGraph._simple_cdu_attrs(self, anno)
        attrs['rank'] = 'same'
        if anno in self.node_order:
            attrs['label'] = '%d. CDU' % self.node_order[anno]
        if 'highlight' in anno.features:
            attrs['color'] = anno.features['highlight']
        return attrs

# ---------------------------------------------------------------------
# enclosure graph
# ---------------------------------------------------------------------


class WrappedToken(Annotation):
    """
    Thin wrapper around POS tagged token which adds a local_id
    field for use by the EnclosureGraph mechanism
    """

    def __init__(self, token):
        self.token = token
        anno_id = WrappedToken._mk_id(token)
        super(WrappedToken, self).__init__(anno_id,
                                           token.span,
                                           "token",
                                           {"tag": token.tag,
                                            "word": token.word})

    @classmethod
    def _mk_id(cls, token):
        """
        Generate a string that could work as a node identifier
        in the enclosure graph
        """
        span = token.text_span()
        return "%s_%s_%d_%d"\
            % (token.word,
               token.tag,
               span.char_start,
               span.char_end)


def _stac_enclosure_ranking(anno):
    """
    Given an annotation, return an integer representing its position in
    a hierarchy of nodes that are expected to enclose each other.

    Smaller negative numbers are higher (say the top of the hiearchy
    might be something like -1000 whereas the very bottom would be 0)
    """
    ranking = {"token": -1,
               "edu": -2,
               "turn": -3,
               "dialogue": -4}

    key = None
    if anno.type == "token":
        key = "token"
    elif educe.stac.is_edu(anno):
        key = "edu"
    elif educe.stac.is_turn(anno):
        key = "turn"
    elif educe.stac.is_dialogue(anno):
        key = "dialogue"

    return ranking[key] if key else 0


class EnclosureGraph(educe.graph.EnclosureGraph):
    """
    An enclosure graph based on STAC conventions
    """
    _BLACKLIST = ["Preference", "Resource", "paragraph"]

    def __init__(self, doc, postags=None):
        annos = [anno for anno in doc.units
                 if anno.type not in EnclosureGraph._BLACKLIST]
        if postags:
            annos += [WrappedToken(tok) for tok in postags]
        super(EnclosureGraph, self).__init__(annos,
                                             key=_stac_enclosure_ranking)


class EnclosureDotGraph(educe.graph.EnclosureDotGraph):
    """
    Conventions for visualising STAC enclosure graphs
    """
    def __init__(self, core):
        super(EnclosureDotGraph, self).__init__(core)

    def _unit_label(self, anno):
        span = anno.text_span()
        if anno.type == "token":
            word = anno.features["word"]
            tag = anno.features["tag"]
            return "%s [%s] %s" % (word, tag, span)
        else:
            return "%s %s" % (anno.type, span)
