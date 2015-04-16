'''
Right frontier constraint and its variants
'''

import collections
import itertools as itr

from educe import stac
from educe.stac.util.context import Context
import educe.stac.util.context
from .annotation import (is_subordinating)


# pylint: disable=too-few-public-methods, no-self-use

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return (itr.chain.from_iterable(itr.combinations(s, r)
        for r in range(len(s)+1)))

def speakers(contexts, anno):
    """ Returns the speakers for given annotation unit

    Takes : contexts (Context dict), Annotation """
    if stac.is_edu(anno):
        edus = [anno]
    else:
        edus = [x for x in anno.terminals() if stac.is_edu(x)]
    return frozenset([contexts[x].speaker() for x in edus])


class BasicRfc(object):
    '''
    The vanilla right frontier constraint ::

        1. X is textually last => RF(X)

        2. Y
           | (sub)
           v
           X

           RF(Y) => RF(X)

        3. X: +----+
              | Y  |
              +----+

           RF(Y) => RF(X)
    '''
    def __init__(self, graph):
        self._graph = graph

    def _build_right_frontier(self, points, last):
        """
        Given a dictionary mapping each node to its closest
        right frontier node, generate a path up that frontier.
        """
        seen = set()
        current = last
        while current in points:
            next_point = points[current]
            if current in seen:
                # corner case: loop in graph
                break
            seen.add(current)
            yield current
            current = next_point

    def _is_on_right_frontier(self, points, last, node):
        """
        Return True if node is on the right frontier as
        represented by the pair points/last.

        This uses `build_frontier`
        """
        return any(fnode == node for fnode in
                   self._build_right_frontier(points, last))

    def _is_incoming_to(self, node, lnk):
        'true if a given link has the given node as target'
        graph = self._graph
        nodes = graph.links(lnk)
        return (graph.is_relation(lnk) and
                len(nodes) == 2 and nodes[1] == node)

    def _frontier_points(self, nodes):
        """
        Given an ordered sequence of nodes, return a dictionary
        mapping each node to the nearest node
        (in the sequence) that either

        * points to it with a subordinating relation
        * includes it as a CDU member
        """
        graph = self._graph

        def position(name):
            'return a relative position for a node'
            if name in nodes:
                return nodes.index(name)
            else:
                return -1

        points = {}
        for node1 in nodes:
            # Computing neighbor of node1
            candidates = []
            for lnk in graph.links(node1):
                if (self._is_incoming_to(node1, lnk) and
                        is_subordinating(graph.annotation(lnk))):
                    # N2 -S> N1
                    node2 = graph.links(lnk)[0]
                    candidates.append((node2, position(node2)))
                elif graph.is_cdu(lnk):
                    # N2 = [...N1...]
                    node2 = graph.mirror(lnk)
                    candidates.append((node2, position(node2)))

            if candidates:
                # Get the last/nearest (in textual order) candidate
                best = max(candidates, key=lambda x: x[1])
                points[node1] = best[0]
            else:
                points[node1] = None

        return points

    def frontier(self):
        """
        Return the list of nodes on the right frontier of a graph
        """
        graph = self._graph
        nodes = graph.first_widest_dus()
        points = self._frontier_points(nodes)
        if nodes:
            last = nodes[-1]
            res = []
            for rfc_node in self._build_right_frontier(points, last):
                res.append(rfc_node)
            return res
        else:
            return []

    def violations(self):
        '''
        Return a list of relation instance names, corresponding to the
        RF violations for the given graph.

        You'll need a stac graph object to interpret these names with.

        :rtype: [string]
        '''
        graph = self._graph
        nodes = graph.first_outermost_dus()
        res = list()
        if len(nodes) < 2:
            return res

        points = self._frontier_points(nodes)
        nexts = itr.islice(nodes, 1, None)
        for last, node1 in itr.izip(nodes, nexts):
            for lnk in graph.links(node1):
                if not self._is_incoming_to(node1, lnk):
                    continue
                node2 = graph.links(lnk)[0]
                if not self._is_on_right_frontier(points, last, node2):
                    res.append(lnk)
        return res


class ThreadedRfc(BasicRfc):
    '''
    Same as BasicRfc except for point 1:

        1. X is the textual last utterance of any speaker => RF(X)
    '''
    def _last_nodes(self):
        """
        Return the dict of node names to the set of last elements up to
        that node, and the last utterances by speakers
        """
        nodes = self._graph.first_widest_dus()
        contexts = Context.for_edus(self._graph.doc)
        doc_speakers = frozenset(ctx.speaker()
            for ctx in contexts.values())

        current_last = dict()
        res = dict()
        for node in nodes:
            anno_node = self._graph.annotation(node)
            res[node] = frozenset(current_last[speaker]
                for speaker in doc_speakers
                if speaker in current_last)

            for speaker in speakers(contexts, anno_node):
                current_last[speaker] = node

        return res, current_last

    def frontier(self):
        """
        Return the list of nodes on the right frontier of a graph.
        """
        graph = self._graph
        nodes = graph.first_widest_dus()
        points = self._frontier_points(nodes)

        lasts = list(self._last_nodes()[1].values())

        if nodes:
            res = []
            for last in lasts:
                res.extend(self._build_right_frontier(points, last))
            return res
        else:
            return []

    def violations(self):
        '''
        Return a list of relation instance names, corresponding to the
        RF violations for the given graph.

        You'll need a stac graph object to interpret these names with.

        :rtype: [string]
        '''
        graph = self._graph
        nodes = graph.first_widest_dus()
        res = list()
        if len(nodes) < 2:
            return res

        lasts, _ = self._last_nodes()
        points = self._frontier_points(nodes)
        nexts = itr.islice(nodes, 1, None)
        for node_tgt in nexts:
            for lnk in graph.links(node_tgt):
                if not self._is_incoming_to(node_tgt, lnk):
                    continue
                # Attachment source
                node_src = graph.links(lnk)[0]
                for last in lasts[node_tgt]:
                    if self._is_on_right_frontier(points, last, node_src):
                        # node_src belongs to some RF
                        break
                else:
                    # node_src doesn't belong to any RF
                    res.append(lnk)
        return res
