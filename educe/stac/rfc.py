'''
Right frontier constraint and its variants
'''

import collections
import itertools as itr

from educe import stac
from educe.stac.context import Context
from .annotation import is_subordinating


# pylint: disable=too-few-public-methods, no-self-use

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return (itr.chain.from_iterable(itr.combinations(s, r)
                                    for r in range(len(s) + 1)))


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
        self._nodes = graph.first_outermost_dus()
        self._points = self._frontier_points(self._nodes)

    def _build_frontier(self, last):
        """
        Return the frontier points of the graph with the given
        node as last.
        """
        return self._build_frontier_from([last])

    def _build_frontier_from(self, starts):
        """
        Given a dictionary mapping each node to its closest
        right frontier nodes and start nodes, generate a path
        up that frontier.
        """
        seen = set()
        points = self._points
        candidates = collections.deque(starts)
        while candidates:
            current = candidates.popleft()
            if current in seen:
                continue
            seen.add(current)
            yield current
            if current in points:
                candidates.extend(points[current])

    def _is_on_frontier(self, last, node):
        """
        Return True if node is on the right frontier as
        represented by the pair points/last.

        This uses `build_frontier`
        """
        return any(fnode == node for fnode in
                   self._build_frontier(last))

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
        gra = self._graph

        points = dict()
        for node1 in nodes:
            # Computing neighbors of node1
            neighbors = list()
            for lnk in gra.links(node1):
                if (self._is_incoming_to(node1, lnk) and
                        is_subordinating(gra.annotation(lnk))):
                    # N2 -S> N1
                    node2, _ = gra.rel_links(lnk)
                    neighbors.append(node2)
                elif gra.is_cdu(lnk):
                    # N2 = [...N1...]
                    node2 = gra.mirror(lnk)
                    neighbors.append(node2)
            points[node1] = neighbors
        return points

    def frontier(self):
        """
        Return the list of nodes on the right frontier of the whole graph
        """
        if not self._nodes:
            return []
        last = self._nodes[-1]
        return list(self._build_frontier(last))

    def violations(self):
        '''
        Return a list of relation instance names, corresponding to the
        RF violations for the given graph.

        You'll need a stac graph object to interpret these names with.

        :rtype: [string]
        '''
        graph = self._graph
        nodes = self._nodes
        if len(nodes) < 2:
            return list()

        violations = list()
        for i, new_node in enumerate(nodes):
            last_node = nodes[i-1] if i > 0 else None
            for lnk in graph.links(new_node):
                if not self._is_incoming_to(new_node, lnk):
                    continue
                src_node, _ = graph.rel_links(lnk)
                if (last_node is None
                    or not self._is_on_frontier(last_node, src_node)):
                    # add link to set of violations
                    violations.append(lnk)

        return violations


class ThreadedRfc(BasicRfc):
    '''
    Same as BasicRfc except for point 1:

        1. X is the textual last utterance of any speaker => RF(X)
    '''
    def __init__(self, graph):
        super(ThreadedRfc, self).__init__(graph)
        self._last = self._last_nodes()

    def _last_nodes(self):
        """
        Return the dict of node names to the set of last elements up to
        that node (included)
        """
        nodes = self._nodes
        contexts = Context.for_edus(self._graph.doc)
        doc_speakers = frozenset(ctx.speaker()
                                 for ctx in contexts.values())

        current_last = dict()
        last_nodes = dict()
        for node in nodes:
            anno_node = self._graph.annotation(node)
            for speaker in speakers(contexts, anno_node):
                current_last[speaker] = node

            last_nodes[node] = frozenset(current_last[speaker]
                                         for speaker in doc_speakers
                                         if speaker in current_last)

        return last_nodes

    def _build_frontier(self, last):
        """
        Return the frontier points of the graph with
        the given node as last.
        """
        return self._build_frontier_from(self._last[last])
