# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
STAC-specific conventions related to graphs.
"""

import copy
import collections
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

class DotGraph(educe.graph.DotGraph):
    """
    A dot representation of this graph for visualisation.
    The `to_string()` method is most likely to be of interest here
    """

    def __init__(self, anno_graph):
        educe.graph.DotGraph.__init__(self, anno_graph)

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
        twin = stac.twin(self.corpus, anno)
        edu  = twin if twin is not None else anno
        return stac.dialogue_act(edu)

    def _edu_label(self, anno):
        speech_acts = ", ".join(self._get_speech_acts(anno))
        speaker     = self._get_speaker(anno)
        if speaker is None:
            speaker_prefix = ''
        else:
            speaker_prefix = '(%s) ' % speaker
        return speaker_prefix + "%s [%s]" % (self.doc.text_for(anno), speech_acts)

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
        return attrs
