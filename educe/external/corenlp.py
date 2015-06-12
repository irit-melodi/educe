# Author: Eric Kow
# License: BSD3

"""
Annotations from the CoreNLP pipeline
"""

import codecs
import collections
import copy
import math
import os
import os.path
import subprocess
import sys

import nltk.tree

from educe            import stac, corpus
from educe.annotation import Span, Standoff
from educe.external   import postag
from educe.external.parser import *
import educe.external.stanford_xml_reader as corenlp_xml

class CoreNlpDocument(Standoff):
    """
    All of the CoreNLP annotations for a particular document as instances of
    `educe.annotation.Standoff` or as structures that contain such instances.
    """
    def __init__(self, tokens, trees, deptrees, chains):
        Standoff.__init__(self, None)

        self.tokens   = tokens
        "list of `CoreNlpToken`"

        self.trees    = trees
        "constituency trees"

        self.deptrees = deptrees
        "dependency trees"

        self.chains   = chains
        "coreference chains"

    def _members(self):
        return self.tokens + self.trees

class CoreNlpToken(postag.Token):
    """
    A single token and its POS tag.

    Attributes
    ----------
    features: dict(string, string)
        Additional info found by corenlp about the token
        (eg. `x.features['lemma']`)
    """
    def __init__(self, t, offset, origin=None):
        extent  = t['extent']
        word    = t['word']
        tag     = t['POS']
        span    = Span(extent[0], extent[1] + 1).shift(offset)
        postag.Token.__init__(self, postag.RawToken(word, tag), span)
        self.features = copy.copy(t)
        for k in [ 's_id', 'word', 'extent', 'POS' ]:
            del self.features[k]

    def __str__(self):
        return postag.Token.__str__(self) + ' ' + str(self.features)

    # for debugging
    def __repr__(self):
        return self.word
