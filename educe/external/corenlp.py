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

    Fields:

        * tokens   - `CoreNlpToken` annotations
        * trees    - constituency trees
        * deptrees - dependency   trees
    """
    def __init__(self, tokens, trees, deptrees):
        Standoff.__init__(self, None)
        self.tokens   = tokens
        self.trees    = trees
        self.deptrees = deptrees

    def _members(self, doc):
        """
        Note that the doc field is ignored here and for any members
        of this document
        """
        return self.tokens + self.trees

    def annotations(self):
        """
        Return all annotations. Note that some of the annotations are trees,
        and may themselves contain annotations.

        A crude way to search these would be to use `Tree.subtrees`, but you
        may find it more efficient to do a first pass with `Tree.topdown`
        function.
        """
        return self.tokens + self.trees

class CoreNlpToken(postag.Token):
    """
    A single token and its POS tag. Other information is stored in `features`
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
