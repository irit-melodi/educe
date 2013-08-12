#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
Coreference chain output in the form of educe standoff annotations
(at least as emitted by Stanford's CoreNLP_ pipeline)

A coreference chain is considered to be a set of mentions.
Each mention contains a set of tokens.

.. CoreNLP:       http://nlp.stanford.edu/software/corenlp.shtml
"""

from   educe.annotation import Span, Standoff

class Chain(Standoff):
    def __init__(self, mentions):
        Standoff.__init__(self)
        self.mentions = mentions

    def _members(self):
        return self.mentions

class Mention(Standoff):
    def __init__(self, tokens, head, most_representative=False):
        Standoff.__init__(self)
        self.tokens = tokens
        self.head   = head
        self.most_representative = most_representative

    def _members(self):
        return self.tokens
