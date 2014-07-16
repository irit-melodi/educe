# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: CeCILL-B (French BSD3)

# pylint: disable=R0904

"""
Tests for educe.external
"""

import unittest

from educe.annotation import Span
from .postag import generic_token_spans


class PosTag(unittest.TestCase):
    """Working with part of speech taggers"""

    def test_simple_align(self):
        "trivial token realignment"

        tokens = ["a", "bb", "ccc"]
        text = "a bb    ccc"
        spans = generic_token_spans(text, tokens)
        expected = [Span(0, 1),
                    Span(2, 4),
                    Span(8, 11)]
        self.assertEquals(expected, spans)

    def test_messy_align(self):
        "ignore whitespace in token"

        tokens = ["a", "b b", "c c c"]
        text = "a bb    ccc"
        spans = generic_token_spans(text, tokens)
        expected = [Span(0, 1),
                    Span(2, 4),
                    Span(8, 11)]
        self.assertEquals(expected, spans)
