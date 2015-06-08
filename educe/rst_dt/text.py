#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: CeCILL-B (BSD-3 like)

# disable "pointless string" warning because we want attribute docstrings
# pylint: disable=W0105

# too few public methods, don't yet see why I should care
# pylint: disable=R0903

"""
Educe-style annotations for RST discourse treebank text
objects (paragraphs and sentences)
"""

import re

from educe.annotation import Standoff, Span


class Paragraph(Standoff):
    """
    A paragraph is a sequence of `Sentence`s (also standoff
    annotations).
    """
    def __init__(self, num, sentences):
        self.sentences = sentences
        "sentence-level annotations"

        self.num = num
        "paragraph ID in document"

        super(Paragraph, self).__init__()

    def _members(self):
        return self.sentences

    # left padding
    _lpad_num = -1

    @classmethod
    def left_padding(cls, sentences):
        """Return a left padding Paragraph"""
        return cls(cls._lpad_num, sentences)


class Sentence(Standoff):
    """
    Just a text span really
    """
    def __init__(self, num, span):
        super(Sentence, self).__init__()
        self.span = span

        self.num = num
        "sentence ID in document"

    def text_span(self):
        return self.span

    # left padding
    _lpad_num = -1
    _lpad_span = Span(0, 0)

    @classmethod
    def left_padding(cls):
        """Return a left padding Sentence"""
        return cls(cls._lpad_num, cls._lpad_span)


# helper function, formerly in learning.features
def clean_edu_text(text):
    """
    Strip metadata from EDU text and compress extraneous whitespace
    """
    clean_text = text
    clean_text = re.sub(r'(\.|,)*$', r'', clean_text)
    clean_text = re.sub(r'^"', r'', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text
