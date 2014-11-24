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


def parse_text(text):
    """
    Return a sequence of Paragraph annotations from an RST text.
    By convention:

    * paragraphs are separated by double newlines
    * sentences by single newlines

    Note that this segmentation isn't particularly reliable, and
    seems to cut at some abbreviations, like "Prof.". It shouldn't
    be taken too seriously, but if you need some sort of rough
    approximation, it may be helpful.
    """
    start = 0
    sent_id = 0
    output_paras = []
    for para_id, paragraph in enumerate(text.split("\n\n")):
        output_sentences = []
        for sentence in paragraph.split("\n"):
            end = start + len(sentence)
            if end > start:
                output_sentences.append(Sentence(sent_id, Span(start, end)))
                sent_id += 1
            start = end + 1  # newline
        output_paras.append(Paragraph(para_id, output_sentences))
        start += 1  # second newline
    return output_paras
