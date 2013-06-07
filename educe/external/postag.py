#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
CONLL_ formatted POS tagger output into educe
standoff annotations
(at least as emitted by CMU's ark-tweet-nlp.

Files are assumed to be UTF-8 encoded.

Note: NLTK has a CONLL reader too which looks a lot
more general than this one


.. CONLL:         http://ifarm.nl/signll/conll/
.. ark-tweet-nlp: http://www.ark.cs.cmu.edu/TweetNLP/
"""

import codecs
import copy
import itertools

from educe.annotation import Span, Standoff

# ---------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------

class RawToken:
    """
    A token with a part of speech tag associated with it
    """
    def __init__(self, word, tag):
        self.word = word
        self.tag  = tag

    def __str__(self):
        return self.word + "/" + self.tag

    def __unicode__(self):
        return self.word + "/" + self.tag

class Token(RawToken, Standoff):
    """
    A token with a part of speech tag and some character offsets
    associated with it.
    """
    def __init__(self, tok, span):
        RawToken.__init__(self, tok.word, tok.tag)
        Standoff.__init__(self)
        self.span = span

    def __str__(self):
        return '%s\t%s' % (RawToken.__str__(self), self.span)

    def __unicode__(self):
        return '%s\t%s' % (RawToken.__unicode__(self), self.span)


def read_token_file(fname):
    """
    Return a list of lists of RawToken

    The input file format is what I believe to be the CONLL format
    (at least as emitted by the CMU Twitter POS tagger)
    """
    segment  = []
    segments = []
    with codecs.open(fname, 'r', 'utf-8') as f:
        for l in f:
            spl = l.split()
            if len(spl) == 2 or len(spl) == 3:
                segment.append(RawToken(spl[0],spl[1]))
            elif len(spl) == 0:
                segments.append(segment)
                segment = []
            else:
                raise Exception("Did not understand this line in tokens file: " + l)
        return segments

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def token_spans(text, tokens, offset=0):
    """
    Given a string and a sequence of RawToken representing tokens
    in that string, infer the span for each token.  Return the
    results as a sequence of Token objects.

    We infer these spans by walking the text as we consume tokens,
    and skipping over any whitespace in between. For this to work,
    the raw token text must be identical to the text modulo whitespace.

    Spans are relative to the start of the string itself, but can be
    shifted by passing an offset (the start of the original string's
    span)
    """
    def next_token():
        if len(tokens) == 0:
            msg  = "Ran out of tokens but still have %d characters" % len(text)
            msg += "of text in [%s...] left" % text[:8]
            raise Exception(msg)
        else:
            return tokens.pop(0)

    orig_tokens = copy.copy(tokens)
    orig_text   = text
    res   = []
    left  = 0
    right = 0
    tok   = None
    while (len(text.lstrip()) > 0):
        if tok is None:
            tok = next_token()
        if text.startswith(tok.word):
            right = left + len(tok.word)
            pair  = Token(tok, Span(left + offset, right + offset))
            res.append(pair)
            left  = right
            text  = text[len(tok.word):]
            tok   = None
        elif text[0].isspace():
            # next token
            prefix = list(itertools.takewhile(lambda x:x.isspace(), text))
            left  += len(prefix)
            right  = left
            text   = text[len(prefix):]
        else:
            snippet = text[0:len(tok.word) + 3]
            msg =  "Was expecting [%s] to be the next token " % tok.word
            msg += "in the text, but got [%s...] instead." % snippet
            raise Exception(msg)

    if len(tokens) == 0:
        # sanity checks that should be moved to tests
        for orig_tok, new_tok in zip(orig_tokens, res):
            sp = Span(new_tok.span.char_start - offset,
                      new_tok.span.char_end   - offset)
            snippet = orig_text[sp.char_start : sp.char_end]
            assert snippet  == new_tok.word
            assert orig_tok.word == new_tok.word
            assert orig_tok.tag  == new_tok.tag
        return res
    else:
        msg =  "Still have %d tokens left " % len(tokens)
        msg += "[%s...] after consuming text" % tokens[:3]
        raise Exception(msg)
