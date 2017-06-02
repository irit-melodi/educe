#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Eric Kow
# License: BSD3

"""
CONLL_ formatted POS tagger output into educe
standoff annotations
(at least as emitted by CMU's ark-tweet-nlp_.

Files are assumed to be UTF-8 encoded.

Note: NLTK has a CONLL reader too which looks a lot
more general than this one


.. _CONLL:         http://ifarm.nl/signll/conll/
.. _ark-tweet-nlp: http://www.ark.cs.cmu.edu/TweetNLP/
"""

from itertools import islice
import codecs

from educe.annotation import Span, Standoff
from educe.internalutil import ifilterfalse

# I don't yet see how "too few public methods" is helpful
# pylint: disable=R0903


class EducePosTagException(Exception):
    """
    Exceptions that arise during POS tagging or when reading
    POS tag resources
    """
    def __init__(self, *args, **kw):
        super(EducePosTagException, self).__init__(*args, **kw)

# ---------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------


class RawToken(object):
    """
    A token with a part of speech tag associated with it
    """
    def __init__(self, word, tag):
        self.word = word
        self.tag = tag

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

    # left padding Token
    _lpad_word = '__START__'
    _lpad_tag = '__START__'
    _lpad_span = Span(0, 0)

    @classmethod
    def left_padding(cls):
        "Return a special Token for left padding"
        return Token(RawToken(cls._lpad_word, cls._lpad_tag), cls._lpad_span)


def read_token_file(fname):
    """
    Return a list of lists of RawToken

    The input file format is what I believe to be the CONLL format
    (at least as emitted by the CMU Twitter POS tagger)
    """
    segment = []
    segments = []
    with codecs.open(fname, 'r', 'utf-8') as stream:
        for line in stream:
            spl = line.split()
            if len(spl) == 2 or len(spl) == 3:
                segment.append(RawToken(spl[0], spl[1]))
            elif len(spl) == 0:
                segments.append(segment)
                segment = []
            else:
                raise EducePosTagException("Did not understand this"
                                           "line in tokens file: " + line)
        return segments

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def generic_token_spans(text, tokens, offset=0, txtfn=None):
    """
    Given a string and a sequence of substrings within than string,
    infer a span for each of the substrings.

    We do this spans by walking the text and the tokens we consume
    substrings and skipping over any whitespace (including that
    which is within the tokens). For this to work, the substring
    sequence must be identical to the text modulo whitespace.

    Spans are relative to the start of the string itself, but can be
    shifted by passing an offset (the start of the original string's
    span). Empty tokens are accepted but have a zero-length span.

    Note: this function is lazy so you can use it incrementally
    provided you can generate the tokens lazily too

    You probably want `token_spans` instead; this function is meant
    to be used for similar tasks outside of pos tagging

    :param txtfn: function to extract text from a token (default None,
                  treated as identity function)
    """
    txt_iter = ifilterfalse(lambda x: x[1].isspace(),
                            enumerate(text))
    txtfn = txtfn or (lambda x: x)
    last = offset  # for corner case of empty tokens
    for token in tokens:
        tok_chars = list(ifilterfalse(lambda x: x.isspace(),
                                      txtfn(token)))
        if not tok_chars:
            yield Span(last, last)
            continue
        prefix = list(islice(txt_iter, len(tok_chars)))
        if not prefix:
            msg = "Too many tokens (current: %s)" % txtfn(token)
            raise EducePosTagException(msg)
        last = prefix[-1][0] + 1 + offset
        span = Span(prefix[0][0] + offset, last)
        pretty_prefix = text[span.char_start:span.char_end]
        # check the text prefix to make sure we have the same
        # non-whitespace characters
        for txt_pair, tok_char in zip(prefix, tok_chars):
            idx, txt_char = txt_pair
            if txt_char != tok_char:
                msg = "token mismatch at char %d (%s vs %s)\n"\
                    % (idx, txt_char, tok_char)\
                    + " token: [%s]\n" % token\
                    + " text:  [%s]" % pretty_prefix
                raise EducePosTagException(msg)
        yield span


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
    span).

    Parameters
    ----------
    text : str
        Base text.
    tokens : sequence of RawToken
        Sequence of raw tokens in the text.
    offset : int, defaults to 0
        Offset for spans.

    Returns
    -------
    res : list of Token
        Sequence of proper educe Tokens with their span.
    """
    token_words = [tok.word for tok in tokens]
    spans = generic_token_spans(text, token_words, offset)
    res = [Token(tok, span) for tok, span in zip(tokens, spans)]

    # sanity checks that should be moved to tests
    for orig_tok, new_tok in zip(tokens, res):
        span = Span(new_tok.span.char_start - offset,
                    new_tok.span.char_end - offset)
        snippet = text[span.char_start:span.char_end]
        assert snippet == new_tok.word
        assert orig_tok.word == new_tok.word
        assert orig_tok.tag == new_tok.tag
    return res
