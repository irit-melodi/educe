"""This module provides loaders for file formats found in the RST-WSJ-corpus.
"""

from __future__ import print_function

import codecs
import os

from educe.annotation import Span
from educe.rst_dt.text import Paragraph, Sentence


def _load_rst_wsj_corpus_edus_file(f):
    """Actually do load"""
    txt = []
    offset = []
    for line in f:
        # line pattern: '    <EDU>\n' (4 spaces, EDU text, newline)
        edu_txt = line.strip()
        txt.append(edu_txt)
        offset.append(len(edu_txt))
    # FIXME ' '.join() can introduce whitespaces not in the original text
    # at least partly due to bad EDU segmentation
    clean_txt = ' '.join(txt)
    return clean_txt, offset


def load_rst_wsj_corpus_edus_file(f):
    """Load a file that contains the EDUs of a document.

    Return clean text and the list of EDU offsets on the clean text.
    """
    with codecs.open(f, 'r', 'utf-8') as f:
        clean_txt, offset = _load_rst_wsj_corpus_edus_file(f)
    return (clean_txt, offset)


# TODO what the _text_ functions in this module should return is
# clean text plus the extractible annotation,
# i.e. paragraph spans (if any) and sentence spans,
# *on the clean text*

##############
# file## files
##############

# TODO: maybe don't generate Paragraph, as there is no paragraph marking
FIL_SEP_PARA = '\n\n'  # probably useless
FIL_SEP_SENT = '\n  '


def _load_rst_wsj_corpus_text_file_file(f):
    """Actually do load"""
    text = f.read()

    start = 0
    sent_id = 0
    output_paras = []
    for para_id, paragraph in enumerate(text.split(FIL_SEP_PARA)):
        output_sentences = []
        for sentence in paragraph.split(FIL_SEP_SENT):
            end = start + len(sentence)
            # NEW: remove leading white spaces
            lws = len(sentence) - len(sentence.lstrip())
            if lws:
                start += lws
            # end NEW
            if end > start:
                output_sentences.append(Sentence(sent_id, Span(start, end)))
                sent_id += 1
            start = end + 3  # + 3 for + len(FIL_SEP_SENT)
        output_paras.append(Paragraph(para_id, output_sentences))
        start -= 1  # start += len(FIL_SEP_PARA) - len(FIL_SEP_SENT)
    # TODO remove trailing '\n' of last sentence

    return text, output_paras


def load_rst_wsj_corpus_text_file_file(f):
    """Load a text file whose name is of the form `file##`

    These files do not mark paragraphs.
    Each line contains a sentence preceded by two or three leading spaces.
    """
    with codecs.open(f, 'r', 'utf-8') as f:
        text, paragraphs = _load_rst_wsj_corpus_text_file_file(f)
    return (text, paragraphs)


##################
# wsj_## files
##################
WSJ_SEP_PARA = ' \n\n'  # was: '\n\n'
WSJ_SEP_SENT = '\n'


def _load_rst_wsj_corpus_text_file_wsj(f):
    """Actually do load"""
    text = f.read()

    start = 0
    sent_id = 0
    output_paras = []
    for para_id, paragraph in enumerate(text.split(WSJ_SEP_PARA)):
        output_sentences = []
        for sentence in paragraph.split(WSJ_SEP_SENT):
            end = start + len(sentence)
            # NEW: remove trailing white space
            rws = len(sentence) - len(sentence.rstrip())
            if rws:
                end -= rws
            # end NEW
            if end > start:
                output_sentences.append(Sentence(sent_id, Span(start, end)))
                sent_id += 1
            start = end + rws + 1  # + 1 for + len(WSJ_SEP_SENT)
        output_paras.append(Paragraph(para_id, output_sentences))
        start += 2  # whitespace and second newline

    return text, output_paras


def load_rst_wsj_corpus_text_file_wsj(f):
    """Load a text file whose name is of the form `wsj_##`

    By convention:

    * paragraphs are separated by double newlines
    * sentences by single newlines

    Note that this segmentation isn't particularly reliable, and
    seems to both over- (e.g. cut at some abbreviations, like "Prof.")
    and under-segment (e.g. not separate contiguous sentences).
    It shouldn't be taken too seriously, but if you need some sort of
    rough approximation, it may be helpful.
    """
    with codecs.open(f, 'r', 'utf-8') as f:
        text, paragraphs = _load_rst_wsj_corpus_text_file_wsj(f)
    return (text, paragraphs)


def load_rst_wsj_corpus_text_file(f):
    """Load a text file from the RST-WSJ-CORPUS.

    Return a sequence of Paragraph annotations from an RST text.

    The corpus contains two types of text files, so this function is
    mainly an entry point that delegates to the appropriate function.
    """
    bn = os.path.basename(f)
    bn_prefix = bn[:4]
    if bn_prefix == 'file':
        return load_rst_wsj_corpus_text_file_file(f)
    elif bn_prefix == 'wsj_':
        return load_rst_wsj_corpus_text_file_wsj(f)
    else:
        # raise ValueError(err_msg.format(f))
        err_msg = 'W: using wsj_ loader for file of unknown type: {}'
        print(err_msg.format(f))
        return load_rst_wsj_corpus_text_file_wsj(f)
