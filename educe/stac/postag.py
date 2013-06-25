# Author: Eric Kow
# License: BSD3

"""
STAC conventions for running a pos tagger, saving the results,
and reading them.
"""

import codecs
import copy
import os.path
import subprocess
import sys

from educe import stac, util
from educe.annotation import Span
import educe.external.postag as ext

def sorted_by_span(xs):
    """
    Annotations sorted by text span
    """
    return sorted(xs, key=lambda x:x.span)

def is_turn(x):
    return x.type == 'Turn'

def tagger_file_name(k, dir):
    """
    Given an educe.corpus.FileId and directory, return the file path
    within that directory that corresponds to the tagger output
    """
    k2 = copy.copy(k)
    k2.stage     = 'pos-tagged'
    k2.annotator = 'ark-tweet-nlp'
    return os.path.join(dir, stac.id_to_path(k2) + '.conll')

def extract_turns(doc):
    """
    Return a string representation of the document's turn text
    for use by a tagger
    """
    turns = sorted_by_span(filter(is_turn, doc.units))
    def ttext(turn):
        return stac.split_turn_text(doc.text_for(turn))[1]
    return "\n".join(map(ttext, turns))

def tagger_cmd(tagger_jar, txt_file):
    return [ 'java'
           , '-XX:ParallelGCThreads=2'
           , '-Xmx500m'
           , '-jar', tagger_jar
           , '--input-format', 'text'
           , '--output-format', 'conll'
           , txt_file
           ]
 
def run_tagger(corpus, outdir, tagger_jar):
    """
    Run the ark-tweet-tagger on all the (unannotated) documents in
    the corpus and save the results in the specified directory
    """
    for k in corpus:
        doc   = corpus[k]

        k_txt           = copy.copy(k)
        k_txt.stage     = 'turns'
        k_txt.annotator = None

        root  = stac.id_to_path(k_txt)
        txt_file = os.path.join(outdir, 'tmp', root + '.txt')
        txt_dir  = os.path.split(txt_file)[0]
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        with codecs.open(txt_file, 'w', 'utf-8') as f:
            print >> f, extract_turns(doc)

        tagged_file = tagger_file_name(k, outdir)
        tagged_dir  = os.path.split(tagged_file)[0]
        if not os.path.exists(tagged_dir):
            os.makedirs(tagged_dir)
        # from the runTagger srcipt
        cmd = tagger_cmd(tagger_jar, txt_file)
        with open(tagged_file, 'wb') as tf:
            subprocess.call(cmd, stdout=tf)

def read_tags(corpus, dir):
    """
    Read stored POS tagger output from a directory, and convert them to
    educe.annotation.Standoff objects.

    Return a dictionary mapping 'FileId's to sets of tokens.
    """
    pos_tags = {}
    for k in corpus:
        doc   = corpus[k]
        turns = sorted_by_span(filter(is_turn, doc.units))

        tagged_file = tagger_file_name(k, dir)
        raw_toks    = ext.read_token_file(tagged_file)
        pos_tags[k] = []
        for turn, seg in zip(turns, raw_toks):
            prefix, body = stac.split_turn_text(doc.text_for(turn))
            start        = turn.span.char_start + len(prefix)
            toks = ext.token_spans(body, seg, start)
            for t in toks:
                t.origin = doc
                dtxt = doc.text_for(t)
                assert dtxt == t.word
            pos_tags[k].extend(toks)
    return pos_tags
