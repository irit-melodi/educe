# Author: Eric Kow
# License: BSD3

"""
STAC conventions for running a pos tagger, saving the results,
and reading them.
"""

from __future__ import print_function
import codecs
import copy
import os.path
import subprocess

from educe import stac
import educe.external.postag as ext


def sorted_by_span(annos):
    """
    Annotations sorted by text span
    """
    return sorted(annos, key=lambda x: x.span)


def tagger_file_name(doc_key, root):
    """Get the file path to the output of the POS tagger for a document.

    The returned file path is a .conll file within the given directory.

    Parameters
    ----------
    doc_key : educe.corpus.FileId
        FileId of the document

    root : string
        Path to the folder containing annotations for this corpus,
        including the output of the POS tagger.

    Returns
    -------
    res : string
        Path to the .conll file output by the POS tagger.
    """
    doc_key2 = copy.copy(doc_key)
    doc_key2.stage = 'pos-tagged'
    doc_key2.annotator = 'ark-tweet-nlp'
    return os.path.join(root, stac.id_to_path(doc_key2) + '.conll')


def extract_turns(doc):
    """
    Return a string representation of the document's turn text
    for use by a tagger
    """
    turns = sorted_by_span(x for x in doc.units if stac.is_turn(x))

    def ttext(turn):
        """Get the turn text"""
        return stac.split_turn_text(doc.text(turn.text_span()))[1]

    return "\n".join(ttext(x) for x in turns)


def tagger_cmd(tagger_jar, txt_file):
    """Command to run the POS tagger"""
    return [
        'java',
        '-XX:ParallelGCThreads=2',
        '-Xmx500m',
        '-jar', tagger_jar,
        '--input-format', 'text',
        '--output-format', 'conll',
        txt_file
    ]


def run_tagger(corpus, outdir, tagger_jar):
    """
    Run the ark-tweet-tagger on all the (unannotated) documents in
    the corpus and save the results in the specified directory
    """
    for k in corpus:
        doc = corpus[k]

        k_txt = copy.copy(k)
        k_txt.stage = 'turns'
        k_txt.annotator = None

        root = stac.id_to_path(k_txt)
        txt_file = os.path.join(outdir, 'tmp', root + '.txt')
        txt_dir = os.path.split(txt_file)[0]
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        with codecs.open(txt_file, 'w', 'utf-8') as f:
            print(extract_turns(doc), file=f)

        tagged_file = tagger_file_name(k, outdir)
        tagged_dir = os.path.split(tagged_file)[0]
        if not os.path.exists(tagged_dir):
            os.makedirs(tagged_dir)
        # from the runTagger srcipt
        cmd = tagger_cmd(tagger_jar, txt_file)
        with open(tagged_file, 'wb') as tf:
            subprocess.call(cmd, stdout=tf)


def read_tags(corpus, root_dir):
    """
    Read stored POS tagger output from a directory, and convert them to
    educe.annotation.Standoff objects.

    Return a dictionary mapping 'FileId's to sets of tokens.

    Parameters
    ----------
    corpus : dict(FileId, GlozzDocument)
        Dictionary of documents keyed by their FileId.
    root_dir : str
        Path to the directory containing the output of the POS tagger,
        one file per document.

    Returns
    -------
    pos_tags : dict(FileId, list(Token))
        Map from each document id to the list of tokens predicted by a
        POS tagger.
    """
    pos_tags = {}
    for k in corpus:
        doc = corpus[k]
        turns = sorted_by_span(x for x in doc.units if stac.is_turn(x))

        tagged_file = tagger_file_name(k, root_dir)
        raw_toks = ext.read_token_file(tagged_file)
        pos_tags[k] = []
        for turn, seg in zip(turns, raw_toks):
            prefix, body = stac.split_turn_text(doc.text(turn.text_span()))
            start = turn.span.char_start + len(prefix)
            toks = ext.token_spans(body, seg, start)
            for t in toks:
                t.origin = doc
                dtxt = doc.text(t.text_span())
                assert dtxt == t.word
            pos_tags[k].extend(toks)
    return pos_tags
