# Author: Eric Kow
# License: BSD3

"""
STAC conventions for running the Stanford CoreNLP
pipeline, saving the results, and reading them.

The most useful functions here are

  * run_pipeline
  * read_results

"""

from __future__ import print_function
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
from educe.external.corenlp import *
from educe.external.coref   import *
from educe.external.parser  import *
import educe.external.stanford_xml_reader as corenlp_xml

# ---------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------


def turn_id_text(doc):
    """
    Return a list of (turn ids, text) tuples
    in span order (no speaker)
    """
    turns = sorted(filter(stac.is_turn, doc.units),
                   key=lambda k: k.text_span())
    return [(stac.turn_id(turn),
             stac.split_turn_text(doc.text(turn.text_span()))[1])
            for turn in turns]


def run_pipeline(corpus, outdir, corenlp_dir, split=False):
    """
    Run the standard corenlp pipeline on all the (unannotated) documents in
    the corpus and save the results in the specified directory

    If `split=True`, we output one file per turn, an experimental mode
    to account for switching between multiple speakers.  We don't have
    all the infrastructure to read these back in (it should just be a
    matter of some filename manipulation though) and hope to flesh this
    out later.  We also intend to tweak the notion of splitting
    by aggregating consecutive turns with the same speaker, which may somewhat
    mitigate the lost of coreference information.
    """

    # for each document, how many digits do we need to represent the turns
    # in that document; for essentially cosmetic purposes (padding)
    digits = {}
    for d in frozenset([ k.doc for k in corpus ]):
        turns = []
        for k in corpus:
            if k.doc == d:
                turns.extend(list(filter(stac.is_turn, corpus[k].units)))
        turn_ids  = [ int(t.features['Identifier']) for t in turns ]
        digits[d] = max(2,int(math.ceil(math.log10(max(turn_ids)))))

    # dump the turn text
    # TODO: aggregate consecutive turns by same speaker
    txt_files = []
    for k in corpus:
        doc   = corpus[k]

        k_txt           = copy.copy(k)
        k_txt.stage     = 'turns'
        k_txt.annotator = None

        if split:
            for tid, ttext in turn_id_text(doc):
                root  = stac.id_to_path(k_txt) + '_' + tid.zfill(digits[k.doc])

                txt_file = os.path.join(outdir, 'tmp', root + '.txt')
                txt_dir  = os.path.split(txt_file)[0]
                if not os.path.exists(txt_dir):
                    os.makedirs(txt_dir)

                with codecs.open(txt_file, 'w', 'utf-8') as f:
                    print(ttext, file=f)

                txt_files.append(txt_file)
        else:
            root     = stac.id_to_path(k_txt)
            txt_file = os.path.join(outdir, 'tmp', root + '.txt')
            txt_dir  = os.path.split(txt_file)[0]
            if not os.path.exists(txt_dir):
                os.makedirs(txt_dir)
            with codecs.open(txt_file, 'w', 'utf-8') as f:
                for _, ttext in turn_id_text(doc):
                    print(ttext, file=f)
            txt_files.append(txt_file)

    # manifest tells corenlp what to files to read as input
    manifest_dir  = os.path.join(outdir, 'tmp')
    manifest_file = os.path.join(manifest_dir, 'manifest')
    if not os.path.exists(manifest_dir):
        os.makedirs(manifest_dir)
    with codecs.open(manifest_file, 'w', 'utf-8') as f:
        print('\n'.join(txt_files), file=f)

    # java properties to control behaviour of corenlp
    properties = [] if split else ['ssplit.eolonly=true']
    props_file = os.path.join(manifest_dir, 'corenlp.properties')
    with codecs.open(props_file, 'w', 'utf-8') as f:
        print('\n'.join(properties), file=f)

    # run corenlp (will take a while for it to load its various models)
    jars   = [ x for x in os.listdir(corenlp_dir) if os.path.splitext(x)[1] == '.jar' ]
    cp_sep = ':' if os.name != 'nt' else ';'

    corenlp_outdir = os.path.join(outdir, 'corenlp')
    if not os.path.exists(corenlp_outdir):
        os.makedirs(corenlp_outdir)

    cmd = [ 'java'
          , '-cp', cp_sep.join(jars)
          , '-Xmx3g'
          , 'edu.stanford.nlp.pipeline.StanfordCoreNLP'
          , '-filelist',  manifest_file
          , '-props',     props_file
          , '-outputDirectory', corenlp_outdir
          ]
    subprocess.call(cmd, cwd=corenlp_dir)

    # corenlp dumps all the output into one flat directory;
    # move them to the standard STAC layout paths
    for sfile in os.listdir(corenlp_outdir):
        if os.path.splitext(sfile)[1] != '.xml': continue
        k, tid = from_corenlp_output_filename(sfile)
        from_path = os.path.join(corenlp_outdir, sfile)
        to_path   = parsed_file_name(k, outdir)
        to_dir    = os.path.dirname(to_path)
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)
        os.rename(from_path, to_path)

def from_corenlp_output_filename(f):
    """
    Return a tuple of FileId and turn id

    This is entirely by convention we established when calling corenlp
    of course
    """
    prefix = os.path.basename(f)
    prefix = os.path.splitext(prefix)[0]
    prefix = os.path.splitext(prefix)[0]

    parts   = prefix.split('_')
    file_id = corpus.FileId(doc=parts[0],
                            subdoc=parts[1] if len(parts) > 1 else None,
                            stage='unannotated',
                            annotator=None)

    turn_id = parts[-1] if len(parts) == 3 else None
    return file_id, turn_id

# ---------------------------------------------------------------------
# Reading the results
# ---------------------------------------------------------------------

def parsed_file_name(k, dir):
    """
    Given an educe.corpus.FileId and directory, return the file path
    within that directory that corresponds to the corenlp output
    """
    k2 = copy.copy(k)
    k2.stage     = 'parsed'
    k2.annotator = 'stanford-corenlp'
    return os.path.join(dir, stac.id_to_path(k2) + '.xml')

def read_corenlp_result(doc, corenlp_doc, tid=None):
    def is_matching_turn(x):
        if tid is None:
            return stac.is_turn(x)
        else:
            x_tid = x.features['Identifier']
            return stac.is_turn(x) & tid == x_tid

    turns     = sorted(filter(is_matching_turn, doc.units), key=lambda k:k.span)
    sentences = corenlp_doc.get_ordered_sentence_list()

    if len(turns) != len(sentences):
        msg = 'Uh-oh, mismatch between number turns in the corpus (%d) '\
              'and parsed sentences (%d) %s'\
                % (len(turns), len(sentences), doc.origin)
        raise Exception(msg)

    sentence_toks = collections.defaultdict(list)
    for t in corenlp_doc.get_ordered_token_list():
        sid    = t['s_id']
        sentence_toks[sid].append(t)

    # build dict from sid to (dict from tid to fancy token)
    educe_tokens = collections.defaultdict(dict)
    for turn, sent in zip(turns, sentences):
        sid = sent['id']

        # the token offsets are global, ie. for all sentences/turns
        # in the file; so we have to shift them to left to zero them
        # and then shift them back to the right
        sentence_begin = min(t['extent'][0] for t in sentence_toks[sid])

        ttext = doc.text(turn.text_span())
        offset = turn.span.char_start + len(stac.split_turn_text(ttext)[0]) - sentence_begin

        for t in sentence_toks[sid]:
            tid = t['id']
            educe_tokens[sid][tid] = CoreNlpToken(t, offset)

    all_tokens = []
    all_trees  = []
    all_dtrees = []
    for turn, sent in zip(turns, sentences):
        sid         = sent['id']
        tokens_dict = educe_tokens[sid]
        sorted_tokens = [tokens_dict[x] for x in sorted(tokens_dict.keys())]
        tree = nltk.tree.Tree.fromstring(sent['parse'])
        educe_tree = ConstituencyTree.build(tree, sorted_tokens)

        deps   = collections.defaultdict(list)
        for ty, gov_id, dep_id in sent['dependencies']:
            deps[gov_id].append((ty,dep_id))

        educe_dtree = DependencyTree.build(deps, tokens_dict, sid + '-0')

        all_tokens.extend(sorted_tokens)
        all_trees.append(educe_tree)
        all_dtrees.append(educe_dtree)

    all_chains = []
    for ctr,chain in enumerate(corenlp_doc.get_coref_chains()):
        mentions = []
        for m in chain:
            sid         = m['sentence']

            local_id    = lambda x : int(x[len(sid) + 1:])
            global_id   = lambda x : sid + '-' + str(x)

            start       = local_id(m['start'])
            end         = local_id(m['end'])
            token_range = list(map(global_id, range(start, end)))
            tokens      = [ educe_tokens[sid][t] for t in token_range ]
            head        = educe_tokens[sid][m['head']]
            mentions.append(Mention(tokens, head, m['most_representative']))
        all_chains.append(Chain(mentions))

    return CoreNlpDocument(all_tokens, all_trees, all_dtrees, all_chains)

def read_results(corpus, dir):
    """
    Read stored parser output from a directory, and convert them to
    educe.annotation.Standoff objects.

    Return a dictionary mapping 'FileId's to sets of tokens.
    """
    results = {}
    for k in corpus:
        reader = corenlp_xml.Preprocessing_Source()
        reader.read(parsed_file_name(k, dir), suffix='')
        doc        = corpus[k]
        results[k] = read_corenlp_result(doc, reader)
    return results
