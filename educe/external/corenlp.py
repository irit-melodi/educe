# Author: Eric Kow
# License: BSD3

"""
Annotations from the CoreNLP pipeline
"""

from __future__ import print_function

import codecs
import copy
import os
import subprocess

from educe.annotation import Span, Standoff
from educe.external import postag


class CoreNlpDocument(Standoff):
    """
    All of the CoreNLP annotations for a particular document as instances of
    `educe.annotation.Standoff` or as structures that contain such instances.
    """
    def __init__(self, tokens, trees, deptrees, chains):
        Standoff.__init__(self, None)

        self.tokens = tokens
        "list of `CoreNlpToken`"

        self.trees = trees
        "constituency trees"

        self.deptrees = deptrees
        "dependency trees"

        self.chains = chains
        "coreference chains"

    def _members(self):
        return self.tokens + self.trees


class CoreNlpToken(postag.Token):
    """
    A single token and its POS tag.

    Attributes
    ----------
    features : dict from str to str
        Additional info found by corenlp about the token
        (eg. `x.features['lemma']`)
    """
    def __init__(self, t, offset, origin=None):
        """
        Parameters
        ----------
        t : dict
            Token from corenlp's XML output.
        offset : int
            Offset from the span of the corenlp token to the document.
        origin : FileId, optional
            Identifier for the document.
        """
        extent = t['extent']
        word = t['word']
        tag = t['POS']
        span = Span(extent[0], extent[1] + 1).shift(offset)
        postag.Token.__init__(self, postag.RawToken(word, tag), span)
        self.features = copy.copy(t)
        for k in ['s_id', 'word', 'extent', 'POS']:
            del self.features[k]

    def __str__(self):
        return postag.Token.__str__(self) + ' ' + str(self.features)

    # for debugging
    def __repr__(self):
        return self.word


class CoreNlpWrapper(object):
    """Wrapper for the CoreNLP parsing system."""

    def __init__(self, corenlp_dir):
        """Setup common attributes"""
        self.cwd = corenlp_dir
        # CoreNLP's classpath (string version)
        jars = [x for x in os.listdir(corenlp_dir)
                if os.path.splitext(x)[1] == '.jar']
        cp_sep = ':' if os.name != 'nt' else ';'
        self.cp_str = cp_sep.join(jars)

    def process(self, txt_files, outdir, properties=[]):
        """Run CoreNLP on text files

        Parameters
        ----------
        txt_files: list of strings
            Input files

        outdir: string
            Output dir

        properties: list of strings, optional
            Properties to control the behaviour of CoreNLP

        Returns
        -------
        corenlp_outdir: string
            Directory containing CoreNLP's output files
        """
        # local tmp dir for CoreNLP's manifesto and properties
        tmp_outdir = os.path.join(outdir, 'tmp')
        if not os.path.exists(tmp_outdir):
            os.makedirs(tmp_outdir)

        # manifest tells CoreNLP what files to read as input
        manifest_file = os.path.join(tmp_outdir, 'manifest')
        with codecs.open(manifest_file, 'w', 'utf-8') as f:
            print('\n'.join(txt_files), file=f)

        # java properties to control behaviour of CoreNLP
        props_file = os.path.join(tmp_outdir, 'corenlp.properties')
        with codecs.open(props_file, 'w', 'utf-8') as f:
            print('\n'.join(properties), file=f)

        # output dir
        corenlp_outdir = os.path.join(outdir, 'corenlp')
        if not os.path.exists(corenlp_outdir):
            os.makedirs(corenlp_outdir)

        # run CoreNLP
        cmd = ['java',
               '-cp', self.cp_str,
               '-Xmx3g',
               'edu.stanford.nlp.pipeline.StanfordCoreNLP',
               '-filelist', manifest_file,
               '-props', props_file,
               '-outputDirectory', corenlp_outdir,
              ]
        subprocess.call(cmd, cwd=self.cwd)

        return corenlp_outdir
