# Author: Eric Kow
# License: BSD3

"""
Corpus management (re-exported by educe.rst_dt)
"""

from glob import glob
import os
import sys

from educe.corpus import FileId
from educe.rst_dt import parse
import educe.corpus

# ---------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------


class Reader(educe.corpus.Reader):
    """
    See `educe.corpus.Reader` for details
    """
    def __init__(self, corpusdir):
        educe.corpus.Reader.__init__(self, corpusdir)

    def files(self):
        anno_files = {}
        full_glob = os.path.join(self.rootdir, '*.dis')

        for fname in glob(full_glob):
            text_file = os.path.splitext(fname)[0]
            bname = os.path.basename(fname)
            doc = os.path.splitext(bname)[0]
            k = FileId(doc=doc,
                       subdoc=None,
                       stage='discourse',
                       annotator='None')
            anno_files[k] = (fname, text_file)
        return anno_files

    def slurp_subcorpus(self, cfiles, verbose=False):
        """
        See `educe.rst_dt.parse` for a description of `RSTTree`
        """
        corpus = {}
        counter = 0
        for k in cfiles.keys():
            if verbose:
                sys.stderr.write("\rSlurping corpus dir [%d/%d]" %
                                 (counter, len(cfiles)))
            annotations = parse.read_annotation_file(*cfiles[k])
            annotations.set_origin(k)
            corpus[k] = annotations
            counter = counter+1
        if verbose:
            sys.stderr.write("\rSlurping corpus dir [%d/%d done]\n" %
                             (counter, len(cfiles)))
        return corpus


def id_to_path(k):
    """
    Given a fleshed out FileId (none of the fields are None),
    return a filepath for it following RST Discourse Treebank
    conventions.

    You will likely want to add your own filename extensions to
    this path
    """
    return k.doc
