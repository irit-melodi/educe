# Author: Eric Kow
# License: BSD3

"""
PDTB Corpus management (re-exported by educe.pdtb)
"""

from glob import glob
import os
import sys

from educe.corpus import FileId
import educe.corpus
from . import parse


# ---------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------
class Reader(educe.corpus.Reader):
    """
    See `educe.corpus.Reader` for details
    """
    def __init__(self, corpusdir):
        educe.corpus.Reader.__init__(self, corpusdir)

    def files(self, doc_glob=None):
        """
        Parameters
        ----------
        doc_glob : str, optional
            Glob expression for document (folder) names ; if `None`, it
            uses the wildcard '*/*' for folder names and file basenames.
        """
        if doc_glob is None:
            doc_glob = '*/*'
        anno_files = {}
        full_glob = os.path.join(
            self.rootdir, '{doc_glob}.pdtb'.format(doc_glob=doc_glob))
        for fname in glob(full_glob):
            bname = os.path.basename(fname)
            doc = os.path.splitext(bname)[0]
            k = mk_key(doc)
            anno_files[k] = fname
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
            fname = cfiles[k]
            annotations = parse.parse(fname)
            # annotations.set_origin(k)
            corpus[k] = annotations
            counter = counter+1
        if verbose:
            sys.stderr.write("\rSlurping corpus dir [%d/%d done]\n" %
                             (counter, len(cfiles)))
        return corpus


def mk_key(doc):
    """
    Return an corpus key for a given document name
    """
    return FileId(doc=doc,
                  subdoc=None,
                  stage='discourse',
                  annotator='unknown')


def id_to_path(k):
    """
    Given a fleshed out FileId (none of the fields are None),
    return a filepath for it following Penn Discourse Treebank
    conventions.

    You will likely want to add your own filename extensions to
    this path
    """
    if k.doc[:4] == 'wsj_':
        prefix = k.doc[4:6]
        return os.path.join(prefix, k.doc)
    else:
        return k.doc
