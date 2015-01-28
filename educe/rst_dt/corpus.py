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
import educe.util
import educe.corpus
from .document_plus import DocumentPlus
from .annotation import SimpleRSTTree
from .deptree import RstDepTree


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
            k = mk_key(doc)
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
    return a filepath for it following RST Discourse Treebank
    conventions.

    You will likely want to add your own filename extensions to
    this path
    """
    return k.doc


class RstDtParser(object):
    """Fake parser that gets annotation from the RST-DT.
    """

    def __init__(self, corpus_dir, args):
        # TODO: kill `args`
        self.reader = Reader(corpus_dir)
        # pre-load corpus
        is_interesting = educe.util.mk_is_interesting(args)
        anno_files = self.reader.filter(self.reader.files(), is_interesting)
        self.corpus = self.reader.slurp(anno_files, verbose=True)

    def decode(self, doc_key):
        """Decode a document from the RST-DT (gold)"""
        grouping = os.path.basename(id_to_path(doc_key))
        doc = DocumentPlus(doc_key, grouping)

        doc.orig_rsttree = self.corpus[doc_key]
        # get EDUs
        doc.edus.append(doc.orig_rsttree.leaves())
        # convert to binary tree
        doc.rsttree = SimpleRSTTree.from_rst_tree(doc.orig_rsttree)
        # convert to dep tree
        doc.deptree = RstDepTree.from_simple_rst_tree(doc.rsttree)
        # get EDUs
        # TODO: get EDUs from orig_rsttree.leaves(),
        # let document_plus do the left padding
        doc.edus = doc.deptree.edus
        return doc

    def segment(self, doc):
        """Segment the document into EDUs using the RST-DT (gold).
        """
        return doc

    def parse(self, doc):
        """Parse the document using the RST-DT (gold).
        """
        return doc
