# Author: Eric Kow
# License: BSD3

"""
Corpus management (re-exported by educe.rst_dt)
"""

import os
import sys
from glob import glob
from os.path import dirname
from os.path import join

from nltk import Tree

from educe.corpus import FileId
from educe.internalutil import treenode
from educe.rst_dt import parse
import educe.util
import educe.corpus
from .document_plus import DocumentPlus
from .annotation import SimpleRSTTree
from .deptree import RstDepTree


RELMAP_112_18_FILE = join(dirname(__file__), 'rst_112to18.txt')


# ---------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------
class Reader(educe.corpus.Reader):
    """
    See `educe.corpus.Reader` for details
    """
    def __init__(self, corpusdir):
        educe.corpus.Reader.__init__(self, corpusdir)

    def files(self, exclude_file_docs=False):
        """
        Parameters
        ----------
        exclude_file_docs : boolean, optional (default=False)
            If True, fileX documents are ignored. The figures reported by
            (Li et al., 2014) on the RST-DT corpus indicate they exclude
            fileN files, whereas Joty seems to include them.
            fileN documents are more damaged than wsj_XX documents, e.g.
            text mismatches with the corresponding document in the PTB.
        """
        anno_files = {}
        dis_glob = 'wsj_*.dis' if exclude_file_docs else '*.dis'
        full_glob = os.path.join(self.rootdir, dis_glob)

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

    def __init__(self, corpus_dir, args, coarse_rels=False,
                 exclude_file_docs=False):
        """
        TODO

        Parameters
        ----------
        ...

        exclude_file_docs: boolean, default False
            If True, ignore fileX files.
        """
        # TODO: kill `args`
        self.reader = Reader(corpus_dir)
        # pre-load corpus
        anno_files_unfltd = self.reader.files(exclude_file_docs)
        is_interesting = educe.util.mk_is_interesting(args)
        anno_files = self.reader.filter(anno_files_unfltd, is_interesting)
        self.corpus = self.reader.slurp(anno_files, verbose=True)
        # setup label converter for the desired granularity
        # 'fine' means we don't change anything
        if coarse_rels:
            relmap_file = RELMAP_112_18_FILE
            self.rel_conv = RstRelationConverter(relmap_file).convert_tree
        else:
            self.rel_conv = None

    def decode(self, doc_key):
        """Decode a document from the RST-DT (gold)"""
        # create a DocumentPlus
        # grouping is the document name
        grouping = os.path.basename(id_to_path(doc_key))
        # the RST tree is currently pivotal to get all the layers of info,
        # including the RSTContext that contains the document text and
        # structure (paragraphs + poorly segmented sentences)
        orig_rsttree = self.corpus[doc_key]
        rst_context = treenode(orig_rsttree).context
        # finally...
        doc = DocumentPlus(doc_key, grouping, rst_context)

        # TODO get EDUs here rather than below (see dep tree)
        # edus = orig_rsttree.leaves()
        # doc.edus.extend(edus)

        # attach original RST tree
        # convert relation labels if needed
        if self.rel_conv is not None:
            orig_rsttree = self.rel_conv(orig_rsttree)
        doc.orig_rsttree = orig_rsttree

        # convert to binary tree
        rsttree = SimpleRSTTree.from_rst_tree(orig_rsttree)
        # NEW incorporate nuclearity into label
        # TODO add a parameter (in init or this function) to trigger this
        if False:
            rsttree = SimpleRSTTree.incorporate_nuclearity_into_label(rsttree)
        doc.rsttree = rsttree

        # convert to dep tree
        deptree = RstDepTree.from_simple_rst_tree(rsttree)
        doc.deptree = deptree

        # get EDUs (bad)
        # TODO: get EDUs from orig_rsttree.leaves() and let
        # document_plus do the left padding
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


class RstRelationConverter(object):
    """Converter for RST relations (labels)

    Known to work on RstTree, possibly SimpleRstTree (untested).
    """

    def __init__(self, relmap_file):
        """relmap_file is a path to a file containing the mapping"""
        self.relmap = self._read_relmap(relmap_file)

    def _read_relmap(self, relmap_file):
        """read the relmap from file"""
        relmap = dict()
        with open(relmap_file) as f:
            for line in f:
                old_rel, new_rel = line.strip().split()
                relmap[old_rel] = new_rel
        return relmap

    def convert_label(self, label):
        """Convert a label following the mapping, lowercased otherwise"""
        return self.relmap.get(label.lower(), label.lower())

    def convert_tree(self, rst_tree):
        """Change relation labels in rst_tree using the mapping"""
        conv_lbl = self.convert_label
        for pos in rst_tree.treepositions():
            t = rst_tree[pos]
            if isinstance(t, Tree):
                node = treenode(t)
                # replace old rel with new rel
                node.rel = conv_lbl(node.rel)
        return rst_tree
