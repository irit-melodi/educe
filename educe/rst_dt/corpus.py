# Author: Eric Kow
# License: BSD3

"""
Corpus management (re-exported by educe.rst_dt)
"""

import itertools
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
# temp
from .rst_wsj_corpus import TRAIN_FOLDER, TEST_FOLDER, DOUBLE_FOLDER


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

    def __init__(self, corpus_dir, args, coarse_rels=False):
        # TODO: kill `args`
        self.reader = Reader(corpus_dir)
        # pre-load corpus
        is_interesting = educe.util.mk_is_interesting(args)
        anno_files = self.reader.filter(self.reader.files(), is_interesting)
        self.corpus = self.reader.slurp(anno_files, verbose=True)
        # setup label converter for the desired granularity
        # 'fine' means we don't change anything
        if coarse_rels:
            relmap_file = join(dirname(__file__), 'rst_112to18.txt')
            self.rel_conv = RstRelationConverter(relmap_file).convert_tree
        else:
            self.rel_conv = None

    def decode(self, doc_key):
        """Decode a document from the RST-DT (gold)"""
        # create a DocumentPlus
        # the RST tree is currently pivotal to get all the layers of info
        orig_rsttree = self.corpus[doc_key]
        # convert relation labels if needed
        if self.rel_conv is not None:
            orig_rsttree = self.rel_conv(orig_rsttree)
        # the RST tree is backed by an RSTContext that contains the document
        # text and structure (paragraphs and badly segmented sents)
        rst_context = treenode(orig_rsttree).context
        # grouping is the document name
        grouping = os.path.basename(id_to_path(doc_key))
        # finally...
        doc = DocumentPlus(doc_key, grouping, rst_context)

        # TODO get EDUs here
        # edus = orig_rsttree.leaves()
        # doc.edus.extend(edus)
        # attach original RST tree
        doc.orig_rsttree = orig_rsttree

        # convert to binary tree
        rsttree = SimpleRSTTree.from_rst_tree(orig_rsttree)
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
        relmap = self.relmap
        conv_lbl = self.convert_label
        for pos in rst_tree.treepositions():
            t = rst_tree[pos]
            if isinstance(t, Tree):
                node = treenode(t)
                # replace old rel with new rel
                node.rel = conv_lbl(node.rel)
        return rst_tree


# TODO I think these should belong to rst_wsj_corpus
def _open_plus(doc_key, dis_filename, txt_filename, rel_conv=None,
               syn_parser=None):
    """Create a DocumentPlus filled with all the right information.

    This function is absolutely terrible but provides needed functionality.
    """
    # grouping is the document name
    grouping = os.path.basename(id_to_path(doc_key))
    # get text, document structure, EDUs and RST tree from the .dis file
    # TODO re-implement in a clean manner
    orig_rsttree = parse.read_annotation_file(dis_filename,
                                              txt_filename)
    # TODO re-implement in a clean manner
    orig_rsttree.set_origin(doc_key)  # re-think origin
    rst_context = treenode(orig_rsttree).context
    # create the DocumentPlus
    docp = DocumentPlus(doc_key, grouping, rst_context)

    # attach RST information
    # edus = orig_rsttree.leaves()
    # docp.edus.extend(edus)
    # TODO move where appropriate
    # convert relation labels if needed
    if rel_conv is not None:
        orig_rsttree = rel_conv(orig_rsttree)
    # attach original RST tree
    docp.orig_rsttree = orig_rsttree
    # convert to binary tree
    rsttree = SimpleRSTTree.from_rst_tree(orig_rsttree)
    docp.rsttree = rsttree
    # convert to dep tree
    deptree = RstDepTree.from_simple_rst_tree(rsttree)
    docp.deptree = deptree
    # get EDUs (bad)
    # TODO: get EDUs from orig_rsttree.leaves() (or even better from .edus)
    # and let DocumentPlus do the left padding
    docp.edus = deptree.edus

    # align EDUs with document structure
    docp.align_with_doc_structure()

    # get syntactic information
    if syn_parser is not None:
        docp = syn_parser.tokenize(docp)
        docp = syn_parser.parse(docp)
        # align with EDUs
        docp = docp.align_with_trees()
        docp = docp.align_with_tokens()
    # dummy, fallback tokenization if there is not PTB gold or silver
    docp = docp.align_with_raw_words()

    return docp

def load_rst_wsj_documents(corpusdir, load_content=True, labelset='coarse',
                           syn_parser=None):
    """Load documents.

    Parameters
    ----------
    corpusdir: path to the directory whose files must be loaded

    load_content: boolean, optional (default=True)
        Whether to load or not the content of the documents.
        If true, this function returns a dict from FileId to
        RSTTree (with an RSTContext). If not, it returns a dict
        from FileId to filepaths (dis_file, text_file).
    """
    # TODO directly get raw documents and from there .edus and .dis
    dis_filenames = sorted(glob(os.path.join(corpusdir, '*.dis')))
    txt_filenames = [os.path.splitext(dis_fname)[0]
                     for dis_fname in dis_filenames]
    edus_filenames = [os.path.join(txt_filename, '.edus')
                      for txt_filename in txt_filenames]

    # create `educe.FileId`s
    file_ids = [mk_key(os.path.splitext(os.path.basename(dis_fname))[0])
                for dis_fname in dis_filenames]

    if load_content:
        # RST relation converter: fine to coarse label
        if labelset is 'coarse':
            relmap_file = join(dirname(__file__), 'rst_112to18.txt')
            tree_relconv = RstRelationConverter(relmap_file).convert_tree
        else:
            tree_relconv = None
        # return dict(FileId, DocumentPlus)
        return {file_id: _open_plus(file_id, dis_fn, txt_fn, tree_relconv,
                                    syn_parser)
                for file_id, dis_fn, txt_fn
                in zip(file_ids, dis_filenames, txt_filenames)}

    # otherwise, return a dict(FileId, (dis_filename, txt_filename))
    # for backwards compatibility (equiv. to Reader.files())
    return {file_id: (dis_fn, txt_fn)
            for file_id, dis_fn, txt_fn
            in zip(file_ids, dis_filenames, txt_filenames)}



def load_rst_wsj_corpus(corpus_home, subset='train', labelset='coarse',
                        syn_parser=None):
    """Load the documents from the RST-WSJ-corpus.

    A standard call would be
    `load_rst_wsj_corpus('data/rst_discourse_treebank/data')`

    Parameters
    ----------
    corpus_home: path to the root dir of the corpus
        The expected value is a path like
        `educe/data/rst_discourse_treebank/data`.

    subset: 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both.

    labelset: 'coarse' or any other value, optional
        If 'coarse', apply the standard mapping from fine- to coarse-grained
        labels, otherwise leave labels as they are.

    syn_parser: PtbParser, optional
        Pass the syntactic parser that will provide tokenization, POS
        tagging and syntactic constituent trees.
    """
    train_path = os.path.join(corpus_home, TRAIN_FOLDER)
    test_path = os.path.join(corpus_home, TEST_FOLDER)
    double_path = os.path.join(corpus_home, DOUBLE_FOLDER)

    if subset == 'train':
        return load_rst_wsj_documents(train_path, labelset=labelset,
                                      syn_parser=syn_parser)
    elif subset == 'test':
        return load_rst_wsj_documents(test_path, labelset=labelset,
                                      syn_parser=syn_parser)
    elif subset == 'double':
        return load_rst_wsj_documents(double_path, labelset=labelset,
                                      syn_parser=syn_parser)
    elif subset == 'all':
        corpus = dict(train=load_rst_wsj_documents(train_path,
                                                   labelset=labelset,
                                                   syn_parser=syn_parser),
                      test=load_rst_wsj_documents(test_path,
                                                  labelset=labelset,
                                                  syn_parser=syn_parser),
                      double=load_rst_wsj_documents(double_path,
                                                    labelset=labelset,
                                                    syn_parser=syn_parser))
        # TODO merge and return one big dict (?)
        return corpus
    else:
        err_msg = ("subset can only be 'train', 'test', 'double' or 'all',",
                   " got {}".format(subset))
        raise ValueError(err_msg)
