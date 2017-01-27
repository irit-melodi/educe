# Author: Eric Kow
# License: BSD3

"""
Corpus management (re-exported by educe.rst_dt)
"""

from glob import glob
import os
import sys

from nltk import Tree

from educe.corpus import FileId
from educe.internalutil import treenode
from educe.rst_dt import parse
import educe.util
import educe.corpus
from .document_plus import DocumentPlus
from .annotation import SimpleRSTTree, _binarize
from .deptree import RstDepTree
from .pseudo_relations import rewrite_pseudo_rels


RELMAP_112_18_FILE = os.path.join(
    os.path.dirname(__file__), 'rst_112to18.txt')


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

    Parameters
    ----------
    corpus_dir : string
        TODO

    args : TODO
        TODO

    coarse_rels : boolean, optional
        If True, relation labels are converted to their coarse-grained
        equivalent.

    nary_conv : string, optional
        Conversion method from constituency to dependency tree, for
        n-ary spans, n > 2, whose kids are all nuclei:
        'tree' picks the leftmost nucleus as the head of all the others
        (effectively a tree), 'chain' attaches each nucleus to its
        predecessor (effectively a chain).

    nuc_in_label : boolean, optional
        If True, incorporate nuclearity into the label (ex:
        elaboration-NS) ; currently BROKEN (defined on SimpleRSTTree
        only).

    exclude_file_docs : boolean, default False
        If True, ignore fileX files.

    TODO
    ----
    [ ] port incorporate_nuclearity_into_label from SimpleRSTTree to
        RSTTree

    [ ] kill `args`
    """

    def __init__(self, corpus_dir, args, coarse_rels=False,
                 fix_pseudo_rels=False,
                 nary_conv='chain',
                 nuc_in_label=False,
                 exclude_file_docs=False):
        self.reader = Reader(corpus_dir)
        # pre-load corpus
        anno_files_unfltd = self.reader.files(exclude_file_docs)
        is_interesting = educe.util.mk_is_interesting(args)
        anno_files = self.reader.filter(anno_files_unfltd, is_interesting)
        self.corpus = self.reader.slurp(anno_files, verbose=True)
        # WIP rewrite pseudo-relations
        self.fix_pseudo_rels = fix_pseudo_rels
        # setup label converter for the desired granularity
        # 'fine' means we don't change anything
        if coarse_rels:
            relmap_file = RELMAP_112_18_FILE
            self.rel_conv = RstRelationConverter(relmap_file).convert_tree
        else:
            self.rel_conv = None
        # how to convert n-ary spans
        self.nary_conv = nary_conv
        if nary_conv not in ['chain', 'tree']:
            err_msg = 'Unknown conversion for n-ary spans: {}'
            raise ValueError(err_msg.format(nary_conv))
        # whether nuclearity should be part of the label
        self.nuc_in_label = nuc_in_label

    def decode(self, doc_key):
        """Decode a document from the RST-DT (gold)

        Parameters
        ----------
        doc_key: string ?
            Identifier (in corpus) of the document we want to decode.

        Returns
        -------
        doc: DocumentPlus
            Bunch of information about this document notably its list of
            EDUs and the structures defined on them: RSTTree,
            SimpleRSTTree, RstDepTree.
        """
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
        # (optional) rewrite pseudo-relations
        if self.fix_pseudo_rels:
            orig_rsttree = rewrite_pseudo_rels(doc_key, orig_rsttree)
        # (optional) convert relation labels
        if self.rel_conv is not None:
            orig_rsttree = self.rel_conv(orig_rsttree)
        doc.orig_rsttree = orig_rsttree

        # TO DEPRECATE - shunt SimpleRSTTree (possible?)
        # convert to binary tree
        rsttree = SimpleRSTTree.from_rst_tree(orig_rsttree)
        # WIP incorporate nuclearity into label
        if self.nuc_in_label:
            rsttree = SimpleRSTTree.incorporate_nuclearity_into_label(rsttree)
        doc.rsttree = rsttree
        # end TO DEPRECATE

        # convert to dep tree
        # WIP
        if self.nary_conv == 'chain':
            # legacy mode, through SimpleRSTTree
            # deptree = RstDepTree.from_simple_rst_tree(rsttree)
            # modern mode, directly from a binarized RSTTree
            deptree = RstDepTree.from_rst_tree(_binarize(orig_rsttree))
        else:  # tree conversion
            deptree = RstDepTree.from_rst_tree(orig_rsttree)
        # end WIP
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
