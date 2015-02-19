"""This submodule implements a document with additional information.

"""

from __future__ import print_function

import copy
import itertools

from educe.external.postag import Token
from .text import Sentence, Paragraph
from .annotation import EDU


# helpers for _align_with_doc_structure

def containing(span):
    """
    span -> anno -> bool

    if this annotation encloses the given span
    """
    return lambda x: x.text_span().encloses(span)


def _filter0(pred, iterable):
    """
    First item that satisifies a predicate in a given
    iterable, otherwise None
    """
    for item in iterable:
        if pred(item):
            return item
    else:
        return None


class DocumentPlus(object):
    """A document and relevant contextual information"""

    def __init__(self, key, grouping):
        """
        key is an educe.corpus.FileId
        grouping designates the corresponding file in the corpus
        """
        # document identification
        self.key = key
        self.grouping = grouping

        # document structure: sentences and paragraphs from the raw text
        _lpad_sent = Sentence.left_padding()
        self.raw_sentences = [_lpad_sent]
        _lpad_para = Paragraph.left_padding([_lpad_sent])
        self.paragraphs = [_lpad_para]

        # RST annotation
        _lpad_edu = EDU.left_padding()
        self.edus = [_lpad_edu]
        # various flavours of RST trees
        self.orig_rsttree = None
        self.rsttree = None
        self.deptree = None

        # syntactic information
        # * tokens
        _lpad_tok = Token.left_padding()
        self.tkd_tokens = [_lpad_tok]
        # * trees
        self.tkd_trees = [None]

    def align_with_doc_structure(self):
        """Align EDUs with the document structure (paragraph and sentence).

        Determine which paragraph and sentence (if any) surrounds
        this EDU. Try to accomodate the occasional off-by-a-smidgen
        error by folks marking these EDU boundaries, eg. original
        text:

        Para1: "Magazines are not providing us in-depth information on
        circulation," said Edgar Bronfman Jr., .. "How do readers feel
        about the magazine?...
        Research doesn't tell us whether people actually do read the
        magazines they subscribe to."

        Para2: Reuben Mark, chief executive of Colgate-Palmolive, said...

        Marked up EDU is wide to the left by three characters:
        "

        Reuben Mark, chief executive of Colgate-Palmolive, said...
        """
        edus = self.edus
        paragraphs = self.paragraphs
        raw_sentences = self.raw_sentences
        rst_context = self.rst_context

        edu2raw_sent = []
        edu2para = []
        surrounders = dict()
        # align fake root EDU with fake paragraph and fake sentence
        edu2raw_sent.append(0)
        edu2para.append(0)
        surrounders[edus[0]] = (paragraphs[0], raw_sentences[0])

        # align the other EDUs 
        for edu in edus[1:]:
            espan = edu.text_span()
            para = _filter0(containing(espan), paragraphs)
            # sloppy EDUs happen; try shaving off some characters
            # if we can't find a paragraph
            if para is None:
                espan = copy.copy(espan)
                espan.char_start += 1
                espan.char_end -= 1
                etext = rst_context.text(espan)
                # kill left whitespace
                espan.char_start += len(etext) - len(etext.lstrip())
                etext = etext.lstrip()
                # kill right whitespace
                espan.char_end -= len(etext) - len(etext.rstrip())
                etext = etext.rstrip()
                # try again
                para = _filter0(containing(espan), paragraphs)

            sent = (_filter0(containing(espan), para.sentences)
                    if para else None)
            # update mapping
            para_idx = (paragraphs.index(para) if para is not None
                        else None)  # TODO or -1 or ... ?
            edu2para.append(para_idx)

            raw_sent_idx = (raw_sentences.index(sent) if sent is not None
                            else None)
            edu2raw_sent.append(raw_sent_idx)
            # update surrounders
            surrounders[edu] = (para, sent)

        self.surrounders = surrounders
        return self

    def align_with_tokens(self):
        """Compute for each EDU the overlapping tokens"""
        if self.tkd_tokens is None:
            self.ptb_tokens = None
            return self

        # TODO possibly: replace this with a greedy procedure
        # that assigns each token to exactly one EDU

        ptb_tokens = dict()  # overlapping PTB tokens

        edus = self.edus
        edu2sent = self.edu2sent
        syn_trees = self.tkd_trees
        for i, edu in enumerate(edus):
            if edu.is_left_padding():
                start_token = Token.left_padding()
                ptb_toks = [start_token]
            else:
                ptb_toks = [tok for tok in self.tkd_tokens
                            if tok.overlaps(edu)]
                # if possible, take tokens only from the tree chosen by
                # align_with_trees()
                ptree_idx = edu2sent[i]
                if ptree_idx is not None:
                    ptree = syn_trees[ptree_idx]
                    if ptree is not None:
                        clean_ptb_toks = [tok for tok in ptree.leaves()
                                          if tok.overlaps(edu)]
                        ptb_toks = clean_ptb_toks
            ptb_tokens[edu] = ptb_toks

        self.ptb_tokens = ptb_tokens

        return self

    def align_with_trees(self):
        """Compute for each EDU the overlapping trees"""
        syn_trees = self.tkd_trees

        # if there is no sentence segmentation from syntax,
        # use the raw (bad) one from the .out
        if syn_trees is None:
            self.edu2sent = self.edu2raw_sent
            self.ptb_trees = None  # mark for deprecation
            return self

        edu2sent = []
        ptb_trees = dict()  # mark for deprecation

        edus = self.edus
        # left padding: EDU 0
        assert edus[0].is_left_padding()
        edu2sent.append(None)
        ptb_trees[edus[0]] = []  # mark for deprecation

        for edu in edus[1:]:
            ptrees = [tree for tree in syn_trees
                      if (tree is not None and
                          tree.overlaps(edu))]
            ptb_trees[edu] = ptrees  # mark for deprecation
            # get the actual tree
            if len(ptrees) == 0:
                ptree_idx = None
            elif len(ptrees) == 1:
                ptree = ptrees[0]
                ptree_idx = syn_trees.index(ptree)
            else:  # len(ptrees) > 1
                # if more than one PTB trees overlap with this EDU,
                # pick the PTB tree with maximal overlap
                # if it covers at least 90% of the chars of the EDU
                len_espan = edu.span.length()  # length of EDU span
                ovlaps = [t.overlaps(edu).length()
                          for t in ptrees]
                ovlap_ratios = [float(ovlap) / len_espan
                                for ovlap in ovlaps]
                # cry for help if it goes bad
                if max(ovlap_ratios) < 0.5:
                    emsg = 'Slightly unsure about this EDU segmentation'
                    print('EDU: ', edu)
                    # print('ptrees: ', [t.leaves() for t in ptrees])
                    # raise ValueError(emsg)
                # otherwise just emit err msgs for info
                if False:
                    err_msg = 'More than one PTB tree for this EDU'
                    print(err_msg)
                    print('EDU: ', edu)
                    print('ovlap_ratios: ', ovlap_ratios)
                # proceed and get the tree with max overlap
                max_idx = ovlap_ratios.index(max(ovlap_ratios))
                ptree = ptrees[max_idx]
                ptree_idx = syn_trees.index(ptree)
            edu2sent.append(ptree_idx)

        self.edu2sent = edu2sent
        self.ptb_trees = ptb_trees  # mark for deprecation

        return self

    def all_edu_pairs(self):
        """Generate all EDU pairs of a document"""
        edus = self.edus
        all_pairs = [epair for epair in itertools.product(edus, edus[1:])
                     if epair[0] != epair[1]]
        return all_pairs

    def relations(self, edu_pairs):
        """Get the relation that holds in each of the edu_pairs"""
        if not self.deptree:
            return [None for epair in edu_pairs]

        rels = {(src, tgt): rel
                for src, tgt, rel in self.deptree.get_dependencies()}
        erels = [rels.get(epair, 'UNRELATED')
                 for epair in edu_pairs]
        return erels
