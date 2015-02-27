"""This submodule implements a document with additional information.

"""

from __future__ import print_function

import copy
import itertools

from educe.external.postag import Token
from .text import Sentence, Paragraph, clean_edu_text
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

    def __init__(self, key, grouping, rst_context):
        """
        key is an educe.corpus.FileId
        grouping designates the corresponding file in the corpus
        rst_context contains the document text and structure
        (paragraphs, raw sentences)
        """
        # document identification
        self.key = key
        self.grouping = grouping

        # document text and basic structure
        self.rst_context = rst_context
        # RSTContext provides paragraphs made of (raw) sentences
        raw_sentences = []
        paragraphs = []
        # add left padding
        _lpad_sent = Sentence.left_padding()
        raw_sentences.append(_lpad_sent)
        _lpad_para = Paragraph.left_padding([_lpad_sent])
        paragraphs.append(_lpad_para)
        # add real paragraphs and raw sentences
        paras = rst_context.paragraphs
        raw_sents_iter = (para.sentences for para in paras)
        raw_sents = list(itertools.chain.from_iterable(raw_sents_iter))
        raw_sentences.extend(raw_sents)
        paragraphs.extend(paras)
        self.raw_sentences = raw_sentences
        self.paragraphs = paragraphs

        # TODO
        # self.words = []
        # self.tags = []
        # self.sentences = []
        # self.syn_trees = []
        # the basic building block should be words/tags,
        # additional annotation being defined by its offset (in nb of words):
        # sentences, paragraphs, EDUs...

        # RST annotation
        self.edus = []
        # left padding on EDUs
        _lpad_edu = EDU.left_padding()
        self.edus.append(_lpad_edu)
        # various flavours of RST trees
        self.orig_rsttree = None
        self.rsttree = None
        self.deptree = None

        # syntactic information
        self.tkd_tokens = []
        self.tkd_trees = []
        # left padding on syntactic info
        _lpad_tok = Token.left_padding()
        self.tkd_tokens.append(_lpad_tok)
        self.tkd_trees.append(None)

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
        rst_context = self.rst_context
        paragraphs = self.paragraphs
        raw_sentences = self.raw_sentences

        # mappings from EDU to raw_sentence and paragraph
        edu2raw_sent = []
        edu2para = []
        surrounders = dict()  # alternative storage (old)

        # map left padding EDU to left padding paragraph and raw_sentence
        edu2raw_sent.append(0)
        edu2para.append(0)
        surrounders[edus[0]] = (paragraphs[0], raw_sentences[0])

        # align the other EDUs
        for edu in edus[1:]:
            espan = edu.text_span()

            # find enclosing paragraph
            para = _filter0(containing(espan), paragraphs)
            # sloppy EDUs happen; try shaving off some characters
            # if we can't find a paragraph
            if para is None:
                # DEBUG
                if False:
                    print('WP ({}) : {}'.format(self.grouping, edu))
                # end DEBUG
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
                # DEBUG
                if False:
                    if para is None:
                        print('EP ({}): {}'.format(self.grouping, edu))
                # end DEBUG

            # update edu to paragraph mapping
            para_idx = (paragraphs.index(para) if para is not None
                        else None)  # TODO or -1 or ... ?
            edu2para.append(para_idx)

            # find enclosing sentence, with raw sentence segmentation from
            # the text file
            # NB: this usually fails due to bad sentence segmentation, e.g.
            # ... Prof.\nHarold ... in wsj_##.out files,
            # or end of sentence missing in file## files.
            sent = (_filter0(containing(espan), para.sentences)
                    if para else None)
            # DEBUG
            if False:
                if sent is None:
                    print('WS ({}): {}'.format(self.grouping, edu))
            # end DEBUG
            # update mapping
            raw_sent_idx = (raw_sentences.index(sent) if sent is not None
                            else None)
            edu2raw_sent.append(raw_sent_idx)

            # update surrounders
            surrounders[edu] = (para, sent)

        self.paragraphs = paragraphs
        self.edu2para = edu2para
        self.raw_sentences = raw_sentences
        self.edu2raw_sent = edu2raw_sent

        self.surrounders = surrounders  # mark for deprecation

        return self

    def align_with_raw_words(self):
        """Compute for each EDU the raw tokens it contains

        This is a dirty temporary hack to enable backwards compatibility.
        There should be one clean text per document, one tokenization and
        so on, but, well.
        """
        edus = self.edus
        raw_words = dict()

        for edu in edus:
            if edu.is_left_padding():
                _lpad_tok = self.tkd_tokens[0]
                raw_wds = [_lpad_tok]
            else:
                # TODO move functionality to rst_wsj_corpus
                cln_txt = clean_edu_text(edu.text())
                # dummy tokenization on whitespaces
                words = cln_txt.split()
                # lowercase all words
                raw_wds = [w.lower() for w in words]
            raw_words[edu] = raw_wds

        self.raw_words = raw_words

        return self

    # TODO move functionality to ptb.py
    def align_with_tokens(self):
        """Compute for each EDU the overlapping tokens"""
        tokens = self.tkd_tokens
        if len(tokens) == 1:  # only lpad
            self.ptb_tokens = None
            return self

        # TODO possibly: replace this with a greedy procedure
        # that assigns each token to exactly one EDU

        ptb_tokens = dict()  # overlapping PTB tokens

        syn_trees = self.tkd_trees
        edus = self.edus
        edu2sent = self.edu2sent

        for i, edu in enumerate(edus):
            if edu.is_left_padding():
                start_token = tokens[0]
                ptb_toks = [start_token]
            else:
                ptb_toks = [tok for tok in tokens
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

    # TODO move functionality to ptb.py
    def align_with_trees(self):
        """Compute for each EDU the overlapping trees"""
        syn_trees = self.tkd_trees

        # if there is no sentence segmentation from syntax,
        # use the raw (bad) one from the .out
        if len(syn_trees) == 1:  # only lpad
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
                # no tree at all can happen when the EDU text is totally
                # absent from the list of sentences of this doc in the PTB
                # ex: wsj_0696.out, last sentence
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
                    # print('EDU: ', edu)
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
