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
        rst_context is an RSTContext that encapsulates the text and raw
            document structure
        """
        # document identification
        self.key = key
        self.grouping = grouping
        # document text and basic structure
        self.text = rst_context.text()

        # document structure
        # prepare left padding objects, in case we need them later
        _lpad_sent = Sentence.left_padding()
        _lpad_para = Paragraph.left_padding([_lpad_sent])
        # raw sentences
        raw_sentences = rst_context.sentences
        if raw_sentences is None:
            raw_sents = None
        else:
            raw_sents = []
            raw_sents.append(_lpad_sent)
            raw_sents.extend(raw_sentences)
        self.raw_sentences = raw_sents
        # paragraphs
        paragraphs = rst_context.paragraphs
        if paragraphs is None:
            paras = None
        else:
            paras = []
            paras.append(_lpad_para)
            paras.extend(paragraphs)
        self.paragraphs = paras

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

        # syntactic information

        # tokens
        self.tkd_tokens = []
        # left padding
        _lpad_tok = Token.left_padding()
        self.tkd_tokens.append(_lpad_tok)

        # trees
        self.tkd_trees = []
        # left padding
        _lpad_tree = None
        self.tkd_trees.append(_lpad_tree)

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
        text = self.text
        edus = self.edus

        # align EDUs with paragraphs
        paragraphs = self.paragraphs
        if paragraphs is None:
            edu2para = [None for edu in edus]
        else:
            edu2para = []
            edu2para.append(0)  # left padding
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
                    etext = text[espan.char_start:espan.char_end]
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
                            else None)  # TODO or -1 or ...
                edu2para.append(para_idx)

        self.edu2para = edu2para

        # align EDUs with sentences
        # NB: this usually fails due to bad sentence segmentation, e.g.
        # ... Prof.\nHarold ... in wsj_##.out files,
        # or end of sentence missing in file## files.
        raw_sentences = self.raw_sentences
        if raw_sentences is None:
            edu2raw_sent = [None for edu in edus]
        else:
            edu2raw_sent = []
            edu2raw_sent.append(0)  # left padding
            # align the other EDUs
            for edu in edus[1:]:
                espan = edu.text_span()
                # find enclosing raw sentence
                sent = _filter0(containing(espan), raw_sentences)
                # sloppy EDUs happen; try shaving off some characters
                # if we can't find a sentence
                if sent is None:
                    # DEBUG
                    if False:
                        print('WP ({}) : {}'.format(self.grouping, edu))
                    # end DEBUG
                    espan = copy.copy(espan)
                    espan.char_start += 1
                    espan.char_end -= 1
                    etext = text[espan.char_start:espan.char_end]
                    # kill left whitespace
                    espan.char_start += len(etext) - len(etext.lstrip())
                    etext = etext.lstrip()
                    # kill right whitespace
                    espan.char_end -= len(etext) - len(etext.rstrip())
                    etext = etext.rstrip()
                    # try again
                    sent = _filter0(containing(espan), raw_sentences)
                    # DEBUG
                    if False:
                        if sent is None:
                            print('EP ({}): {}'.format(self.grouping, edu))
                    # end DEBUG

                # update edu to sentence mapping
                raw_sent_idx = (raw_sentences.index(sent) if sent is not None
                                else None)  # TODO or -1 or ... ?
                edu2raw_sent.append(raw_sent_idx)

        self.edu2raw_sent = edu2raw_sent

        # surrounders (backwards compat)
        # map left padding EDU to left padding paragraph and raw_sentence
        surrounders = dict()
        for i, edu in enumerate(edus):
            if edu2para is None or edu2para[i] is None:
                para = None
            else:
                para = paragraphs[edu2para[i]]

            if edu2raw_sent is None or edu2raw_sent[i] is None:
                raw_sent = None
            else:
                raw_sent = raw_sentences[edu2raw_sent[i]]

            surrounders[edu] = (para, raw_sent)

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

        # left padding EDU
        assert edus[0].is_left_padding()
        edu2sent.append(0)
        ptb_trees[edus[0]] = []  # mark for deprecation
        # regular EDUs
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
