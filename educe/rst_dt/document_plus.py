"""This submodule implements a document with additional information.

"""

from __future__ import print_function

import copy
import itertools

import numpy as np

from educe.external.postag import Token
from educe.util import relative_indices
from .text import Sentence, Paragraph, clean_edu_text
from .annotation import EDU
from .ptb import align_edus_with_sentences


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


# dirty temporary extraction from DocumentPlus
def align_edus_with_paragraphs(doc_edus, doc_paras, text, strict=False):
    """Align EDUs with paragraphs, if any.

    Parameters
    ----------
    doc_edus:

    doc_paras:

    strict:


    Returns
    -------
    edu2para: list(int) or None
        Index of the paragraph that contains each EDU, None if the
        paragraph segmentation is missing.
    """
    if doc_paras is None:
        return None

    edu2para = []
    for edu in doc_edus:
        espan = edu.text_span()
        # find enclosing paragraph
        para = _filter0(containing(espan), doc_paras)
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
            para = _filter0(containing(espan), doc_paras)
            # DEBUG
            if False:
                if para is None:
                    print('EP ({}): {}'.format(self.grouping, edu))
            # end DEBUG

        # update edu to paragraph mapping
        para_idx = (doc_paras.index(para) if para is not None
                    else None)  # TODO or -1 or ...
        edu2para.append(para_idx)

    return edu2para
# end dirty


class DocumentPlus(object):
    """A document and relevant contextual information"""

    def __init__(self, key, grouping, rst_context):
        """
        Parameters
        ----------
        key: educe.corpus.FileId
            Unique identifier for this document
        grouping: string
            Path to the corresponding file in the corpus
        rst_context: RSTContext
            Encapsulating object for the text and raw document structure
            (sentences and paragraphs)
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
        # dirty extraction
        if paragraphs is None:
            edu2para = None
        else:
            edu2para = align_edus_with_paragraphs(edus[1:], paragraphs[1:],
                                                  text)
        # prepend [0] for the left padding EDU and paragraph
        edu2para = ([0] + edu2para if edu2para is not None
                    else [None for edu in edus])
        self.edu2para = edu2para

        # compute relative index of each EDU to the beginning (resp. to
        # the end) of the paragraph
        idxes_in_para = relative_indices(edu2para)
        rev_idxes_in_para = relative_indices(edu2para, reverse=True)
        self.edu2idx_in_para = idxes_in_para
        self.edu2rev_idx_in_para = rev_idxes_in_para

        # align EDUs with raw sentences
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
    def align_with_tokens(self, verbose=False):
        """Compute for each EDU the overlapping tokens."""
        edus = self.edus
        tokens = self.tkd_tokens
        if len(tokens) == 1:  # only lpad
            self.edu2tokens = None
            return self

        # EDU segmentation in the RST corpus and token segmentation
        # in the PTB occasionally conflict ; in such cases, a PTB
        # token overlaps with two distinct EDUs.
        # In the first, naive and costly implementation of this
        # procedure, overlapping tokens were considered part of both
        # overlapped EDUs.
        # We stick to this behaviour by computing two mappings, on the
        # beginning and end of the spans of tokens and EDUs, and
        # using the union of them as the final mapping
        edu_ends = [x.span.char_end for x in edus]
        tok_ends = [x.span.char_end for x in tokens]
        tok2edu_ends = np.searchsorted(edu_ends, tok_ends)
        edu_begs = [x.span.char_start for x in edus]
        tok_begs = [x.span.char_start for x in tokens]
        tok2edu_begs = np.searchsorted(edu_begs, tok_begs, side='right')
        tok2edu_begs = tok2edu_begs - 1
        # dirty hack to recover a proper mapping for the left padding
        # token and EDU ; we could avoid this by setting the span of
        # the left padding to e.g. (-1, 0), but there are possible
        # side-effects, so maybe later?
        tok2edu_begs[0] = 0

        # optional check for mismatches between EDU and token
        # segmentations
        if verbose:
            differences = (tok2edu_begs != tok2edu_ends)
            if any(differences):
                print('Mismatch: EDU vs token segmentation')
                print(self.key.doc)
                print([str(x) for x in np.array(tokens)[differences]])
                diff_idc = np.where(differences)[0]
                print(diff_idc)
                print(np.array(edu_begs)[tok2edu_begs[diff_idc]],
                      np.array(edu_ends)[tok2edu_begs[diff_idc]])
                print(np.array(edu_begs)[tok2edu_ends[diff_idc]],
                      np.array(edu_ends)[tok2edu_ends[diff_idc]])

        # build the mapping from each EDU to token indices
        edu2tokens_begs = {k: [tok_idx for tok_idx, edu_idx in g]
                           for k, g in itertools.groupby(
                                   enumerate(tok2edu_begs),
                                   key=lambda x: x[1])}
        edu2tokens_ends = {k: [tok_idx for tok_idx, edu_idx in g]
                           for k, g in itertools.groupby(
                                   enumerate(tok2edu_ends),
                                   key=lambda x: x[1])}
        edu2tokens = [np.union1d(edu2tokens_begs.get(edu_idx, []),
                                 edu2tokens_ends.get(edu_idx, []))
                      for edu_idx in range(len(edus))]

        self.edu2tokens = edu2tokens
        return self

    def align_with_trees(self, strict=False):
        """Compute for each EDU the overlapping trees"""
        edus = self.edus
        syn_trees = self.tkd_trees

        # if there is no sentence segmentation from syntax,
        # use the raw (bad) one from the .out
        if len(syn_trees) == 1:  # only lpad
            self.edu2sent = self.edu2raw_sent
            return self

        # compute edu2sent, prepend 0 for lpad, shift all indices by 1
        assert edus[0].is_left_padding()
        edu2sent = align_edus_with_sentences(self.edus[1:], syn_trees[1:],
                                             strict=strict)
        edu2sent = [0] + [(i+1 if i is not None else i)
                          for i in edu2sent]
        self.edu2sent = edu2sent

        # compute relative index of each EDU from the beginning (resp. to
        # the end) of the sentence around
        idxes_in_sent = relative_indices(edu2sent)
        rev_idxes_in_sent = relative_indices(edu2sent, reverse=True)
        self.edu2idx_in_sent = idxes_in_sent
        self.edu2rev_idx_in_sent = rev_idxes_in_sent

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
