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
    def align_with_tokens(self):
        """Compute for each EDU the overlapping tokens"""
        tokens = self.tkd_tokens
        if len(tokens) == 1:  # only lpad
            self.edu2tokens = None
            return self

        # TODO possibly: replace this with a greedy procedure
        # that assigns each token to exactly one EDU

        edu2tokens = []  # tokens that overlap with this EDU

        edus = self.edus

        for i, edu in enumerate(edus):
            if edu.is_left_padding():
                tok_idcs = [0]  # 0 is the index of the start token
            else:
                tok_idcs = [tok_idx
                            for tok_idx, tok in enumerate(tokens)
                            if tok.overlaps(edu)]
                # TODO store the index of the first token of each EDU
                # this will be useful for future features

            edu2tokens.append(tok_idcs)

        self.edu2tokens = edu2tokens
        return self

    # TODO move functionality to ptb.py
    def align_with_trees(self, strict=False):
        """Compute for each EDU the overlapping trees"""
        syn_trees = self.tkd_trees

        # if there is no sentence segmentation from syntax,
        # use the raw (bad) one from the .out
        if len(syn_trees) == 1:  # only lpad
            self.edu2sent = self.edu2raw_sent
            return self

        edu2sent = []

        edus = self.edus

        # left padding EDU
        assert edus[0].is_left_padding()
        edu2sent.append(0)

        # regular EDUs
        for edu in edus[1:]:
            tree_idcs = [tree_idx
                         for tree_idx, tree in enumerate(syn_trees)
                         if tree is not None and tree.overlaps(edu)]

            if len(tree_idcs) == 1:
                tree_idx = tree_idcs[0]
            elif len(tree_idcs) == 0:
                # "no tree at all" can happen when the EDU text is totally
                # absent from the list of sentences of this doc in the PTB
                # ex: wsj_0696.out, last sentence
                if strict:
                    print(edu)
                    emsg = 'No PTB tree for this EDU'
                    raise ValueError(emsg)

                tree_idx = None
            else:
                # more than one PTB trees overlap with this EDU
                if strict:
                    emsg = ('Segmentation mismatch:',
                            'one EDU, more than one PTB tree')
                    print(edu)
                    for ptree in ptrees:
                        print('    ', [str(leaf) for leaf in ptree.leaves()])
                    raise ValueError(emsg)

                # heuristics: pick the PTB tree with maximal overlap
                # with the EDU span
                len_espan = edu.span.length()
                ovlaps = [syn_trees[tree_idx].overlaps(edu).length()
                          for tree_idx in tree_idcs]
                ovlap_ratios = [float(ovlap) / len_espan
                                for ovlap in ovlaps]
                # find the argmax
                max_idx = ovlap_ratios.index(max(ovlap_ratios))
                tree_idx = tree_idcs[max_idx]

            edu2sent.append(tree_idx)

        self.edu2sent = edu2sent
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
