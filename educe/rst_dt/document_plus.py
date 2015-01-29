"""This submodule implements a document with additional information.

"""

import copy
import itertools

from educe.internalutil import ifilter
from educe.external.postag import Token
from .text import Sentence, Paragraph
from .annotation import EDU


# document structure

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
    matches = ifilter(pred, iterable)
    try:
        return matches.next()
    except StopIteration:
        return None


def _surrounding_text(edu):
    """
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
    if edu.is_left_padding():
        sent = Sentence.left_padding()
        para = Paragraph.left_padding([sent])
        return para, sent
    # normal case
    espan = edu.text_span()
    para = _filter0(containing(espan), edu.context.paragraphs)
    # sloppy EDUs happen; try shaving off some characters
    # if we can't find a paragraph
    if para is None:
        espan = copy.copy(espan)
        espan.char_start += 1
        espan.char_end -= 1
        etext = edu.context.text(espan)
        # kill left whitespace
        espan.char_start += len(etext) - len(etext.lstrip())
        etext = etext.lstrip()
        # kill right whitespace
        espan.char_end -= len(etext) - len(etext.rstrip())
        etext = etext.rstrip()
        # try again
        para = _filter0(containing(espan), edu.context.paragraphs)

    sent = _filter0(containing(espan), para.sentences) if para else None
    return para, sent

# end document structure


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

        # document content
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
        self.tkd_trees = []

    def align_with_doc_structure(self):
        """Align EDUs with the document structure (paragraph and sentence)."""
        self.surrounders = {edu: _surrounding_text(edu) for edu in self.edus}
        return self

    def align_with_tokens(self):
        """Compute for each EDU the overlapping tokens"""
        if self.tkd_tokens is None:
            self.ptb_tokens = None
            self.eduptr = None
            return self

        self.ptb_tokens = dict()  # overlapping PTB tokens
        self.eduptr = [0]  # EDU i is made of tokens [eduptr[i]:eduptr[i+1]]

        # greedily add tokens to each EDU in turn
        tokens = iter(self.tkd_tokens)
        tok = tokens.next()
        for edu in self.edus:
            ptb_toks = []
            while tok.overlaps(edu):
                ptb_toks.append(tok)
                tok = tokens.next()
            self.ptb_tokens[edu] = ptb_toks
            self.eduptr.append(len(ptb_toks))
        # DEBUG
        for edu in self.edus:
            print('EDU: ', edu)
            print('===> ', self.ptb_tokens[edu])
        import sys
        sys.exit()
        # end DEBUG
        return self

    def align_with_trees(self):
        """Compute for each EDU the overlapping trees"""
        if self.tkd_trees is None:
            self.ptb_trees = None
            return self

        self.ptb_trees = dict()
        for edu in self.edus:
            if edu.is_left_padding():
                ptb_trees = []
            else:
                ptb_trees = [tree for tree in self.tkd_trees
                             if tree.overlaps(edu)]
            self.ptb_trees[edu] = ptb_trees
        return self

    def all_edu_pairs(self):
        """Generate all EDU pairs of a document"""
        edus = self.edus
        all_pairs = [epair for epair in itertools.product(edus, edus[1:])
                     if epair[0] != epair[1]]
        return all_pairs

    def sorted_all_inv_edu_pairs(self):
        """Get a sorted list of all the inverted EDU pairs of a document"""
        epairs = self.all_edu_pairs()
        # sort EDU pairs by target EDU, then source EDU
        inv_pairs = sorted((tgt, src) for src, tgt in epairs)
        return inv_pairs

    def relations(self, edu_pairs):
        """Get the relation that holds in each of the edu_pairs"""
        if not self.deptree:
            return [None for epair in edu_pairs]

        rels = {(src, tgt): rel
                for src, tgt, rel in self.deptree.get_dependencies()}
        erels = [rels.get(epair, 'UNRELATED')
                 for epair in edu_pairs]
        return erels

    def inv_relations(self, inv_edu_pairs):
        """Get the relation that holds in each of the inv_edu_pairs"""
        epairs = [(src, tgt) for tgt, src in inv_edu_pairs]
        erels = self.relations(epairs)
        return erels
