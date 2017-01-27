# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Alignment the RST-WSJ-corpus with the Penn Treebank
"""

from __future__ import print_function

from os import path as fp
import itertools
import re

# pylint: disable=no-name-in-module
# pylint squawks about import error, but this seems to
# be some sort of fancy lazily loaded module which it's
# maybe a bit confused by
from nltk.corpus.reader import BracketParseCorpusReader
# pylint: enable=no-name-in-module

from educe.annotation import Span
from educe.external.parser import (ConstituencyTree)
from educe.external.postag import (generic_token_spans, Token)
from educe.internalutil import izip
from educe.ptb.annotation import (PTB_TO_TEXT, is_nonword_token,
                                  TweakedToken, transform_tree,
                                  strip_subcategory, prune_tree,
                                  is_non_empty, is_empty_category)
from educe.ptb.head_finder import find_lexical_heads


# map RST-WSJ files to PTB files

# fileN are exceptions to the regular mapping scheme
# they also (except for file4 ?) have a few mismatches between the RST-WSJ
# and PTB versions: missing tokens, wrong tokens...
# see _PTB_SUBSTS_OTHER
FILE_TO_PTB = {
    'file1': 'wsj_0764',
    'file2': 'wsj_0430',
    'file3': 'wsj_0766',
    'file4': 'wsj_0778',
    'file5': 'wsj_2172'
}


def _guess_ptb_name(k):
    """
    Given an RST DT corpus key, guess the equivalent filename from the
    Penn Tree Bank.

    Return None if the name doesn't look like it has an equivalent
    (note that returning something is not a guarantee either)
    """
    bname = fp.splitext(fp.basename(k.doc))[0]

    # use manual RST-WSJ to PTB file mapping if necessary
    if bname in FILE_TO_PTB:
        bname = FILE_TO_PTB[bname]

    # standard mapping scheme
    nparts = bname.split("_")
    if len(nparts) > 1:
        section = nparts[1][:2]  # wsj_2431 => 24
        return fp.join(section, bname + ".mrg")
    else:
        return None


# docs for which the PTB misses text at the end of the doc but the
# RST-WSJ does not, such as reference to other articles, place of
# writing (in signature at the end of readers' mail)...
# ex: (See: "XXXX Plans Rule on YYY" -- WSJ Oct. 27, 1989) in erratas
# NB: this means we don't have gold PTB trees for the extra text
PTB_MISSING_TEXT = [
    'wsj_0603',  # last 2 EDUs, footnote "(See: ...)"
    'wsj_0605',  # last 3 EDUs, footnote "(See: ...)"
    'wsj_0608',  # last 4 EDUs, footnote "(See: ...)"
    'wsj_0609',  # last 1 EDU, source
    'wsj_0611',  # last 2 EDUs, footnote "(See: ...)"
    'wsj_0614',  # last 7 EDUs, footnote "(See: ...)"
    'wsj_0694',  # last 1 EDU, end of address => 1 "sentence" (NP-LOC)
    'wsj_0696',  # last 2 EDUs, footnote "Source: ..." => 1 "sentence" (NP)
    'wsj_1107',  # last 1 EDU, end of address => 1 "sentence" (NP-LOC)
    'wsj_1377',  # last 1 EDU, footnote "(...)"
    'wsj_1382',  # last 2 EDUs, footnote "Source: ..." => 1 "sentence" (NP)
    'wsj_1970',  # last 2 EDUs, footnote "Source: ..." => 1 "sentence" (NP)
    'wsj_2352',  # last 1 EDU, end of address => 1 "sentence" (NP-LOC)
]

# docs for which the RST-WSJ-corpus file misses text at the end of the doc
RST_MISSING_TEXT = [
    # 'file1',  # handled in _PTB_SUBSTS_OTHER under wsj_0764
]


# docs for which the PTB contains erroneous sentence segmentation
PTB_WRONG_SENTENCE_SEG = [
    'wsj_0678',
    'wsj_1105',
    'wsj_1125',
    'wsj_1128',
    'wsj_1158',
    'wsj_1323',
    'wsj_2303',
]

# docs for which the RST-WSJ contains erroneous EDU segmentation that
# conflicts with correct segmentation in the PTB ; this is either
# sentence segmentation or tokenization
RST_WRONG_EDU_SEG = [
    'wsj_1123',
    'wsj_1158',  # tokenization
    'wsj_1373',
    'wsj_2317',
    'wsj_2343',
]


# PTB has a (virtual) fullstop after sentence-final abbreviation, eg.
# he went to the U.S.
_PTB_EXTRA_FULLSTOPS = [
    ('06/wsj_0617.mrg', 966),
    ('06/wsj_0695.mrg', 64),
    ('07/wsj_0764.mrg', 882),  # aka file1
    ('11/wsj_1101.mrg', 736),
    ('11/wsj_1125.mrg', 222),
    ('13/wsj_1318.mrg', 212),
    ('13/wsj_1377.mrg', 790),
    ('13/wsj_1390.mrg', 320),
    ('19/wsj_1988.mrg', 476),
    ('23/wsj_2303.mrg', 301),
    ('23/wsj_2320.mrg', 85),
    ('23/wsj_2320.mrg', 643),
    ('23/wsj_2321.mrg', 46),
    ('23/wsj_2398.mrg', 559)
]


# these specific fileid, token number combinations are skipped or rewritten
# (prefix, subst)
_PTB_SUBSTS_OTHER = {
    # * bad (fileX) files
    # file1
    ('07/wsj_0764.mrg', 981): (None, ""),  # token in PTB missing from RST-WSJ
    ('07/wsj_0764.mrg', 982): (None, ""),  # token in PTB missing from RST-WSJ
    ('07/wsj_0764.mrg', 983): (None, ""),  # token in PTB missing from RST-WSJ
    # file2
    ('04/wsj_0430.mrg', 413): (None, ","),  # '.' in PTB, ',' in RST-WSJ
    # file3
    ('07/wsj_0766.mrg', 111): (None, "&amp;"),  # & in PTB, &amp; in RST-WSJ
    ('07/wsj_0766.mrg', 1836): (None, ""),  # token in PTB missing from RST-WSJ
    ('07/wsj_0766.mrg', 1839): (None, ""),  # token in PTB missing from RST-WSJ
    # file5
    ('21/wsj_2172.mrg', 113): (None, "``"),  # `` in PTB, `` too in RST-WSJ
    # where it is usually " in RST-WSJ
    ('21/wsj_2172.mrg', 177): (None, "among analysts"),  # 2nd token in
    # RST-WSJ missing from PTB
    ('21/wsj_2172.mrg', 359): (None, "17"),  # 5 in PTB, 17 in RST-WSJ
    ('21/wsj_2172.mrg', 439): (None, "&amp;"),  # & in PTB, &amp; in RST-WSJ
    ('21/wsj_2172.mrg', 742): (None, "3.00"),  # 2 in PTB, 3.00 in RST-WSJ
    ('21/wsj_2172.mrg', 759): (None, "17"),  # 5 in PTB, 17 in RST-WSJ
    ('21/wsj_2172.mrg', 1001): (None, "&amp;"),  # & in PTB, &amp; in RST-WSJ
    ('21/wsj_2172.mrg', 1250): (None, ""),  # token in PTB missing from RST-WSJ
    ('21/wsj_2172.mrg', 1280): (None, ""),  # token in PTB missing from RST-WSJ
    # * regular (wsj_XXXX.out) files
    ('06/wsj_0675.mrg', 546): ("-", None),  # --
    ('11/wsj_1139.mrg', 582): ("> ", None),  # insertion
    ('11/wsj_1161.mrg', 845): ("<", None),  # insertion
    ('11/wsj_1171.mrg', 207): (None, "'"),   # backtick
    ('13/wsj_1303.mrg', 388): (None, ""),  # extra full stop
    ('13/wsj_1331.mrg', 930): (None, "`S"),
    ('13/wsj_1367.mrg', 364): ("-- ", None),  # insertion
    ('13/wsj_1377.mrg', 4): (None, "")
}

_PTB_SUBSTS = dict(
    [(_k, (None, "")) for _k in _PTB_EXTRA_FULLSTOPS] +
    list(_PTB_SUBSTS_OTHER.items())
)


def _tweak_token(ptb_name):
    """
    Return a function that normalises a token, sometimes including horribly
    specific one-off changes for one-off errors.

    :rtype: (string, string) -> TweakedToken
    """

    slash_re = re.compile(r'\\/')
    star_re = re.compile(r'\\\*')

    def _norm(toknum, tagged_token):
        "tweak a token to match RST_DT text"

        word, tag = tagged_token
        if (ptb_name, toknum) in _PTB_SUBSTS:
            prefix, tweak = _PTB_SUBSTS[(ptb_name, toknum)]
            return TweakedToken(word, tag, tweak, prefix)
        elif is_empty_category(tag) and is_nonword_token(word):
            return TweakedToken(word, tag, "")

        tweak = PTB_TO_TEXT.get(word, word)
        tweak = slash_re.sub('/', tweak)
        tweak = star_re.sub('*', tweak)
        tweak = None if tweak == word else tweak
        return TweakedToken(word, tag, tweak)

    return _norm


def _mk_token(ttoken, span):
    """
    Convert a tweaked token and the span it's been aligned with
    into a proper Token object.
    """
    if ttoken.offset != 0:
        span = Span(span.char_start + ttoken.offset, span.char_end)
    return Token(ttoken, span)


class PtbParser(object):
    """Gold parser that gets annotations from the PTB.

    It uses an instantiated NLTK BracketedParseCorpusReader
    for the PTB section relevant to the RST DT corpus.

    Note that the path you give to this will probably end with
    something like `parsed/mrg/wsj`
    """

    def __init__(self, corpus_dir):
        """ """
        self.reader = BracketParseCorpusReader(corpus_dir,
                                               r'../wsj_.*\.mrg',
                                               encoding='ascii')

    def tokenize(self, doc):
        """Tokenize the document text using the PTB gold annotation.

        Parameters
        ----------
        doc: DocumentPlus
            Rich representation of the document.

        Returns
        -------
        doc: DocumentPlus
            Rich representation of the document, with tokenization.
        """
        # get tokens from PTB
        ptb_name = _guess_ptb_name(doc.key)
        if ptb_name is None:
            return doc

        # get doc text
        # here we cheat and get it from the RST-DT tree
        # was: rst_text = doc.orig_rsttree.text()
        rst_text = doc.text
        tagged_tokens = self.reader.tagged_words(ptb_name)
        # tweak tokens THEN filter empty nodes
        tweaked1, tweaked2 =\
            itertools.tee(_tweak_token(ptb_name)(i, tok) for i, tok in
                          enumerate(tagged_tokens)
                          if not is_empty_category(tok[1]))
        spans = generic_token_spans(rst_text, tweaked1,
                                    txtfn=lambda x: x.tweaked_word)
        result = [_mk_token(t, s) for t, s in izip(tweaked2, spans)]

        # store in doc
        doc.set_tokens(result)

        return doc

    def parse(self, doc):
        """Parse a document, using the gold PTB annotation.

        Given a document, return a list of educified PTB parse trees
        (one per sentence).

        These are almost the same as the trees that would be returned by the
        `parsed_sents` method, except that each leaf/node is
        associated with a span within the RST DT text.

        Note: does nothing if there is no associated PTB corpus entry.

        Parameters
        ----------
        doc: DocumentPlus
            Rich representation of the document.

        Returns
        -------
        doc: DocumentPlus
            Rich representation of the document, with syntactic
            constituency trees.
        """
        # get PTB trees
        ptb_name = _guess_ptb_name(doc.key)
        if ptb_name is None:
            return doc

        # get tokens from tokenized document
        # FIXME alignment/reconstruction should never have to deal
        # with the left padding token in the first place
        doc_tokens = doc.tkd_tokens[1:]  # skip left padding token
        tokens_iter = iter(doc_tokens)

        trees = []
        lex_heads = []
        for tree in self.reader.parsed_sents(ptb_name):
            # apply standard cleaning to tree
            # strip function tags, remove empty nodes
            tree_no_empty = prune_tree(tree, is_non_empty)
            tree_no_empty_no_gf = transform_tree(tree_no_empty,
                                                 strip_subcategory)
            #
            leaves = tree_no_empty_no_gf.leaves()
            tslice = itertools.islice(tokens_iter, len(leaves))
            clean_tree = ConstituencyTree.build(tree_no_empty_no_gf,
                                                tslice)
            trees.append(clean_tree)

            # lexicalize the PTB tree: find the head word of each constituent
            # constituents and their heads are designated by their Gorn address
            # ("tree position" in NLTK) in the tree
            lheads = find_lexical_heads(clean_tree)
            lex_heads.append(lheads)

        # store trees in doc
        doc.set_syn_ctrees(trees, lex_heads=lex_heads)

        return doc


# FIXME refactor, maybe move to a better place:
# educe.rst_dt.annotation ? educe.external.parser ?
# none of this code is specific to the PTB corpus itself, only to
# NLTK-style syntactic trees
def align_edus_with_sentences(edus, syn_trees, strict=False):
    """Map each EDU to its sentence.

    If an EDU span overlaps with more than one sentence span, the
    sentence with maximal overlap is chosen.

    Parameters
    ----------
    edus: list(EDU)
        List of EDUs.

    syn_trees: list(Tree)
        List of syntactic trees, one per sentence.

    strict: boolean, default False
        If True, raise an error if an EDU does not map to exactly
        one sentence.

    Returns
    -------
    edu2sent: list(int or None)
        Map from EDU to (0-based) sentence index or None.

    TODO
    ----
    * [ ] rewrite a faster version using numpy
    """
    edu2sent = []
    for edu in edus:
        # find the syntactic trees that overlap with this EDU
        tree_idcs = [t_idx
                     for t_idx, tree in enumerate(syn_trees)
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
                emsg = ('Segmentation mismatch:'
                        'one EDU, more than one PTB tree')
                print(edu)
                ptrees = [syn_trees[t_idx] for t_idx in tree_idcs]
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
        # append the computed index
        edu2sent.append(tree_idx)

    return edu2sent
