# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Alignment with the Penn Treebank
"""

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
from educe.external.parser import\
    ConstituencyTree
from educe.external.postag import\
    generic_token_spans, Token
from educe.internalutil import izip
from educe.ptb.annotation import\
    PTB_TO_TEXT, is_nonword_token, TweakedToken,\
    transform_tree, strip_subcategory, prune_tree, is_non_empty,\
    is_empty_category


def _guess_ptb_name(k):
    """
    Given an RST DT corpus key, guess the equivalent filename from the
    Penn Tree Bank.

    Return None if the name doesn't look like it has an equivalent
    (note that returning something is not a guarantee either)
    """
    bname = fp.splitext(fp.basename(k.doc))[0]
    nparts = bname.split("_")
    if len(nparts) > 1:
        section = nparts[1][:2]  # wsj_2431 => 24
        return fp.join(section, bname + ".mrg")
    else:
        return None


# PTB has a (virtual) fullstop after sentence-final abbreviation, eg.
# he went to the U.S.
_PTB_EXTRA_FULLSTOPS =\
    [('06/wsj_0617.mrg', 966),
     ('06/wsj_0695.mrg', 64),
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
     ('23/wsj_2398.mrg', 559)]


# these specific fileid, token number combinations are skipped or rewritten
# (prefix, subst)
_PTB_SUBSTS_OTHER =\
    {('06/wsj_0675.mrg', 546): ("-", None),  # --
     ('11/wsj_1139.mrg', 582): (">", None),  # insertion
     ('11/wsj_1161.mrg', 845): ("<", None),  # insertion
     ('11/wsj_1171.mrg', 207): (None, "'"),   # backtick
     ('13/wsj_1303.mrg', 388): (None, ""),  # extra full stop
     ('13/wsj_1331.mrg', 930): (None, "`S"),
     ('13/wsj_1367.mrg', 364): ("--", None),  # insertion
     ('13/wsj_1377.mrg', 4): (None, "")}

_PTB_SUBSTS = dict([(_k, (None, "")) for _k in _PTB_EXTRA_FULLSTOPS] +
                   list(_PTB_SUBSTS_OTHER.items()))


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
    span = span if ttoken.offset == 0 else\
        Span(span.char_start + ttoken.offset, span.char_end)
    return Token(ttoken, span)


def align(corpus, k, ptb):
    """
    Align PTB annotations to the corpus raw text.
    Return a generator of `Token` objects

    Note: returns None if there is no associated PTB corpus entry.

    See also `parse_tree` (which calls this function internall)
    """
    ptb_name = _guess_ptb_name(k)
    if ptb_name is None:
        return None
    rst_text = corpus[k].text()
    tagged_tokens = ptb.tagged_words(ptb_name)
    # tweak tokens THEN filter empty nodes
    tweaked1, tweaked2 =\
        itertools.tee(_tweak_token(ptb_name)(i, tok) for i, tok in
                      enumerate(tagged_tokens)
                      if not is_empty_category(tok[1]))
    spans = generic_token_spans(rst_text, tweaked1,
                                txtfn=lambda x: x.tweaked_word)
    return (_mk_token(t, s) for t, s in izip(tweaked2, spans))


def parse_trees(corpus, k, ptb):
    """
    Given an RST DT tree and an NLTK PTB reader, return a list of
    educified PTB parse trees (one per sentence). These are
    almost the same as the trees that would be returned by the
    `parsed_sents` method, except that each leaf/node is
    associated with a span within the RST DT text.

    Note: returns None if there is no associated PTB corpus entry.
    """
    ptb_name = _guess_ptb_name(k)
    if ptb_name is None:
        return None
    tokens_iter = align(corpus, k, ptb)

    results = []
    for tree in ptb.parsed_sents(ptb_name):
        # apply standard cleaning to tree
        # strip function tags, remove empty nodes
        tree_no_empty = prune_tree(tree, is_non_empty)
        tree_no_empty_no_gf = transform_tree(tree_no_empty,
                                             strip_subcategory)
        #
        leaves = tree_no_empty_no_gf.leaves()
        tslice = itertools.islice(tokens_iter, len(leaves))
        results.append(ConstituencyTree.build(tree_no_empty_no_gf,
                                              tslice))
    return results


def reader(corpus_dir):
    """
    An instantiated NLTK BracketedParseCorpusReader for the PTB
    section relevant to the RST DT corpus.

    Note that the path you give to this will probably end with
    something like `parsed/mrg/wsj`
    """
    return BracketParseCorpusReader(corpus_dir,
                                    r'../wsj_.*\.mrg',
                                    encoding='ascii')
