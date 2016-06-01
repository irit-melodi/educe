"""
Educe representation of Penn Tree Bank annotations.

We actually just use the token and constituency tree representations
from `educe.external.postag` and `educe.external.parse`, but included
here are tools that can also be used to align the PTB with other
corpora based off the same text (eg. the RST Discourse Treebank)
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

import re

from nltk.tree import Tree

from educe.annotation import Span
from educe.external.postag import RawToken


PTB_TO_TEXT = {"``": "\"",
               "''": "\"",
               "-LRB-": "(",
               "-RRB-": ")",
               "-LSB-": "[",
               "-RSB-": "]",
               "-LCB-": "{",
               "-RCB-": "}"}
"""
Straight substitutions you can use to replace some PTB-isms
with their likely original text
"""


# prefixes for things we can skip
_SKIP_RE = re.compile(r'^(' +
                      r'(\*((T|ICH|EXP|RNR|PPA)\*)?-\d*)' +
                      r'|0|\*' +
                      r'|(\*(U|\?|NOT)\*)' +
                      r')$')


def is_nonword_token(text):
    """
    True if the text appears to correspond to some kind of non-textual
    token, for example, `*T*-1` for some kind of trace. These seem to
    only appear with tokens tagged `-NONE-`.
    """
    return bool(_SKIP_RE.match(text))


def is_empty_category(postag):
    """True if postag is the empty category, i.e. `-NONE-` in the PTB."""
    return postag == '-NONE-'


# utility functions to work on sequences of tokens
PUNC_POSTAGS = set([
    '``', "''",  # double quotes
    ',',
    ':',
    '.',  # strong punctuations
])


def strip_punctuation(tokens):
    """Strip leading and trailing punctuation from a sequence of tokens.

    Parameters
    ----------
    tokens: list of Token
        Sequence of tokens.

    Returns
    -------
    tokens_strip: list of Token
        Corresponding list of tokens with no leading or trailing
        punctuation.
    """
    nopunc_tokens = [t.tag not in PUNC_POSTAGS for t in tokens]
    nopunc_lmost = nopunc_tokens.index(True)
    nopunc_rmost = len(nopunc_tokens) - 1 - nopunc_tokens[::-1].index(True)
    tokens_strip = tokens[nopunc_lmost:nopunc_rmost + 1]
    return tokens_strip


# pylint: disable=too-few-public-methods
class TweakedToken(RawToken):
    """
    A token with word, part of speech, plus "tweaked word" (what the
    token should be treated as when aligning with corpus), and offset
    (some tokens should skip parts of the text)

    This intermediary class should only be used within the educe library
    itself. The context is that we sometimes want to align PTB
    annotations (see `educe.external.postag.generic_token_spans`)
    against text which is almost but not quite identical to
    the text that PTB annotations seem to represent. For example, the
    source text might have sentences that end in abbreviations, like
    "He moved to the U.S." and the PTB might annotation an extra full
    stop after this for an end-of-sentence marker. To deal with these,
    we use wrapped tokens to allow for some manual substitutions:

    * you could "delete" a token by assigning it an empty tweaked word
      (it would then be assigned a zero-length span)
    * you could skip some part of the text by supplying a prefix
      (this expands the tweaked word, and introduces an offset which
      you can subsequentnly use to adjust the detected token span)
    * or you could just replace the token text outright

    These tweaked tokens are only used to obtain a span within the text
    you are trying to align against; they can be subsequently discarded.
    """

    def __init__(self, word, tag, tweaked_word=None, prefix=None):
        tweak = word if tweaked_word is None else tweaked_word
        if prefix is None:
            offset = 0
        else:
            tweak = prefix + tweak
            offset = len(prefix)
        self.tweaked_word = tweak
        self.offset = offset
        super(TweakedToken, self).__init__(word, tag)

    def __str__(self):
        return unicode(self)

    def __unicode__(self):
        res = self.word
        if self.tweaked_word != self.word:
            res += " [%s]" % self.tweaked_word
        res += "/%s" % self.tag
        if self.offset != 0:
            res += " (%d)" % self.offset
        return res
# pylint: enable=too-few-public-methods


#
# TreebankLanguagePack (after edu.stanford.nlp.trees)
#

# label annotation introducing characters
_LAIC = [
    '-',  # function tags, identity index, reference index
    '=',  # gap co-indexing
]


# pylint: disable=invalid-name
def post_basic_category_index(label):
    """Get the index of the first char after the basic label.

    This should never match the first char of the label ;
    if the first char is such a char, then a matched char is also
    not used iff there is something in between, e.g.
    (-LRB- => -LRB-) but (--PU => -).
    """
    first_char = ''
    for i, c in enumerate(label):
        if c in _LAIC:
            if i == 0:
                first_char = c
            elif first_char and (i > 1) and (c == first_char):
                first_char = ''
            else:
                break
    else:
        i += 1
    return i
# pylint: enable=invalid-name


def basic_category(label):
    """Get the basic syntactic category of a label.

    This is done by truncating whatever comes after a
    (non-word-initial) occurrence of one of the
    label_annotation_introducing_characters().
    """
    return label[0:post_basic_category_index(label)] if label else label


# Reimplementation of most of the most standard parser parameters for the PTB
# ref: edu.stanford.nlp.parser.lexparser.EnglishTreebankParserParams

# pylint: disable=invalid-name
def strip_subcategory(tree,
                      retain_TMP_subcategories=False,
                      retain_NPTMP_subcategories=False):
    """Transform tree to strip additional label annotation at each node"""
    if not isinstance(tree, Tree):
        return tree

    label = tree.label()
    if retain_TMP_subcategories and ('-TMP' in label):
        label = '{bc}-TMP'.format(bc=basic_category(label))
    elif retain_NPTMP_subcategories and label.startswith('NP-TMP'):
        label = 'NP-TMP'
    else:
        label = basic_category(label)
    tree.set_label(label)
    return tree
# pylint: enable=invalid-name


# ref: edu.stanford.nlp.trees.BobChrisTreeNormalizer
def is_non_empty(tree):
    """Filter (return False for) nodes that cover a totally empty span."""
    # always keep leaves
    if not isinstance(tree, Tree):
        return True

    label = tree.label()
    if is_empty_category(label):
        # check this is a pre-terminal ; probably superfluous
        assert ((len(tree) == 1) and
                (not isinstance(tree[0], Tree)))
        return False
    else:
        return True


def prune_tree(tree, filter_func):
    """Prune a tree by applying filter_func recursively.

    All children of filtered nodes are pruned as well.
    Nodes whose children have all been pruned are pruned too.

    The filter function must be applicable to Tree but also non-Tree,
    as are leaves in an NLTK Tree.
    """
    # prune node if filter returns False
    if not filter_func(tree):
        return None

    if isinstance(tree, Tree):
        # recurse
        new_kids = [new_kid
                    for new_kid in (prune_tree(kid, filter_func)
                                    for kid in tree)
                    if new_kid is not None]
        # prune node if it had children and lost them all
        if tree and not new_kids:
            return None
        # return new node, pruned
        return Tree(tree.label(), new_kids)
    else:
        # return leaf unchanged
        return tree


# TODO: see if we can partly use nltk.treetransforms
def transform_tree(tree, transformer):
    """Transform a tree by applying a transformer at each level.

    The tree is traversed depth-first, left-to-right, and the
    transformer is applied at each node.
    """
    # recurse
    if isinstance(tree, Tree):
        new_kids = [new_kid
                    for new_kid in (transform_tree(kid, transformer)
                                    for kid in tree)
                    if new_kid is not None]
        new_tree = (type(tree)(tree.label(), new_kids)
                    if new_kids else None)
    else:
        new_tree = tree
    # apply to current node
    return transformer(new_tree)
# end of TreebankLanguagePack


# maybe this belongs to educe.external.parse ?
# it operates on an NLTK-style constituency tree
def syntactic_node_seq(ptree, tokens):
    """Find the sequence of syntactic nodes covering a sequence of tokens.

    Parameters
    ----------
    ptree: `nltk.tree.Tree`
        Syntactic tree.
    tokens: sequence of `Token`
        Sequence of tokens under scrutiny.

    Returns
    -------
    syn_nodes: list of `nltk.tree.Tree`
        Spanning sequence of nodes of the syntactic tree.
    """
    txt_span = Span(tokens[0].text_span().char_start,
                    tokens[-1].text_span().char_end)

    for tpos in ptree.treepositions():
        node = ptree[tpos]
        # skip nodes whose span does not enclose txt_span
        node_txt_span = node.text_span()
        if not node_txt_span.encloses(txt_span):
            continue

        # * spanning node
        if node.text_span() == txt_span:
            return [node]

        # * otherwise: spanning subsequence of kid nodes
        if not isinstance(node, Tree):
            continue
        txt_span_start = txt_span.char_start
        txt_span_end = txt_span.char_end
        kids_start = [x.text_span().char_start for x in node]
        kids_end = [x.text_span().char_end for x in node]
        try:
            idx_left = kids_start.index(txt_span_start)
        except ValueError:
            continue
        try:
            idx_right = kids_end.index(txt_span_end)
        except ValueError:
            continue
        if idx_left == idx_right:
            continue
        return [x for x in node[idx_left:idx_right + 1]]
    else:
        return []
