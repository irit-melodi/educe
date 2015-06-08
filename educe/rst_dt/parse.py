#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Philippe Muller, Eric Kow
#
"""
From RST discourse treebank trees to Educe-style objects
(reading the format from Di Eugenio's corpus of instructional texts).

The main classes of interest are `RSTTree` and `EDU`.  `RSTTree` can be treated
as an NLTK Tree structure.  It is also an educe `Standoff` object, which means
that it points to other RST trees (their children) or to `EDU`.
"""

import collections
import copy
import re
import codecs
from nltk import Tree

from educe.annotation import Span
from .annotation import\
    RSTTreeException,\
    EDU, Node,\
    RSTContext, RSTTree, SimpleRSTTree
from .rst_wsj_corpus import load_rst_wsj_corpus_text_file
from ..external.postag import generic_token_spans
from ..internalutil import treenode


# pre-processing leaves
_TEXT_RE = re.compile(r"\(text (?P<text>.+(</EDU>|</s>|_!))\)")

# pre-processing heads
_TYPE_RE = r"\((?P<type>(Nucleus|Satellite))"
_ROOT_TYPE_RE = r"\((?P<type>(Root))"
_SPAN_RE = r"\((?P<span>(leaf [0-9]+|span [0-9]+ [0-9]+))\)"
_REL_RE = r"\((?P<rel>rel2par [\-A-Za-z0-9:]+)\)"
_PARA_PATTERN = re.compile(r"<P>")
_HEAD_PATTERN = re.compile("%s %s %s" % (_TYPE_RE, _SPAN_RE, _REL_RE))
_ROOT_PATTERN = re.compile("%s %s" % (_ROOT_TYPE_RE, _SPAN_RE))

# parsing
_LEAF_PATTERN = r"\[[^\]]+\]"  # non-']' chars in square brackets


def _process_text(matchobj):
    """
    Helper function, ignore
    """
    text = matchobj.group("text")
    return "[%s]" % text
    #.replace(" ","¤")


def _process_head(matchobj):
    """
    Helper function, ignore
    """
    ntype = matchobj.group("type")
    span = matchobj.group("span").replace(" ", "-")
    rel = "---" if ntype == "Root" else matchobj.group("rel").split()[1]
    return "(%s|%s|%s" % (ntype, span, rel)


def _mark_leaves(tstr):
    """
    Helper function, ignore
    """
    return _TEXT_RE.sub(_process_text, tstr)


def _mark_heads(tstr):
    """
    Helper function, ignore
    """
    hstr = _ROOT_PATTERN.sub(_process_head, tstr)
    return _HEAD_PATTERN.sub(_process_head, hstr)


def _parse_edu(descr, edu_start, start=0):
    """
    Parse an RST DT leaf string
    """
    sdesc = descr.strip()
    if sdesc.startswith("<s>"):
        sdesc = sdesc[3:]
    if sdesc.endswith("</s>"):
        sdesc = sdesc[:-4]

    if sdesc.startswith("<EDU>") and sdesc.endswith("</EDU>"):
        text = sdesc[5:-6]  # remove <EDU></EDU> mark
    elif sdesc.startswith("_!") and sdesc.endswith("_!"):
        text = sdesc[2:-2]
    else:
        text = sdesc

    end = start + len(text)
    span = Span(start, end)  # text-span (not the same as EDU span)
    return EDU(edu_start, span, text)


def _parse_node(descr, span):
    """
    Build a node from a piece of an RST DT tree
    """
    nuclearity, edu_span, rel = descr.split("|")
    _edu_span = edu_span.split("-")[1:]
    if len(_edu_span) == 1:
        _edu_span.append(_edu_span[0])
    edu_span = (int(_edu_span[0]),
                int(_edu_span[1]))
    return Node(nuclearity, edu_span, span, rel)


def _preprocess(tstr):
    """
    Helper function: Given a raw RST treebank string, return a massaged
    representation for easier parsing, along with its first/last EDU number.
    """
    res = tstr.strip()
    res = _mark_leaves(res)
    res = re.sub(r"\(\s+", "(", res)  # .replace("\n",""))
    res = re.sub(r"\s+\)", ")", res)
    res = re.sub(r"\s\s+", " ", res)
    res = _mark_heads(res)
    return res


def _tree_span(tree):
    """
    Span for the current node or leaf in the tree
    """
    return treenode(tree).span if isinstance(tree, Tree)\
        else tree.span


def _postprocess(tree, start=0, edu_start=1):
    """
    Helper function: Convert the NLTK-parsed representation of an RST tree
    to one using educe-style Standoff objects
    """
    if isinstance(tree, Tree):
        children = []
        position = start - 1  # compensate for virtual whitespace added below
        node = _parse_node(treenode(tree), Span(-1, -1))
        edu_start2 = node.edu_span[0]

        for child_ in tree:
            # (NB: +1 to add virtual whitespace between EDUs)
            child = _postprocess(child_, position + 1, edu_start2)
            children.append(child)
            # pylint: disable=E1101
            child_sp = _tree_span(child)
            # pylint: enable=E1101
            position = child_sp.char_end

        node.span = Span(start, position)
        return RSTTree(node, children)
    else:
        if tree.startswith("["):
            return _parse_edu(tree[1:-1], edu_start, start)
        else:
            raise RSTTreeException("ERROR in rst tree format for leaf : ",
                                   child)


def _recompute_spans(tree, context):
    """
    Recalculate tree node spans from the bottom up
    (helper for _align_with_context)
    """
    if isinstance(tree, Tree):
        spans = []
        for child in tree:
            _recompute_spans(child, context)
            spans.append(_tree_span(child))
        treenode(tree).span = Span.merge_all(spans)
        treenode(tree).context = context


def _align_with_context(tree, context):
    """
    Update a freshly parsed RST DT tree with proper standoff
    annotations pointing to its base text
    """
    leaves = tree.leaves()

    leaf_tokens = [_PARA_PATTERN.sub("\n\n", l.raw_text.strip())
                   for l in leaves]
    spans = generic_token_spans(context.text(),
                                leaf_tokens)
    for edu, span in zip(leaves, spans):
        edu.span = span
        edu.set_context(context)
    _recompute_spans(tree, context)
    return tree


def parse_rst_dt_tree(tstr, context=None):
    """
    Read a single RST tree from its RST DT string representation.
    If context is set, align the tree with it. You should really
    try to pass in a context (see `RSTContext` if you can, the
    None case is really intended for testing, or in cases where
    you don't have an original text)
    """
    pstr = _preprocess(tstr)
    tree_ = Tree.fromstring(pstr, leaf_pattern=_LEAF_PATTERN)
    tree_ = _postprocess(tree_)
    if context:
        tree_ = _align_with_context(tree_, context)
    return tree_


def read_annotation_file(anno_filename, text_filename):
    """
    Read a single RST tree
    """
    tree = None

    # read text file
    text, sents, paras = load_rst_wsj_corpus_text_file(text_filename)
    # use it as context for the RST tree
    context = RSTContext(text, sents, paras)
    # read RST tree
    with codecs.open(anno_filename, 'r', 'utf-8') as stream:
        tree = parse_rst_dt_tree(stream.read(), context)

    return tree


def parse_lightweight_tree(tstr):
    """
    Parse lightweight RST debug syntax into SimpleRSTTree, eg. ::

        (R:attribution
           (N:elaboration (N foo) (S bar)
           (S quux)))

    This is motly useful for debugging or for knocking out quick
    examples
    """
    _lw_type_re = re.compile(r'(?P<nuc>[RSN])(:(?P<rel>.*)|$)')
    _lw_nuc_map = dict((nuc[0], nuc)
                       for nuc in ["Root", "Nucleus", "Satellite"])
    # pylint: disable=C0103
    PosInfo = collections.namedtuple("PosInfo", "text edu")
    # pylint: enable=C0103

    def walk(subtree, posinfo=PosInfo(text=0, edu=0)):
        """
        walk down first-cut tree, counting span info and returning a
        fancier tree along the way
        """
        if isinstance(subtree, Tree):
            start = copy.copy(posinfo)
            children = []
            for kid in subtree:
                tree, posinfo = walk(kid, posinfo)
                children.append(tree)

            match = _lw_type_re.match(treenode(subtree))
            if not match:
                raise RSTTreeException("Missing nuclearity annotation in ",
                                       subtree)
            nuclearity = _lw_nuc_map[match.group("nuc")]
            rel = match.group("rel") or "leaf"
            edu_span = (start.edu, posinfo.edu - 1)
            span = Span(start.text, posinfo.text)
            node = Node(nuclearity, edu_span, span, rel)
            return SimpleRSTTree(node, children), posinfo
        else:
            text = subtree
            start = posinfo.text
            end = start + len(text)
            posinfo2 = PosInfo(text=end, edu=posinfo.edu+1)
            return EDU(posinfo.edu, Span(start, end), text), posinfo2

    return walk(Tree.fromstring(tstr))[0]
