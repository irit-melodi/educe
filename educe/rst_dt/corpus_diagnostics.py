"""This module is a loose collection of diagnostic functions on an RST corpus.

"""

from __future__ import print_function

from collections import Counter, defaultdict
import os

import pandas as pd

from educe.internalutil import treenode
from educe.rst_dt.annotation import (RSTTree, _binarize)
from educe.rst_dt.corpus import (Reader as RstReader,
                                 RstRelationConverter,
                                 RELMAP_112_18_FILE)
# imports for dirty stuff (mostly)
import itertools

from nltk import Tree
from nltk.corpus.reader import BracketParseCorpusReader

from educe.annotation import Span
from educe.external.parser import ConstituencyTree
from educe.internalutil import izip
from educe.rst_dt.document_plus import align_edus_with_paragraphs
from educe.rst_dt.ptb import (_guess_ptb_name, _tweak_token,
                              is_empty_category, generic_token_spans,
                              _mk_token, align_edus_with_sentences)
from educe.ptb.annotation import (prune_tree, is_non_empty, transform_tree,
                                  strip_subcategory)
from educe.ptb.head_finder import find_lexical_heads
# end imports for dirty stuff


# RST corpus
# TODO import CORPUS_DIR/CD_TRAIN e.g. from educe.rst_dt.rst_wsj_corpus
CORPUS_DIR = os.path.join(os.path.dirname(__file__),
                          '..', '..', '..', '..', 'corpora',
                          'rst_discourse_treebank', 'data',
                          'RSTtrees-WSJ-main-1.0')
CD_TRAIN = os.path.join(CORPUS_DIR, 'TRAINING')
# relation converter (fine- to coarse-grained labels)
REL_CONV = RstRelationConverter(RELMAP_112_18_FILE).convert_tree


def load_training_as_dataframe():
    """Load training section of the RST-WSJ corpus as a pandas.DataFrame.

    Returns
    -------
    df: pandas.DataFrame
        DataFrame of all instances of relations in the training section.
        Interesting columns are 'rel', 'nuc_sig', 'arity'
    """
    rst_phrases = []  # list of rows, each represented as a dict

    rst_reader = RstReader(CD_TRAIN)
    rst_corpus = rst_reader.slurp()
    for doc_id, rtree_ref in sorted(rst_corpus.items()):
        # convert labels to coarse
        coarse_rtree_ref = REL_CONV(rtree_ref)
        # store "same-unit" subtrees
        heterogeneous_nodes = []
        internal_nodes = lambda t: isinstance(t, RSTTree) and len(t) > 1
        for su_subtree in coarse_rtree_ref.subtrees(filter=internal_nodes):
            # get each kid's relation
            kid_rels = tuple(treenode(kid).rel for kid in su_subtree)
            # filter out nodes whose kids have different relations
            rels = [r for r in set(kid_rels) if r != 'span']
            if len(rels) > 1:
                heterogeneous_nodes.append(kid_rels)
                continue

            # process homogeneous nodes
            res = dict()
            rel = rels[0]
            res['rel'] = rel
            # arity
            res['arity'] = len(su_subtree)  # number of kids
            # nuclearity signature
            kid_nucs = tuple(treenode(kid).nuclearity for kid in su_subtree)
            nuc_sig = ''.join('S' if kid_nuc == 'Satellite' else 'N'
                              for kid_nuc in kid_nucs)
            res['nuc_sig'] = (nuc_sig if nuc_sig in frozenset(['SN', 'NS'])
                              else 'NN')
            # TODO len(kid_rels) - 1 is the nb of bin rels

            # height
            rel_hgt = su_subtree.height()
            res['height'] = rel_hgt

            # TODO disc relations of the grandchildren
            #

            rst_phrases.append(res)

    # turn into a DataFrame
    df = pd.DataFrame(rst_phrases)
    # add calculated columns
    # * "undirected" nuclearity, e.g. NS == SN
    df['unuc_sig'] = map(lambda nuc_sig: ('NS' if nuc_sig in ['NS', 'SN']
                                          else 'NN'),
                         df.nuc_sig)
    return df


def get_most_frequent_unuc(df, verbose=False):
    """Get the most frequent undirected nuclearity for each relation.

    Parameters
    ----------
    df: pandas.DataFrame
        Relations present in the corpus

    verbose: boolean, default: False
        Trigger textual traces

    Returns
    -------
    most_freq_unuc: dict(str, str)
        Map each relation to its most frequent (aka mode) undirected
        nuclearity signature.
    """
    # get number of occurrences of each relation
    # * "directed" nuclearity, e.g. NS != SN
    # TODO check and try pandas' hierarchical indexing ;
    # my understanding is we would get relation as 1st and nuclearity as 2nd
    # levels
    if verbose:
        grouped_rel_nuc = df.groupby(['rel', 'nuc_sig'])
        print('\n'.join('{}: {}'.format(rel_nuc, len(occs))
                        for rel_nuc, occs in grouped_rel_nuc))
    # * "undirected" nuclearity, e.g. NS == SN
    # TODO hierarchical indexing (again)
    if verbose:
        grouped_rel_unuc = df.groupby(['rel', 'unuc_sig'])
        print('\n'.join(
            '{:{}s}\t{}\t{}'.format(
                rel, max(len(r) for r in df.rel.unique()), unuc, len(occs))
            for (rel, unuc), occs in grouped_rel_unuc))

    # use this data to get:
    # * unambiguously mono-nuclear relations
    # * unambiguously multi-nuclear relations
    # * most common nuclearity for the remaining (ambiguous wrt nuclearity)
    # relations
    most_freq_unuc = {rel: occs['unuc_sig'].mode()[0]
                      for rel, occs in df.groupby('rel')}
    if verbose:
        print('\n'.join('{}\t{}'.format(rel, unuc)
                        for rel, unuc in sorted(most_freq_unuc.items())))

    return most_freq_unuc


def check_label_ranks():
    """Examine label and rank of attachment"""
    # FIXME rewrite entirely
    labels_ranks_gold = []  # TODO
    labels_ranks_no_nuc = [(lbl[:-3] if lbl is not None and lbl != 'ROOT'
                            else lbl, rnk)
                           for lbl, rnk in labels_ranks_gold]
    print('\n'.join('{}\t{}\t{}'.format(lbl, rnk, occ)
                    for (lbl, rnk), occ
                    in sorted(Counter(labels_ranks_no_nuc).items())))
    print(sorted(set(lbl for lbl, rnk in labels_ranks_gold)))
    print('labels inc. nuc: {}'.format(
        len(set(lbl for lbl, rnk in labels_ranks_gold))))
    print(('labels inc. rank: {}'.format(
        len(Counter(labels_ranks_no_nuc)))))
    print('\n'.join('{}\t{}\t{}'.format(lbl, rnk, occ)
                    for (lbl, rnk), occ
                    in sorted(Counter(labels_ranks_gold).items())))
    print('labels inc. nuc and rank :', len(Counter(labels_ranks_gold)))
    print('\n\n\n')


# WIP
# dirty, almost copies from educe.rst_dt.ptb.PtbParser...
# TODO go and fix educe.rst_dt.{ptb, corenlp}
PTB_DIR = os.path.join(os.path.dirname(__file__),
                       '..', '..', '..', '..', 'corpora',
                       'PTBIII', 'parsed', 'mrg', 'wsj')
PTB_READER = BracketParseCorpusReader(PTB_DIR,
                                      r'../wsj_.*\.mrg',
                                      encoding='ascii')


def tokenize_doc_ptb(doc_id, doc_text):
    """Dirty PTB tokenizer"""
    ptb_name = _guess_ptb_name(doc_id)
    if ptb_name is None:
        return None

    # get doc text
    # here we cheat and get it from the RST-DT tree
    # was: rst_text = doc.orig_rsttree.text()
    rst_text = doc_text
    tagged_tokens = PTB_READER.tagged_words(ptb_name)
    # tweak tokens THEN filter empty nodes
    tweaked1, tweaked2 =\
        itertools.tee(_tweak_token(ptb_name)(i, tok) for i, tok in
                      enumerate(tagged_tokens)
                      if not is_empty_category(tok[1]))
    spans = generic_token_spans(rst_text, tweaked1,
                                txtfn=lambda x: x.tweaked_word)
    result = [_mk_token(t, s) for t, s in izip(tweaked2, spans)]
    return result


def parse_doc_ptb(doc_id, doc_tkd_toks):
    """Dirty PTB parser"""
    # get PTB trees
    ptb_name = _guess_ptb_name(doc_id)
    if ptb_name is None:
        return None

    # use tweaked tokens
    doc_tokens = doc_tkd_toks
    tokens_iter = iter(doc_tokens)

    trees = []
    lex_heads = []
    for tree in PTB_READER.parsed_sents(ptb_name):
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
    return trees  #, lex_heads


# clean stuff
def load_training_as_dataframe_new(binarize=False):
    """Load training section of the RST-WSJ corpus as a pandas.DataFrame.

    Parameters
    ----------
    binarize: boolean, default: False
        If True, apply right-heavy binarization on RST trees.

    Returns
    -------
    node_df: pandas.DataFrame
        DataFrame of all nodes from the constituency trees.
    rel_df: pandas.DataFrame
        DataFrame of all relations.
    edu_df: pandas.DataFrame
        DataFrame of all EDUs.

    TODO
    ----
    [ ] propose left-heavy binarization
    """
    node_rows = []  # list of dicts, one dict per node
    rel_rows = []  # list of dicts, one dict per relation
    # edu_rows contains pre-EDUs rather than EDUs themselves, but maybe
    # conflating both does no harm
    edu_rows = []  # list of dicts, one dict per EDU
    sent_rows = []  # ibid
    para_rows = []  # ibid

    rst_reader = RstReader(CD_TRAIN)
    rst_corpus = rst_reader.slurp()

    for doc_id, rtree_ref in sorted(rst_corpus.items()):
        doc_ctx = rtree_ref.label().context
        doc_text = doc_ctx.text()
        doc_edus = rtree_ref.leaves()

        # 1. Collect constituency nodes and EDUs from the gold RST trees
        doc_rst_rel_rows = []
        doc_edu_rows = []

        # convert labels to coarse
        coarse_rtree_ref = REL_CONV(rtree_ref)
        # binarize if necessary
        if binarize:
            coarse_rtree_ref = _binarize(coarse_rtree_ref)

        # RST nodes: constituents are either relations or EDUs
        for tpos in coarse_rtree_ref.treepositions():
            node = coarse_rtree_ref[tpos]
            # skip EDUs themselves
            if not isinstance(node, Tree):
                continue

            node_label = node.label()
            node_id = '{}_const{}'.format(
                node.origin.doc,
                '-'.join(str(x) for x in tpos))
            # node
            row = {
                'node_id': node_id,
                # WIP tree position
                'treepos': tpos,
                # char span
                'span_start': node_label.span.char_start,
                'span_end': node_label.span.char_end,
                # nuclearity
                'nuclearity': node_label.nuclearity,
                # relation
                'relation': node_label.rel,
            }
            # add pointer to parent, except for root node
            if tpos:
                parent_tpos = tpos[:-1]
                parent_id = '{}_const{}'.format(
                    node.origin.doc,
                    '-'.join(str(x) for x in parent_tpos))
                row.update({
                    'parent_id': parent_id,
                })

            if len(node) > 1:  # internal node => relation
                row.update({
                    # edu span
                    'edu_span_start': node_label.edu_span[0],
                    'edu_span_end': node_label.edu_span[1],
                    # computed column
                    'edu_span_len': (node_label.edu_span[1] -
                                     node_label.edu_span[0]) + 1,
                    # info on children
                    'arity': len(node),
                    'kids_rel': '+'.join(
                        set(kid.label().rel for kid in node
                            if kid.label().rel != 'span')),
                    'kids_nuc': ''.join(
                        ('N' if kid.label().nuclearity == 'Nucleus' else 'S')
                        for kid in node),
                })
                doc_rst_rel_rows.append(row)
            else:  # pre-EDU
                edu = node[0]
                row.update({
                    'num': edu.num,
                })
                doc_edu_rows.append(row)

        # prepare this info to find "leaky" substructures:
        # sentences and paragraphs
        # dict of EDU spans to constituent node from the RST tree
        rst_tree_node_spans = {
            (row['edu_span_start'], row['edu_span_end']): row['treepos']
            for row in doc_rst_rel_rows
        }
        # list of EDU spans of constituent nodes, sorted by length of span
        # then start
        rst_tree_node_spans_by_len = list(sorted(
            rst_tree_node_spans, key=lambda x: (x[1] - x[0], x[0])))

        # 2. Collect sentences
        doc_sent_rows = []
        # use dirty PTB tokenizer + parser
        doc_tkd_toks = tokenize_doc_ptb(doc_id, doc_text)
        doc_tkd_trees = parse_doc_ptb(doc_id, doc_tkd_toks)

        # sentence <-> EDU mapping and the information that depends on this
        # mapping might be more appropriate as a separate DataFrame
        # align EDUs with sentences
        edu2sent = align_edus_with_sentences(doc_edus, doc_tkd_trees,
                                             strict=False)
        # get the codomain of edu2sent
        # if we want to be strict, we can assert that the codomain is
        # a gapless interval
        # assert sent_idc == list(range(len(doc_tkd_trees)))
        # this assertion is currently known to fail on:
        # * RST-WSJ/TRAINING/wsj_0678.out: wrong sentence segmentation in PTB
        #     (1 sentence is split in 2)
        edu2sent_codom = set([sent_idx for sent_idx in edu2sent
                              if sent_idx is not None])

        # find the index of the first and last EDU of each sentence
        # indices in both lists are offset by 1 to map to real EDU
        # numbering (which is 1-based)
        sent_edu_starts = [(edu2sent.index(i) + 1 if i in edu2sent_codom
                            else None)
                           for i in range(len(doc_tkd_trees))]
        sent_edu_ends = [(len(edu2sent) - 1 - edu2sent[::-1].index(i) + 1
                          if i in edu2sent_codom
                          else None)
                         for i in range(len(doc_tkd_trees))]
        # sentences that don't have their own RST subtree are 'leaky'
        # end of sentence <-> EDU mapping et al.

        # iterate over syntactic trees as proxy for sentences
        for sent_idx, tkd_tree in enumerate(doc_tkd_trees):
            row = {
                # data directly from the sentence segmenter
                'sent_id': '{}_sent{}'.format(doc_id.doc, sent_idx),
                'span_start': tkd_tree.span.char_start,
                'span_end': tkd_tree.span.char_end,
            }
            # sentence <-> EDU mapping dependent data
            # should probably have its own dataframe
            # to better handle disagreement between sentence and EDU
            # segmentation, that translates in the following entries
            # as None for missing data
            if sent_idx in edu2sent_codom:
                row.update({
                    'edu_span_start': sent_edu_starts[sent_idx],
                    'edu_span_end': sent_edu_ends[sent_idx],
                    # computed column
                    'edu_span_len': (sent_edu_ends[sent_idx] -
                                     sent_edu_starts[sent_idx]) + 1,
                })
                # use sentence <-> RST tree alignment
                if row['edu_span_len'] > 1:
                    row.update({
                        'leaky': ((sent_edu_starts[sent_idx],
                                   sent_edu_ends[sent_idx])
                                  not in rst_tree_node_spans),
                    })
                else:
                    row.update({'leaky': False})
                # WIP find for each leaky sentence the smallest RST subtree
                # that covers it
                if row['leaky']:
                    sent_edu_first = sent_edu_starts[sent_idx]
                    sent_edu_last = sent_edu_ends[sent_idx]
                    for edu_span in rst_tree_node_spans_by_len:
                        if (edu_span[0] <= sent_edu_first and
                            sent_edu_last <= edu_span[1]):
                            parent_span = edu_span
                            # WIP get immediate members of parent span
                            parent_tpos = rst_tree_node_spans[parent_span]
                            parent_subtree = coarse_rtree_ref[parent_tpos]
                            member_spans = [m.label().edu_span
                                            for m in parent_subtree
                                            if isinstance(m, Tree)]
                            # leaky types 1 and 2: members of the parent
                            # constituent are "pure" wrt sentence span:
                            # each member is either fully inside or fully
                            # outside the sentence ;
                            # no member straddles either of the sentence
                            # boundaries
                            leaky_type_12 = all(
                                ((sent_edu_first <= mspan[0] and
                                  mspan[1] <= sent_edu_last) or
                                 mspan[1] < sent_edu_first or
                                 mspan[0] > sent_edu_last)
                                for mspan in member_spans)
                            # end WIP immediate members
                            break
                    else:
                        raise ValueError(
                            'No minimal spanning node for {}'.format(row))
                    # add info to row
                    row.update({
                        # parent span, on EDUs
                        'parent_span_start': parent_span[0],
                        'parent_span_end': parent_span[1],
                        # length of parent span, in sentences
                        'parent_span_sent_len': (
                            edu2sent[parent_span[1] - 1] -
                            edu2sent[parent_span[0] - 1] + 1),
                        # distance between the current sentence and the most
                        # remote sentence covered by the parent span,
                        # in sentences
                        'parent_span_sent_dist': (
                            max([(edu2sent[parent_span[1] - 1] - sent_idx),
                                 (sent_idx - edu2sent[parent_span[0] - 1])])),
                        # types of leaky, in the taxonomy of
                        # (van der Vliet et al. 2011)
                        # currently {1,2} vs {3,4}
                        'leaky_type_12': leaky_type_12,
                    })
                else:
                    row.update({
                        'parent_span_start': row['edu_span_start'],
                        'parent_span_end': row['edu_span_end'],
                    })
                # end WIP
            doc_sent_rows.append(row)

        # 3. collect paragraphs
        doc_para_rows = []
        doc_paras = doc_ctx.paragraphs
        doc_text = doc_ctx.text()
        # doc_paras is None when the original text has no explicit marking
        # for paragraphs ; this is true for 'fileX' documents in the RST-WSJ
        # corpus
        if doc_paras is not None:
            # EDU to paragraph mapping
            edu2para = align_edus_with_paragraphs(doc_edus, doc_paras,
                                                  doc_text, strict=False)
            edu2para_codom = set([para_idx for para_idx in edu2para
                                 if para_idx is not None])
            # index of the first and last EDU of each paragraph
            para_edu_starts = [(edu2para.index(i) + 1 if i in edu2para_codom
                                else None)
                               for i in range(len(doc_paras))]
            para_edu_ends = [(len(edu2para) - 1 - edu2para[::-1].index(i) + 1
                              if i in edu2para_codom
                              else None)
                             for i in range(len(doc_paras))]
            # paragraphs that don't have their own RST subtree are "leaky" ;
            # end of paragraph <-> EDU mapping et al.

            # iterate over paragraphs
            for para_idx, para in enumerate(doc_paras):
                # dirty, educe.rst_dt.text.Paragraph should have a span
                para_span = Span(para.sentences[0].span.char_start,
                                 para.sentences[-1].span.char_end)
                # end dirty
                row = {
                    # data directly from the paragraph segmenter
                    'para_id': '{}_para{}'.format(doc_id.doc, para_idx),
                    'span_start': para_span.char_start,
                    'span_end': para_span.char_end,
                }
                # paragraph <-> EDU mapping dependent data
                # should probably have its own dataframe etc.
                if para_idx in edu2para_codom:
                    row.update({
                        'edu_span_start': para_edu_starts[para_idx],
                        'edu_span_end': para_edu_ends[para_idx],
                        # computed column
                        'edu_span_len': (para_edu_ends[para_idx] -
                                         para_edu_starts[para_idx]) + 1,
                    })
                    # use paragraph <-> RST tree alignment
                    if row['edu_span_len'] > 1:  # complex paragraphs only
                        row.update({
                            'leaky': ((para_edu_starts[para_idx],
                                       para_edu_ends[para_idx])
                                      not in rst_tree_node_spans),
                        })
                    else:
                        row.update({'leaky': False})
                    # WIP find for each leaky paragraph the smallest RST subtree
                    # that covers it
                    if row['leaky']:
                        for edu_span in rst_tree_node_spans_by_len:
                            if (edu_span[0] <= para_edu_starts[para_idx] and
                                para_edu_ends[para_idx] <= edu_span[1]):
                                parent_span = edu_span
                                break
                        else:
                            raise ValueError(
                                'No minimal spanning node for {}'.format(row))
                        # add info to row
                        row.update({
                            # parent span, on EDUs
                            'parent_span_start': parent_span[0],
                            'parent_span_end': parent_span[1],
                            # length of parent span, in paragraphs
                            'parent_span_para_len': (
                                edu2para[parent_span[1] - 1] -
                                edu2para[parent_span[0] - 1] + 1),
                            # distance between the current paragraph and the most
                            # remote paragraph covered by the parent span, in
                            # paragraphs
                            'parent_span_para_dist': (
                                max([(edu2para[parent_span[1] - 1] - para_idx),
                                     (para_idx - edu2para[parent_span[0] - 1])])),
                        })
                    else:
                        row.update({
                            'parent_span_start': row['edu_span_start'],
                            'parent_span_end': row['edu_span_end'],
                        })
                    # end WIP
                doc_para_rows.append(row)

        # NB: these are leaky sentences wrt the original constituency
        # trees ; leaky sentences wrt the binarized constituency trees
        # might be different (TODO), similarly for the dependency trees
        # (TODO too) ;
        # I should count them, see if the ~5% Joty mentions are on the
        # original or binarized ctrees, and compare with the number of
        # leaky for deptrees ; I suspect the latter will be much lower...
        # HYPOTHESIS: (some or all?) leaky sentences in ctrees correspond
        # to cases where nodes that are not the head of their sentence
        # have dependents in other sentences
        # this would capture the set (or a subset) of edges that fall
        # outside of the search space for the "iheads" intra/inter
        # strategy

        # add doc entries to corpus entries
        para_rows.extend(doc_para_rows)
        sent_rows.extend(doc_sent_rows)
        rel_rows.extend(doc_rst_rel_rows)
        edu_rows.extend(doc_edu_rows)

    # turn list into a DataFrame
    node_df = pd.DataFrame(node_rows)
    rel_df = pd.DataFrame(rel_rows)
    edu_df = pd.DataFrame(edu_rows)
    sent_df = pd.DataFrame(sent_rows)
    para_df = pd.DataFrame(para_rows)
    # add calculated columns here? (leaky and complex sentences)

    return node_df, rel_df, edu_df, sent_df, para_df


nodes_train, rels_train, edus_train, sents_train, paras_train = load_training_as_dataframe_new(binarize=True)
# print(rels_train)
# as of version 0.17, pandas handles missing boolean values by degrading
# column type to object, which makes boolean selection return true for
# all non-None values ; the best workaround is to explicitly fill boolean
# na values
sents_train = sents_train.fillna(value={'leaky': False})
paras_train = paras_train.fillna(value={'leaky': False})
# exclude 'fileX' documents
if False:
    sents_train = sents_train[~sents_train['sent_id'].str.startswith('file')]
    paras_train = paras_train[~paras_train['para_id'].str.startswith('file')]

# SENTENCES
# complex sentences
complex_sents = sents_train[sents_train.edu_span_len > 1]
print('Complex: {} / {} = {}'.format(
    len(complex_sents), len(sents_train),
    float(len(complex_sents)) / len(sents_train)))

# leaky sentences
# assert there is no sentence which is leaky but not complex
assert sents_train[sents_train.leaky & (sents_train.edu_span_len <= 1)].empty
# proportion of leaky sentences
leaky_sents = sents_train[sents_train.leaky]
# according to (Soricut and Marcu, 2003), p. 2, should be 323/6132 = 5.3%
print('Leaky: {} / {} = {}'.format(
    len(leaky_sents), len(sents_train),
    float(len(leaky_sents)) / len(sents_train)))
# proportion of leaky among complex sentences
print('Leaky | Complex: {} / {} = {}'.format(
    len(leaky_sents), len(complex_sents),
    float(len(leaky_sents)) / len(complex_sents)))
print()

# compare leaky with non-leaky complex sentences: EDU length
print('EDU span length of leaky vs non-leaky complex sentences')
print(complex_sents.groupby('leaky')['edu_span_len'].describe().unstack())
print()

# for each leaky sentence, number of sentences included in the smallest
# RST node that fully covers the leaky sentence
# According to the CODRA paper, as I first understood it, 75% of leaky
# sentences need only their left and right neighboring sentences to form
# a complete span ;
# our counts sensibly differ
# new hypothesis: 75% of leaky sentences can be split so that their EDUs
# + the neighboring sentences form complete spans
if False:
    print(leaky_sents[['parent_span_sent_len', 'parent_span_sent_dist']].describe(
        percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9]))
    print(leaky_sents[(leaky_sents['parent_span_sent_dist'] == 1)].describe())
    print(leaky_sents[(leaky_sents['parent_span_sent_dist'] == 1) &
                      (leaky_sents['parent_span_sent_len'] > 2)])
if False:
    print(leaky_sents['parent_span_sent_len'].value_counts())
# taxonomy of leaky sentences
print(leaky_sents.groupby('leaky_type_12')['edu_span_len'].describe().unstack())


# PARAGRAPHS
if False:
    # complex paragraphs
    complex_paras = paras_train[paras_train.edu_span_len > 1]
    print('Complex: {} / {} = {}'.format(
        len(complex_paras), len(paras_train),
        float(len(complex_paras)) / len(paras_train)))

    # leaky paragraphs
    # assert there is no paragraph which is leaky but not complex
    assert paras_train[paras_train.leaky & (paras_train.edu_span_len <= 1)].empty
    # proportion of leaky paragraphs
    leaky_paras = paras_train[paras_train.leaky]
    print('Leaky: {} / {} = {}'.format(
        len(leaky_paras), len(paras_train),
        float(len(leaky_paras)) / len(paras_train)))
    # proportion of leaky among complex paragraphs
    print('Leaky | Complex: {} / {} = {}'.format(
        len(leaky_paras), len(complex_paras),
        float(len(leaky_paras)) / len(complex_paras)))
    print()

    # compare leaky with non-leaky complex paragraphss: EDU length
    print('EDU span length of leaky vs non-leaky complex paragraphs')
    print(complex_paras.groupby('leaky')['edu_span_len'].describe().unstack())
    print()

    # for each leaky paragraph, number of paragraphs included in the smallest
    # RST node that fully covers the leaky paragraph
    if False:
        print(leaky_paras[['parent_span_para_len', 'parent_span_para_dist']].describe(
            percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        print(leaky_paras[(leaky_paras['parent_span_para_dist'] == 1)].describe())
        print(leaky_paras[(leaky_paras['parent_span_para_dist'] == 1) &
                          (leaky_paras['parent_span_para_len'] > 2)])
        #
        print(leaky_paras['parent_span_para_len'].value_counts())
    print(leaky_paras[:10])
# end leaky paragraphs


# WEIRDOS
# constituent nodes whose kids bear different relations
print([kid_rel for kid_rel in rels_train['kids_rel']
       if '+' in kid_rel])
