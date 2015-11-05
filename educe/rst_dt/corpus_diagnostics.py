"""This module is a loose collection of diagnostic functions on an RST corpus.

"""

from __future__ import print_function

from collections import Counter, defaultdict
import os

import pandas as pd

from educe.internalutil import treenode
from educe.rst_dt.annotation import RSTTree
from educe.rst_dt.corpus import (Reader as RstReader,
                                 RstRelationConverter,
                                 RELMAP_112_18_FILE)
# imports for dirty stuff (mostly)
import itertools

from nltk.corpus.reader import BracketParseCorpusReader

from educe.annotation import Span
from educe.external.parser import ConstituencyTree
from educe.internalutil import izip
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
def load_training_as_dataframe_new():
    """Load training section of the RST-WSJ corpus as a pandas.DataFrame.

    Returns
    -------
    node_df: pandas.DataFrame
        DataFrame of all nodes from the constituency trees.
    rel_df: pandas.DataFrame
        DataFrame of all relations.
    edu_df: pandas.DataFrame
        DataFrame of all EDUs.
    """
    node_rows = []  # list of dicts, one dict per node
    rel_rows = []  # list of dicts, one dict per relation
    # edu_rows contains pre-EDUs rather than EDUs themselves, but maybe
    # conflating both does no harm
    edu_rows = []  # list of dicts, one dict per EDU
    sent_rows = []  # ibid
    # TODO para_rows, look at leaky paragraphs (if they exist)

    rst_reader = RstReader(CD_TRAIN)
    rst_corpus = rst_reader.slurp()

    for doc_id, rtree_ref in sorted(rst_corpus.items()):
        doc_text = rtree_ref.label().context.text()
        doc_edus = rtree_ref.leaves()

        # 1. Collect constituency nodes and EDUs from the gold RST trees
        doc_rst_rel_rows = []
        doc_edu_rows = []

        # convert labels to coarse
        coarse_rtree_ref = REL_CONV(rtree_ref)

        # RST nodes: constituents are either relations or EDUs
        for node_idx, node in enumerate(coarse_rtree_ref.subtrees()):
            node_label = node.label()
            node_id = '{}_const{}'.format(node.origin.doc, node_idx)
            # node
            row = {
                'node_id': node_id,
                # char span
                'span_start': node_label.span.char_start,
                'span_end': node_label.span.char_end,
                # nuclearity
                'nuclearity': node_label.nuclearity,
                # relation
                'relation': node_label.rel,
            }

            if len(node) > 1:  # internal node => relation
                row.update({
                    # edu span
                    'edu_span_start': node_label.edu_span[0],
                    'edu_span_end': node_label.edu_span[1],
                    # info on children
                    'arity': len(node),
                    'kids_rel': '+'.join(
                        set(kid.label().rel for kid in node
                            if kid.label().rel != 'span')),
                    'kids_nuc': ''.join(
                        ('N' if kid.label().nuclearity == 'Nucleus' else 'S')
                        for kid in node),
                })
                # TODO add a column for sentential status (intra or inter)
                # (once we have the sentences and EDU <-> sentence mapping)
                doc_rst_rel_rows.append(row)
            else:  # pre-EDU
                edu = node[0]
                row.update({
                    'num': edu.num,
                })
                doc_edu_rows.append(row)

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
        # sentences with 2+ EDUs are 'complex'
        # TODO rewrite with numpy.unique
        complex_sent_idc = set(sent_idx for sent_idx, nb_occs
                               in Counter(edu2sent).items()
                               if nb_occs > 1 and sent_idx is not None)
        # sentences that don't have their own RST subtree are 'leaky' ;
        # collect the EDU spans of all constituent nodes from the RST tree
        rst_tree_node_spans = set((row['edu_span_start'], row['edu_span_end'])
                                  for row in doc_rst_rel_rows)
        # WIP
        rst_tree_node_spans_by_len = defaultdict(list)
        for edu_span in rst_tree_node_spans:
            edu_span_len = edu_span[1] - edu_span[0]
            rst_tree_node_spans_by_len[edu_span_len].append(edu_span)
        # end WIP
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
                    # TODO add columns using pandas
                    # or: sent_edu_starts[sent_idx] != sent_edu_ends[sent_idx]
                    'complex': (sent_idx in complex_sent_idc),
                    'leaky': (sent_idx in complex_sent_idc and
                              ((sent_edu_starts[sent_idx],
                                sent_edu_ends[sent_idx])
                               not in rst_tree_node_spans)),
                })
                # WIP
                # WIP find for each leaky sentence the smallest RST subtree
                # that covers it
                if row['leaky']:
                    for edu_span in itertools.chain.from_iterable(
                            [edu_spans for span_len, edu_spans
                             in sorted(
                                 rst_tree_node_spans_by_len.items())]):
                        if (edu_span[0] <= sent_edu_starts[sent_idx] and
                            sent_edu_ends[sent_idx] <= edu_span[1]):
                            parent_span = edu_span
                            break
                    else:
                        raise ValueError(
                            'No minimal spanning node for {}'.format(row))
                    # add info to row
                    try:
                        row.update({
                            # parent span, on EDUs
                            'parent_span_start': parent_span[0],
                            'parent_span_end': parent_span[1],
                            # length of parent span, in sentences
                            'parent_span_sent_len': (
                                edu2sent[parent_span[1] - 1] -
                                edu2sent[parent_span[0] - 1] + 1),
                        })
                    except TypeError:
                        print(doc_id.doc)
                        print(row['edu_span_start'], row['edu_span_end'])
                        print(parent_span)
                        raise
                    sent_span = Span(row['span_start'], row['span_end'])
                    print('{}: Leaky sentence [{}-{}] in [{}-{}]'.format(
                        doc_id, row['edu_span_start'], row['edu_span_end'],
                        row['parent_span_start'], row['parent_span_end']))
                    print(rtree_ref.label().context.text(sent_span))
                    print('Parent span covers {} sentences'.format(
                        row['parent_span_sent_len']))
                    print()
                else:
                    row.update({
                        'parent_span_start': row['edu_span_start'],
                        'parent_span_end': row['edu_span_end'],
                    })
                # end WIP
            doc_sent_rows.append(row)
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
        sent_rows.extend(doc_sent_rows)
        rel_rows.extend(doc_rst_rel_rows)
        edu_rows.extend(doc_edu_rows)

    # turn list into a DataFrame
    node_df = pd.DataFrame(node_rows)
    rel_df = pd.DataFrame(rel_rows)
    edu_df = pd.DataFrame(edu_rows)
    sent_df = pd.DataFrame(sent_rows)
    # add calculated columns here? (leaky and complex sentences)

    return node_df, rel_df, edu_df, sent_df


nodes_train, rels_train, edus_train, sents_train = load_training_as_dataframe_new()
# print(rels_train)
# as of version 0.17, pandas handles missing boolean values by degrading
# column type to object, which makes boolean selection return true for
# all non-None values ; the best workaround is to explicitly fill boolean
# na values
sents_train = sents_train.fillna(value={'complex': False, 'leaky': False})
# exclude 'fileX' documents
if False:
    sents_train = sents_train[~sents_train['sent_id'].str.startswith('file')]

# proportion of leaky sentences
leaky_sents = sents_train[sents_train.leaky]
# according to (Soricut and Marcu, 2003), p. 2, should be 323/6132 = 5.3%
print('Leaky: {} / {} = {}'.format(
    len(leaky_sents), len(sents_train),
    float(len(leaky_sents)) / len(sents_train)))
complex_sents = sents_train[sents_train.complex]
print('Complex: {} / {} = {}'.format(
    len(complex_sents), len(sents_train),
    float(len(complex_sents)) / len(sents_train)))
# assert there is no sentence which is leaky but not complex
assert sents_train[sents_train.leaky & (~sents_train.complex)].empty
# proportion of complex sentences that are leaky
print('Leaky | Complex: {} / {} = {}'.format(
    len(leaky_sents), len(complex_sents),
    float(len(leaky_sents)) / len(complex_sents)))
# compare leaky sentences with all complex sentences: length
print(complex_sents['edu_span_len'].describe())
print(leaky_sents['edu_span_len'].describe())
print(leaky_sents['parent_span_sent_len'].describe())
print(leaky_sents['parent_span_sent_len'].value_counts())
# WEIRDOS
# constituent nodes whose kids bear different relations
print([kid_rel for kid_rel in rels_train['kids_rel']
       if '+' in kid_rel])
