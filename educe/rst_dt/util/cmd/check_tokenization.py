"""Compare tokenization between PTB and CoreNLP for the RST-WSJ corpus.

"""

from __future__ import print_function

import os

import numpy as np

from nltk.corpus.reader import BracketParseCorpusReader

from educe.external.stanford_xml_reader import PreprocessingSource
from educe.rst_dt.corenlp import read_corenlp_result
from educe.rst_dt.corpus import Reader
from educe.rst_dt.deptree import RstDepTree
from educe.rst_dt.document_plus import DocumentPlus
from educe.rst_dt.ptb import PtbParser


DATA_DIR = 'data'
PTB_DIR = os.path.join(DATA_DIR, 'PTBIII/parsed/mrg/wsj')
RST_DIR = os.path.join(DATA_DIR, 'rst_discourse_treebank/data')
CORENLP_OUT_DIR = os.path.join(DATA_DIR, 'rst_discourse_treebank', '..',
                               'rst-dt-corenlp-2015-01-29')


if __name__ == '__main__':
    if not os.path.exists(PTB_DIR):
        raise ValueError("Unable to find PTB dir {}".format(PTB_DIR))
    if not os.path.exists(RST_DIR):
        raise ValueError("Unable to find RST dir {}".format(RST_DIR))
    if not os.path.exists(CORENLP_OUT_DIR):
        raise ValueError("Unable to find parsed dir {}".format(
            CORENLP_OUT_DIR))

    corpus = 'RSTtrees-WSJ-main-1.0/TRAINING'
    corpus_dir = os.path.join(RST_DIR, corpus)
    # syntactic parsers to compare
    ptb_reader = BracketParseCorpusReader(PTB_DIR,
                                          r'../wsj_.*\.mrg',
                                          encoding='ascii')
    # read the RST corpus
    rst_reader = Reader(corpus_dir)
    rst_corpus = rst_reader.slurp()
    # for each file, compare tokenizations between PTB and CoreNLP
    for key, rst_tree in sorted(rst_corpus.items()):
        doc_name = key.doc.split('.', 1)[0]
        if doc_name.startswith('wsj_'):
            print(doc_name)
            doc_wsj_num = doc_name.split('_')[1]
            section = doc_wsj_num[:2]

            # corenlp stuff
            core_fname = os.path.join(CORENLP_OUT_DIR, corpus,
                                      doc_name + '.out.xml')
            core_reader = PreprocessingSource()
            core_reader.read(core_fname, suffix='')
            corenlp_doc = read_corenlp_result(None, core_reader)
            core_toks = corenlp_doc.tokens
            core_toks_beg = [x.span.char_start for x in core_toks]
            core_toks_end = [x.span.char_end for x in core_toks]

            # PTB stuff
            # * create DocumentPlus (adapted from educe.rst_dt.corpus)
            rst_context = rst_tree.label().context
            ptb_docp = DocumentPlus(key, doc_name, rst_context)
            # * attach EDUs (yerk)
            # FIXME we currently get them via an RstDepTree created from
            # the original RSTTree, so as to get the left padding EDU
            rst_dtree = RstDepTree.from_rst_tree(rst_tree)
            ptb_docp.edus = rst_dtree.edus
            # * setup a PtbParser (re-yerk)
            ptb_parser = PtbParser(PTB_DIR)
            ptb_parser.tokenize(ptb_docp)
            # get PTB toks ; skip left padding token
            ptb_toks = ptb_docp.tkd_tokens[1:]
            ptb_toks_beg = ptb_docp.toks_beg[1:]
            ptb_toks_end = ptb_docp.toks_end[1:]

            # compare !
            core2ptb_beg = np.searchsorted(ptb_toks_beg, core_toks_beg,
                                           side='left')
            core2ptb_end = np.searchsorted(ptb_toks_end, core_toks_end,
                                           side='right') - 1
            # TODO maybe use np.diff?
            mism_idc = np.where(core2ptb_beg != core2ptb_end)[0]
            # group consecutive indices where beg != end
            mismatches = ([(mism_idc[0], mism_idc[0])] if mism_idc.any()
                          else [])
            for elt_cur, elt_nxt in zip(mism_idc[:-1], mism_idc[1:]):
                if elt_nxt > elt_cur + 1:
                    # new mismatch
                    mismatches.append((elt_nxt, elt_nxt))
                else:  # elt_nxt == elt_cur + 1
                    # extend current mismatch
                    mismatches[-1] = (mismatches[-1][0], elt_nxt)
            # print mismatches
            for core_beg, core_end in mismatches:
                m_core_toks = core_toks[core_beg:core_end + 1]  # DEBUG
                ptb_beg = core2ptb_beg[core_beg]
                ptb_end = core2ptb_end[core_end]
                n_ptb_toks = ptb_toks[ptb_beg:ptb_end + 1]
                if not n_ptb_toks:
                    print('* Text missing from PTB:',
                          '({}, {}) '.format(
                              m_core_toks[0].span.char_start,
                              m_core_toks[-1].span.char_end),
                          ' '.join(x.word for x in m_core_toks))
                elif not m_core_toks:
                    print('* Text missing from RST:',
                          '({}, {}) '.format(
                              n_ptb_toks[0].span.char_start,
                              n_ptb_toks[-1].span.char_end),
                          ' '.join(x.word for x in n_ptb_toks))
                else:
                    print('* Mismatch',
                          '\nCore >>>\t', ' '.join(
                              unicode(x) for x in m_core_toks),
                          '\nPTB  <<<\t', ' '.join(
                              unicode(x) for x in n_ptb_toks))
