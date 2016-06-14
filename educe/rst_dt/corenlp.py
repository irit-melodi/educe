"""Functions to get parses on the RST-DT corpus using Stanford CoreNLP.

TODO
----
* [ ] use `educe.stac.corenlp`, `educe.external.corenlp`
      and Eric's rstdt-corenlp ipython notebook
* [ ] adapt to file* documents
"""

from __future__ import print_function

from collections import defaultdict
import os.path

import nltk.tree

from educe.external.corenlp import (CoreNlpToken, CoreNlpDocument)
from educe.external.coref import (Chain, Mention)
from educe.external.parser import (ConstituencyTree, DependencyTree)
from educe.external.stanford_xml_reader import PreprocessingSource
from educe.ptb.annotation import (transform_tree, strip_subcategory)
from educe.ptb.head_finder import find_lexical_heads


def _guess_corenlp_name(k):
    """Guess the CoreNLP output filename for an RST-DT corpus key.

    Parameters
    ----------
    k: educe.corpus.FileId
        RST-DT corpus key

    Returns
    -------
    corenlp_out_file: string or None
        Relative path to the CoreNLP output file, None if we can't find
        any.
    """
    bname = os.path.basename(k.doc)
    if bname.startswith('file'):
        return None

    corenlp_out_file = bname + '.xml'
    return corenlp_out_file


def read_corenlp_result(doc, corenlp_doc):
    """Read CoreNLP's output for a document.

    Parameters
    ----------
    doc: educe.rst_dt.document_plus.DocumentPlus
        The original document (currently unused, could be necessary to
        determine e.g. token offset for specific file formats ; if it
        never gets used, this function should probably to the generic
        default and moved to `educe.external.corenlp`).

    corenlp_doc: educe.external.stanford_xml_reader.PreprocessingSource
        Object that contains all annotations for the document

    Returns
    -------
    corenlp_doc: CoreNlpDocument
        A CoreNlpDocument containing all information
    """
    # sentences
    sentences = corenlp_doc.get_ordered_sentence_list()

    # tokens
    sentence_toks = defaultdict(list)
    for tok in corenlp_doc.get_ordered_token_list():
        sid = tok['s_id']
        sentence_toks[sid].append(tok)

    # educe tokens
    educe_tokens = defaultdict(dict)
    for sent in sentences:
        sid = sent['id']
        sent_toks = sentence_toks[sid]
        offset = 0  # was: sent_begin
        for tok in sent_toks:
            tid = tok['id']
            educe_tokens[sid][tid] = CoreNlpToken(tok, offset)

    # educe tokens, ctree and dtree
    all_tokens = []
    all_ctrees = []
    all_dtrees = []
    for sent in sentences:
        sid = sent['id']
        tokens_dict = educe_tokens[sid]
        # NEW extract local id to properly sort tokens
        tok_local_id = lambda x: int(x[len(sid) + 1:])
        sorted_tokens = [tokens_dict[x]
                         for x in sorted(tokens_dict, key=tok_local_id)]
        # ctree
        tree = nltk.tree.Tree.fromstring(sent['parse'])
        # FIXME 2016-06-13 skip the ROOT node, as in PTB
        # maybe we'd better add ROOT to the empty parentheses in the
        # PTB version, but just getting rid of ROOT here seems simpler:
        # the type of the root node of a tree is informative: usually
        # S, but more interestingly SINV, NP...
        if tree.label() != 'ROOT' or len(tree) > 1:
            print(tree)
            raise ValueError('Atypical root of CoreNLP tree')
        tree = tree[0]  # go down from ROOT to the real root
        educe_ctree = ConstituencyTree.build(tree, sorted_tokens)
        # dtree
        deps = defaultdict(list)
        for lbl, gov_id, dep_id in sent['dependencies']:
            deps[gov_id].append((lbl, dep_id))
        educe_dtree = DependencyTree.build(deps, tokens_dict, sid + '-0')
        # store educe tokens, ctrees and dtrees
        all_tokens.extend(sorted_tokens)
        all_ctrees.append(educe_ctree)
        all_dtrees.append(educe_dtree)

    # coreference chains
    all_chains = []
    for chain in corenlp_doc.get_coref_chains():
        mentions = []
        for mntn in chain:
            sid = mntn['sentence']
            # helper functions to extract local ids and generate global ids
            local_id = lambda x: int(x[len(sid) + 1:])
            global_id = lambda x: sid + '-' + str(x)
            # retrieve tokens for this mention
            start = local_id(mntn['start'])
            end = local_id(mntn['end'])
            tokens = [educe_tokens[sid][global_id(tok_idx)]
                      for tok_idx in range(start, end)]
            head = educe_tokens[sid][mntn['head']]
            mentions.append(Mention(tokens, head,
                                    mntn['most_representative']))
        all_chains.append(Chain(mentions))

    corenlp_doc = CoreNlpDocument(all_tokens, all_ctrees, all_dtrees,
                                  all_chains)
    return corenlp_doc


class CoreNlpParser(object):
    """CoreNLP parser.
    """

    def __init__(self, corenlp_out_dir):
        """ """
        self.corenlp_out_dir = corenlp_out_dir

    def tokenize(self, doc):
        """Tokenize the document text.

        Parameters
        ----------
        doc: educe.rst_dt.DocumentPlus
            Document

        Returns
        -------
        doc: educe.rst_dt.DocumentPlus
            Tokenized document
        """
        corenlp_out_name = _guess_corenlp_name(doc.key)
        if corenlp_out_name is None:
            return doc

        fname = os.path.join(self.corenlp_out_dir,
                             corenlp_out_name)
        if not os.path.exists(fname):
            raise ValueError('CoreNLP XML: no file {}'.format(fname))
        # CoreNLP XML output reader
        reader = PreprocessingSource()
        reader.read(fname, suffix='')
        corenlp_out = read_corenlp_result(doc, reader)

        # modify DocumentPlus doc to add tokens
        doc.set_tokens(corenlp_out.tokens)

        return doc

    def parse(self, doc):
        """Parse
        """
        corenlp_out_name = _guess_corenlp_name(doc.key)
        if corenlp_out_name is None:
            return doc

        fname = os.path.join(self.corenlp_out_dir,
                             corenlp_out_name)
        if not os.path.exists(fname):
            raise ValueError('CoreNLP XML: no file {}'.format(fname))
        # CoreNLP XML output reader
        # FIXME the same reading is done in tokenize(), should find
        # a way to cache or share call
        reader = PreprocessingSource()
        reader.read(fname, suffix='')
        corenlp_out = read_corenlp_result(doc, reader)

        # ctrees and lexical heads on their nodes
        ctrees = corenlp_out.trees
        # strip function tags
        # TODO maybe this should be an internal preprocessing step in
        # find_lexical_heads(), so as to keep the function tags
        # that are kept by default by CoreNLP parser because they were found
        # to be useful e.g. `-retainTMPSubcategories`
        ctrees_no_gf = [transform_tree(ctree, strip_subcategory)
                        for ctree in ctrees]
        lex_heads = [find_lexical_heads(ctree_no_gf)
                     for ctree_no_gf in ctrees_no_gf]

        # store trees in doc
        doc.set_syn_ctrees(ctrees_no_gf, lex_heads=lex_heads)

        return doc
