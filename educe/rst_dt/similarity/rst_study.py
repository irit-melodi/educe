# -*- coding: utf-8 -*-
"""A study of some semantic similarities on the RST corpus.

For each EDU pair on a text, check similarity wrt rhetorical relation
(or None).

This code adapts and extends on
"Word Mover’s Distance in Python" by vene & Matt Kusner [1]_.

References
----------
.. [1] http://vene.ro/blog/word-movers-distance-in-python.html
"""

# Authors: Philippe Muller <philippe.muller@irit.fr>
#          Mathieu Morey <mathieu.morey@irit.fr>

from __future__ import print_function

import argparse
import itertools
import os
import sys

import numpy as np
from joblib import Parallel, delayed

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import normalize

from pyemd import emd

from educe.metrics.wmd import load_embedding
from educe.rst_dt.annotation import SimpleRSTTree
from educe.rst_dt.corpus import Reader
from educe.rst_dt.deptree import RstDepTree


# relative to the educe docs directory
# was: DATA_DIR = '/home/muller/Ressources/'
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..',
    'data',  # alt: '..', '..', 'corpora'
)
RST_DIR = os.path.join(DATA_DIR, 'rst_discourse_treebank', 'data')
RST_CORPUS = {
    'train': os.path.join(RST_DIR, 'RSTtrees-WSJ-main-1.0', 'TRAINING'),
    'test': os.path.join(RST_DIR, 'RSTtrees-WSJ-main-1.0', 'TEST'),
    'double': os.path.join(RST_DIR, 'RSTtrees-WSJ-double-1.0'),
}


def wmd(i, j):
    """Compute the Word Mover's Distance between two EDUs.

    This presupposes the existence of two global variables:
    * `edu_vecs` is a sparse 2-dimensional ndarray where each row
    corresponds to the vector representation of an EDU,
    * `D_common` is a dense 2-dimensional ndarray that contains
    the euclidean distance between each pair of word embeddings.

    Parameters
    ----------
    i : int
        Index of the first EDU.
    j : int
        Index of the second EDU.

    Returns
    -------
    s : np.double
        Word Mover's Distance between EDUs i and j.
    """
    # EMD is extremely sensitive on the number of dimensions it has to
    # work with ; keep only the dimensions where at least one of the
    # two vectors is != 0
    union_idx = np.union1d(edu_vecs[i].indices, edu_vecs[j].indices)
    # EMD segfaults on incorrect parameters:
    # * if both vectors (and thus the distance matrix) are all zeros,
    # return 0.0 (consider they are the same)
    if not np.any(union_idx):
        return 0.0
    D_minimal = D_common[np.ix_(union_idx, union_idx)]
    bow_i = edu_vecs[i, union_idx].A.ravel()
    bow_j = edu_vecs[j, union_idx].A.ravel()
    # NB: emd() has an additional named parameter: extra_mass_penalty
    # pyemd by default sets it to -1, i.e. the max value in the distance
    # matrix
    return emd(bow_i, bow_j, D_minimal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Study the RST corpus')
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('wb'),
                        default=sys.stdout,
                        help='output file')
    parser.add_argument('--pairs', default='related',
                        choices=['related', 'all'],
                        help='selection of EDU pairs to examine')
    # parameters for CountVectorizer
    # NB: the following defaults differ from the standard ones in
    # CountVectorizer.
    # As of 2016-03-16, we define our defaults to be:
    # * strip accents='unicode'
    # * lowercase=False
    # * stop_words='english'
    parser.add_argument('--strip_accents', default='unicode',
                        choices=['ascii', 'unicode', 'None'],
                        help='preprocessing: method to strip accents')
    parser.add_argument('--lowercase', action='store_true',
                        help='preprocessing: lowercase')
    parser.add_argument('--stop_words', default='english',
                        choices=['english', 'None'],  # TODO: add "list"
                        help='preprocessing: filter stop words')
    parser.add_argument('--scale', default='None',
                        choices=['0_1', 'None'],
                        help='scale distance to given range')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='max number of concurrently running jobs')
    parser.add_argument('--verbose', type=int, default=1,
                        help='verbosity level')
    # TODO add arguments for train and test corpora
    args = parser.parse_args()

    # * get parameters for the CountVectorizer and outfile
    # properly recast strip_accents if None
    strip_accents = (args.strip_accents if args.strip_accents != 'None'
                     else None)
    lowercase = args.lowercase
    stop_words = (args.stop_words if args.stop_words != 'None'
                  else None)
    outfile = args.outfile
    n_jobs = args.n_jobs
    verbose = args.verbose
    sel_pairs = args.pairs
    distance_range = (args.scale if args.scale != 'None'
                      else None)

    # * read the corpus
    rst_corpus_dir = RST_CORPUS['double']
    rst_reader = Reader(rst_corpus_dir)
    rst_corpus = rst_reader.slurp(verbose=True)
    corpus_texts = [v.text() for k, v in sorted(rst_corpus.items())]

    # MOVE ~ WMD.__init__()
    # load word embeddings
    vocab_dict, W = load_embedding("embed")
    # end MOVE

    # MOVE ~ WMD.fit(corpus_texts?)
    # fit CountVectorizer to the vocabulary of the corpus
    vect = CountVectorizer(
        strip_accents=strip_accents, lowercase=lowercase,
        stop_words=stop_words
    ).fit(corpus_texts)
    # compute the vocabulary common to the embeddings and corpus, restrict
    # the word embeddings matrix and replace the vectorizer
    common = [word for word in vect.get_feature_names()
              if word in vocab_dict]
    W_common = W[[vocab_dict[w] for w in common]]
    vect = CountVectorizer(
        strip_accents=strip_accents, lowercase=lowercase,
        stop_words=stop_words,
        vocabulary=common, dtype=np.double
    ).fit(corpus_texts)
    # compute the distance matrix between each pair of word embeddings
    print('Computing the distance matrix between each pair of embeddings...',
          file=sys.stderr)
    D_common = euclidean_distances(W_common)
    D_common = D_common.astype(np.double)
    # optional: scale distances to range (0, 1)
    if distance_range is not None:
        D_common /= D_common.max()
    print('done', file=sys.stderr)
    # end MOVE fit()

    # MOVE ~ WMD.transform(params?)
    # print header to file: list parameters used for this run
    # NB: this should really be a dump of the state of the *WMD* object
    params = {
        'corpus': os.path.relpath(rst_corpus_dir, start=DATA_DIR),
        'strip_accents': strip_accents,
        'lowercase': lowercase,
        'stop_words': stop_words,
        'n_jobs': n_jobs,
        'verbose': verbose,
    }
    print('# parameters: ({})'.format(params),
          file=outfile)

    # do the real job
    corpus_items = sorted(rst_corpus.items())
    doc_keys = [key.doc for key, doc in corpus_items]
    doc_key_dtrees = [
        (doc_key.doc,
         RstDepTree.from_simple_rst_tree(SimpleRSTTree.from_rst_tree(doc)))
        for doc_key, doc in corpus_items
    ]
    edu_txts = list(e.text().replace('\n', ' ')
                    for doc_key, dtree in doc_key_dtrees
                    for e in dtree.edus)
    # vectorize each EDU using its text
    edu_vecs = vect.transform(edu_txts)
    # normalize each row of the count matrix using the l1 norm
    # (copy=False to perform in place)
    edu_vecs = normalize(edu_vecs, norm='l1', copy=False)
    # get all pairs of EDUs of interest, here as triples
    # (gov_idx, dep_idx, lbl)
    # TODO maybe sort edu pairs so that dependents with
    # the same governor are grouped (potential speed up?)
    edu_pairs = [
        [(doc_key, gov_idx, dep_idx, lbl)
         for dep_idx, (gov_idx, lbl)
         in enumerate(zip(dtree.heads[1:], dtree.labels[1:]),
                      start=1)]
        for doc_key, dtree in doc_key_dtrees
    ]
    if sel_pairs == 'all':
        # generate all possible pairs
        # we just need to generate half of them because WMD is symmetric,
        # so we generate all combinations, where we keep a distinctive
        # order if the pair is related
        edu_pairs_rel = {(doc_key, gov_idx, dep_idx): lbl
                         for doc_key, gov_idx, dep_idx, lbl
                         in itertools.chain.from_iterable(edu_pairs)}
        edu_pairs = []
        for doc_key, dtree in doc_key_dtrees:
            edu_pairs.append([])
            for gov_idx, dep_idx in itertools.combinations(
                    range(len(dtree.edus)), 2):
                if (doc_key, gov_idx, dep_idx) in edu_pairs_rel:
                    kept_pair = (doc_key, gov_idx, dep_idx,
                                 edu_pairs_rel[(doc_key, gov_idx, dep_idx)])
                elif (doc_key, dep_idx, gov_idx) in edu_pairs_rel:
                    kept_pair = (doc_key, dep_idx, gov_idx,
                                 edu_pairs_rel[(doc_key, dep_idx, gov_idx)])
                else:
                    kept_pair = (doc_key, gov_idx, dep_idx,
                                 'UNRELATED')
                edu_pairs[-1].append(kept_pair)

    # transform local index of EDU in doc into global index in the list
    # of all EDUs from all docs
    doc_lens = [0] + [len(dtree.edus)
                      for doc_key, dtree in doc_key_dtrees[:-1]]
    doc_offsets = np.cumsum(doc_lens)
    edu_pairs = [[(doc_key, gov_idx, dep_idx, lbl,
                   doc_offset + gov_idx, doc_offset + dep_idx)
                  for doc_key, gov_idx, dep_idx, lbl
                  in doc_edu_pairs]
                 for doc_offset, doc_edu_pairs
                 in zip(doc_offsets, edu_pairs)]
    edu_pairs = list(itertools.chain.from_iterable(edu_pairs))
    # WIP
    # compute the WMD between the pairs of EDUs
    edu_pairs_wmd = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(wmd)(gov_idx_abs, dep_idx_abs)
        for doc_key, gov_idx, dep_idx, lbl, gov_idx_abs, dep_idx_abs
        in edu_pairs
    )

    wmd_strs = [
        ("%s::%s::%.5f::(%s)--(%s)" %
         (doc_key, lbl, sim, edu_txts[gov_idx_abs], edu_txts[dep_idx_abs]))
        for (doc_key, gov_idx, dep_idx, lbl, gov_idx_abs, dep_idx_abs), sim
        in zip(edu_pairs, edu_pairs_wmd)
    ]
    print('\n'.join(wmd_strs), file=outfile)
    # end MOVE transform()
