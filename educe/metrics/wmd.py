# coding: utf-8
"""Utility functions to play with Word Mover's Distance.

This code is essentially a refactoring from
"Word Mover’s Distance in Python" by vene & Matt Kusner [1]_.
This post provides a full implementation of the method described in [2]_.

References
----------
.. [1] http://vene.ro/blog/word-movers-distance-in-python.html
.. [2] http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf

TODO
----
* [ ] Mihalcea distance, max/average etc take from old Voiladis code
"""

# Authors: Philippe Muller <philippe.muller@irit.fr>
#          Mathieu Morey <mathieu.morey@irit.fr>

from __future__ import print_function

import os
import sys

import numpy as np

from pyemd import emd

# we need to use open from the new io module in python 3, which is
# available from python 2.6 but needs to be explicitly imported
if sys.version_info[0] > 2:
    pass
else:
    from io import open


def create_cache(filepath="data"):
    if ((not os.path.exists(filepath+"/"+"embed.dat") or
         not os.path.exists(filepath+"/"+"embed.vocab"))):
        print("Cache of word embeddings...",
              file=sys.stderr)
        from gensim.models.word2vec import Word2Vec
        wv = Word2Vec.load_word2vec_format(
            filepath+"/"+"GoogleNews-vectors-negative300.bin.gz",
            binary=True)
        fp = np.memmap(filepath+"/"+"embed.dat", dtype=np.double, mode='w+',
                       shape=wv.syn0.shape)
        fp[:] = wv.syn0[:]
        with open(filepath+"/"+"embed.vocab", "w", encoding="utf8") as f:
            for _, w in sorted((voc.index, word) for word, voc
                               in wv.vocab.items()):
                print(w, file=f)
        del fp, wv
        print('done', file=sys.stderr)


def load_embedding(mapfile="embed"):
    # create cache file if necessary
    if ((not os.path.exists('data/%s.dat' % mapfile) or
         not os.path.exists('data/%s.vocab' % mapfile))):
        create_cache(filepath='data')
    print('Loading embedding...', file=sys.stderr)
    # memmap the cache file
    W = np.memmap("data/%s.dat" % mapfile, dtype=np.double, mode="r",
                  shape=(3000000, 300))
    with open("data/%s.vocab" % mapfile) as f:
        vocab_list = [x.strip() for x in f.readlines()]
    vocab_dict = {w: k for k, w in enumerate(vocab_list)}
    print('done', file=sys.stderr)
    return vocab_dict, W


def wmd(edu_vecs, i, j, D_embed):
    """Compute the Word Mover's Distance between two EDUs.

    Parameters
    ----------
    edu_vecs : sparse matrix
        One row per EDU.
    i : int
        Index of the first EDU.
    j : int
        Index of the second EDU.
    D_embed : dense matrix of np.double
        Distance matrix between each pair of word embeddings.

    Returns
    -------
    s : np.double
        Word Mover's Distance between EDUs i and j.

    Notes
    -----
    This function is an example implementation to compute the WMD
    on a pair of EDUs.
    You are however discouraged to use it as is if speed and memory matter.
    The recommended way is then to copy this function into your script,
    remove the parameters `edu_vecs` and `D_embed` from the signature of
    this function and have them point to variables from a wider scope
    (e.g. global variables from the module, even if they are defined in a
    conditional block such as `if __name__ == "__main__"`).
    This way, joblib.Parallel does not have to pickle parameters.
    An alternative course would be to memmap parameters as in the
    joblib.Parallel documentation, but it still runs an order of magnitude
    slower.
    """
    v_1 = edu_vecs[i].toarray().ravel()
    v_2 = edu_vecs[j].toarray().ravel()
    # NB: emd() has an additional named parameter: extra_mass_penalty
    # pyemd by default sets it to -1, i.e. the max value in the distance
    # matrix
    s = emd(v_1, v_2, D_embed)
    return s
