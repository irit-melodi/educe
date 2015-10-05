"""Loader for Brown clusters induced by Turian and colleagues for ACL 2010.

The official description of the resource is available at:

http://metaoptimize.s3.amazonaws.com/brown-clusters-ACL2010/README.txt


Description of the resource
===========================
The clusters are induced in the RCV1 corpus, cleaned as described in the
ACL 2010 paper [1]_ .

4 different clusterings are provided that correspond respectively to
100, 320, 1000 and 3200 induced classes.

The first column is the bit-string name of the cluster,
the second column is the word,
the third column is the (absolute) frequency of the word in the corpus.

Turian and colleagues used prefixes of length 4, 6, 10 and 20 of the
bit-strings as features in their experiments.


Description of this module
==========================
This resource loader will download the selected variant(s) of the
Brown clusters. The file sizes range from 4.8 to 6.3 MB.

The data is downloaded in the current directory.


References
==========
.. [1] Joseph Turian, Lev-Arie Ratinov and Yoshua Bengio (2010) "WORD
REPRESENTATIONS: A SIMPLE AND GENERAL METHOD FOR SEMI-SUPERVISED
LEARNING"
(http://www.aclweb.org/anthology/P10-1040)

"""

from __future__ import print_function

import os

# python 2
from urllib2 import urlopen
# python 3 variant would be
# from urllib.request import urlopen


BASE_URL = 'http://metaoptimize.s3.amazonaws.com/brown-clusters-ACL2010/'
FNAME = 'brown-rcv1.clean.tokenized-CoNLL03.txt-c{}-freq1.txt'
NB_CLUSTERS = [100, 320, 1000, 3200]


def _load_brown_clusters_file(f):
    """Actually do load"""
    # my initial guess is that we are interested in a mapping from word
    # to cluster name (bit-string)
    word2clust = dict()

    for line in f:
        bstr, word, freq = line.strip().split()
        word2clust[word] = bstr

    return word2clust


def load_brown_clusters_file(f):
    """Load a Brown clusters file
    """
    with open(f) as f:
        clusters = _load_brown_clusters_file(f)
    return clusters


def fetch_brown_clusters(data_home=None, download_if_missing=True):
    """Load the Brown clusters.

    Parameters
    ----------
    data_home: optional, default: None
    Specify a download folder for the word representations. If None,
    all word representations are stored in 'educe/educe/wordreprs/data'.

    download_if_missing: optional, True by default
    If False, raise an IOError if the data is not locally available
    instead of trying to download the data from the source site.
    """
    data_home = (data_home if data_home is not None
                 else os.path.join(os.path.dirname(__file__), 'data'))
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    # for the moment, we store all word representations in the same folder
    brown_home = data_home

    clusters = dict()
    for c in NB_CLUSTERS:
        # file name, used for remote and local files
        fn = FNAME.format(c)
        # local file
        fpath = os.path.join(brown_home, fn)

        # download each missing Brown clusters file (if possible)
        if not os.path.exists(fpath):
            if download_if_missing:
                dl_msg = 'Downloading Brown clusters {} from {}'
                print(dl_msg.format(fn, BASE_URL))
                # TODO update message with progress : i/NB_CLUSTERS
                opener = urlopen(BASE_URL + fn)  # distant file
                open(fpath, 'wb').write(opener.read())
            else:
                raise IOError('Brown clusters not found')

        # load file
        clusters[c] = load_brown_clusters_file(fpath)

    return clusters
