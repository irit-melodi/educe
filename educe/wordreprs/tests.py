"""Unit tests for basic integration of word representations"""


from __future__ import print_function, absolute_import

from educe.wordreprs.brown_clusters_acl2010 import fetch_brown_clusters


def test_brown_clusters_acl2010():
    """Fetch and print Brown clusters for some discourse connectors"""

    clusters = fetch_brown_clusters()

    for nb_clusters, clust in sorted(clusters.items()):
        print('')
        print('Brown clusters with {} classes'.format(nb_clusters))
        print('----------------------------------')
        disc_conns = ['because', 'as', 'thus', 'so', 'then',
                      'according', 'including', 'and', 'or',
                      'for', 'before', 'after', 'while', 'by',
                      'without', 'despite', 'although', 'though',
                      'following', 'once', 'if', 'unless']
        for disc_conn in disc_conns:
            print('{}\t{}'.format(disc_conn, clust[disc_conn]))
