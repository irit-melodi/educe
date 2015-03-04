"""This module implements a loader and dumper for vocabularies.
"""

import codecs

def _dump_vocabulary(vocabulary, f):
    """Actually do dump"""
    line_pattern = u'{fn}\t{fx}\n'
    # order features by idx
    for feat_name, feat_idx in sorted(vocabulary.items(),
                                      key=lambda x: x[1]):
        # feature ids in libsvm are one-based, so feat_idx + 1
        f.write(line_pattern.format(fn=feat_name,
                                    fx=str(feat_idx + 1)))


def dump_vocabulary(vocabulary, f):
    """Dump the vocabulary as a tab-separated file.
    """
    with codecs.open(f, 'w', 'utf-8') as f:
        _dump_vocabulary(vocabulary, f)
