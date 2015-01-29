"""This module implements a dumper for the EDU input format

See `https://github.com/kowey/attelo/blob/scikit/doc/input.rst`
"""

from __future__ import absolute_import

import itertools


def _dump_edu_input_file(inv_epair_gen, f):
    """Actually do dump"""
    # EDU: global id, text, grouping, span start, span end
    line_pattern = '{gid}\t{txt}\t{grp}\t{start}\t{end}'
    # possible parents: [0|gid]*
    line_pattern += '\t{parents}'
    line_pattern += '\n'

    # group entries by target EDU
    for tgt, epairs in itertools.groupby(inv_epair_gen, key=lambda x: x[0][0]):
        tgt_gid = tgt.identifier()
        tgt_txt = tgt.text()
        # TODO find a cleaner way to get grouping
        tgt_grp = [doc_grouping for _, doc_grouping in epairs][0]
        tgt_start = tgt.span.char_start
        tgt_end = tgt.span.char_end
        # possible parents
        srcs = [inv_edge[1].identifier() for inv_edge, _ in epairs]
        f.write(line_pattern.format(gid=tgt_gid,
                                    txt=tgt_txt,
                                    grp=tgt_grp,
                                    start=tgt_start,
                                    end=tgt_end,
                                    parents=' '.join(srcs)))


def dump_edu_input_file(inv_epair_gen, f):
    """Dump a dataset in the EDU input format."""
    with open(f, 'wb') as f:
        _dump_edu_input_file(inv_epair_gen, f)


def dump_all(X_gen, y_gen, f):
    """Dump a whole dataset: features (in svmlight) and EDU pairs"""
    # TODO: write EDU and svmlight files at the same time...
    # dump_edu_input_file(X_gen, y_gen, f)
    # dump_svmlight_file(X_gen, y_gen, f)
    raise NotImplementedError
