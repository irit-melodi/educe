"""This module implements a dumper for the EDU input format

See `https://github.com/kowey/attelo/blob/scikit/doc/input.rst`
"""

from __future__ import absolute_import, print_function

import itertools

from .svmlight_format import dump_svmlight_file


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


def dump_edu_input_file_filter(X_gen, f):
    """Filter X_gen to dump EDU pairs in the EDU input format

    This generates feature vectors, filtered from X_gen.
    """
    edu_input_file = f + '.edu_input'
    with open(edu_input_file, 'wb') as g:
        # EDU: global id, text, grouping, span start, span end
        line_pattern = '{gid}\t{txt}\t{grp}\t{start}\t{end}'
        # possible parents: [0|gid]*
        line_pattern += '\t{parents}'
        line_pattern += '\n'

        for tgt, triples in itertools.groupby(X_gen, key=lambda t: t[1][0]):
            triples = list(triples)
            tgt_gid = tgt.identifier()
            # some EDUs have newlines in their text (...): convert to spaces
            tgt_txt = tgt.text().replace('\n', ' ')
            # TODO find a cleaner way to get grouping
            tgt_grp = [t[2] for t in triples][0]  # should be unique value
            tgt_start = tgt.span.char_start
            tgt_end = tgt.span.char_end
            # possible parents
            srcs = [t[1][1].identifier() for t in triples]
            # write to file
            g.write(line_pattern.format(gid=tgt_gid,
                                        txt=tgt_txt,
                                        grp=tgt_grp,
                                        start=tgt_start,
                                        end=tgt_end,
                                        parents=' '.join(srcs)))
            feat_vecs = [t[0] for t in triples]
            for feat_vec in feat_vecs:
                yield feat_vec


def dump_all(X_gen, y_gen, f):
    """Dump a whole dataset: features (in svmlight) and EDU pairs"""
    # TODO reimplement properly, I guess it will require coroutines

    # produce EDU input file, re-emit feature vectors
    Xp_gen = dump_edu_input_file_filter(X_gen, f)
    # dump features to svmlight file
    dump_svmlight_file(Xp_gen, y_gen, f)
