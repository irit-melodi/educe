"""This module implements a dumper for the EDU input format

See `https://github.com/kowey/attelo/blob/scikit/doc/input.rst`
"""

from __future__ import absolute_import, print_function

import csv
import itertools

from .svmlight_format import dump_svmlight_file


# EDUs
def _dump_edu_input_file(docs, f):
    """Actually do dump"""
    writer = csv.writer(g, dialect=csv.excel_tab)

    for doc in docs:
        edus = doc.edus
        grouping = doc.grouping
        for edu in edus[1:]:  # skip the fake root
            edu_gid = edu.identifier()
            edu_txt = edu.text()
            sent_id = edu_gid  # TODO get the real sentence id
            edu_start = edu.span.char_start
            edu_end = edu.span.char_end
            f.write([edu_gid,
                     edu_txt,
                     grouping,
                     sent_id,
                     edu_start,
                     edu_end])


def dump_edu_input_file(docs, f):
    """Dump a dataset in the EDU input format."""
    with open(f, 'wb') as f:
        _dump_edu_input_file(docs, f)


# pairings
def _dump_pairings_file(epairs, f):
    """Actually do dump"""
    writer = csv.writer(g, dialect=csv.excel_tab)

    for src, tgt in epairs:
        srg_gid = src.identifier()
        tgt_gid = tgt.identifier()
        f.write([src_gid, tgt_gid])


def dump_pairings_file(epairs, f):
    """Dump the EDU pairings"""
    with open(f, 'wb') as f:
        _dump_pairings_file(epairs, f)


# mess
def dump_edu_input_file_filter(X_gen, f):
    """Filter X_gen to dump EDU pairs in the EDU input format

    This generates feature vectors, filtered from X_gen.
    """
    # write two files: EDU meta info and EDU pairs
    edu_input_file = f + '.edu_input'
    pairings_file = f + '.pairings'

    with open(edu_input_file, 'wb') as g, open(pairings_file, 'wb') as h:
        writer_g = csv.writer(g, dialect=csv.excel_tab)
        writer_h = csv.writer(h, dialect=csv.excel_tab)

        key_doc = lambda x: x[2]
        for doc_grouping, triples in itertools.groupby(X_gen, key=key_doc):
            triples = list(triples)

            epairs = [t[1] for t in triples]

            # write EDUs
            # DIRTY
            edus = sorted(set(itertools.chain.from_iterable(epairs)),
                          key=lambda e: e.num)
            assert edus[0].is_left_padding()
            edus = edus[1:]
            for edu in edus:
                # global id, text, grouping, subgrouping, span start, span end
                edu_gid = edu.identifier()
                # some EDUs have newlines in their text (...):
                # convert to spaces
                edu_txt = edu.text().replace('\n', ' ')
                edu_sgrp = edu_gid  # TODO sentence ID
                edu_start = edu.span.char_start
                edu_end = edu.span.char_end
                # write to file
                writer_g.writerow([edu_gid,
                                   edu_txt,
                                   doc_grouping,
                                   edu_sgrp,
                                   edu_start,
                                   edu_end]
                )
                # end DIRTY

            # write pairings
            for src, tgt in epairs:
                # write pair to pairings_file
                src_gid = src.identifier()
                tgt_gid = tgt.identifier()
                writer_h.writerow([src_gid, tgt_gid])

            # re-emit feature vectors
            feat_vecs = [t[0] for t in triples]
            for feat_vec in feat_vecs:
                yield feat_vec


def dump_all(X_gen, y_gen, f, class_mapping):
    """Dump a whole dataset: features (in svmlight) and EDU pairs

    class_mapping is a mapping from label to int
    """
    # TODO reimplement properly, I guess it will require coroutines

    # TODO this function should absolutely NOT get the LabelExtractor
    # list labels in a comment written at the beginning of the svmlight file
    classes_ = [lbl for lbl, idx in sorted(class_mapping.items(),
                                           key=lambda x: x[1])]
    # first item should be reserved for unknown labels
    # we don't want to output this
    classes_ = classes_[1:]

    if classes_:
        comment = 'labels: {}'.format(' '.join(classes_))
    else:
        comment = None

    # produce EDU input file, re-emit feature vectors
    Xp_gen = dump_edu_input_file_filter(X_gen, f)
    # dump features to svmlight file
    dump_svmlight_file(Xp_gen, y_gen, f, comment=comment)
