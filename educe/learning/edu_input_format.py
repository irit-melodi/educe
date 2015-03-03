"""This module implements a dumper for the EDU input format

See `https://github.com/kowey/attelo/blob/scikit/doc/input.rst`
"""

from __future__ import absolute_import, print_function
import csv
import itertools

import six

from .svmlight_format import dump_svmlight_file


# EDUs
def _dump_edu_input_file(docs, f):
    """Actually do dump"""
    writer = csv.writer(f, dialect=csv.excel_tab)

    for doc in docs:
        edus = doc.edus
        grouping = doc.grouping
        edu2sent = doc.edu2sent
        assert edus[0].is_left_padding()
        for i, edu in enumerate(edus[1:], start=1):  # skip the fake root
            edu_gid = edu.identifier()
            # some EDUs have newlines in their text (...):
            # convert to spaces
            edu_txt = edu.text().replace('\n', ' ')
            # subgroup: sentence identifier, backoff on EDU id
            sent_idx = edu2sent[i]
            if sent_idx is None:
                subgroup = edu_gid
            elif isinstance(sent_idx, six.string_types):
                subgroup = sent_idx
            else:
                subgroup = '{}_sent{}'.format(grouping, sent_idx)
            edu_start = edu.span.char_start
            edu_end = edu.span.char_end
            writer.writerow([edu_gid,
                             edu_txt,
                             grouping,
                             subgroup,
                             edu_start,
                             edu_end])


def dump_edu_input_file(docs, f):
    """Dump a dataset in the EDU input format."""
    with open(f, 'wb') as f:
        _dump_edu_input_file(docs, f)


# pairings
def _dump_pairings_file(docs_epairs, f):
    """Actually do dump"""
    writer = csv.writer(f, dialect=csv.excel_tab)

    for epairs in docs_epairs:
        for src, tgt in epairs:
            src_gid = src.identifier()
            tgt_gid = tgt.identifier()
            writer.writerow([src_gid, tgt_gid])


def dump_pairings_file(epairs, f):
    """Dump the EDU pairings"""
    with open(f, 'wb') as f:
        _dump_pairings_file(epairs, f)


def dump_all(X_gen, y_gen, f, class_mapping, docs, instance_generator):
    """Dump a whole dataset: features (in svmlight) and EDU pairs

    class_mapping is a mapping from label to int
    """
    # the labelset will be written in a comment at the beginning of the
    # svmlight file
    classes_ = [lbl for lbl, idx in sorted(class_mapping.items(),
                                           key=lambda x: x[1])]
    # first item should be reserved for unknown labels
    # we don't want to output this
    classes_ = classes_[1:]
    if classes_:
        comment = 'labels: {}'.format(' '.join(classes_))
    else:
        comment = None

    # dump: EDUs, pairings, vectorized pairings with label
    edu_input_file = f + '.edu_input'
    dump_edu_input_file(docs, edu_input_file)

    pairings_file = f + '.pairings'
    dump_pairings_file((instance_generator(doc) for doc in docs),
                       pairings_file)

    dump_svmlight_file(X_gen, y_gen, f, comment=comment)
