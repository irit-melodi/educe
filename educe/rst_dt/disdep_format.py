"""Dependency format for RST discourse trees.

One line per EDU.
"""

from __future__ import absolute_import, print_function
import codecs
import csv
import os

from educe.rst_dt.corpus import RELMAP_112_18_FILE, RstRelationConverter

RELCONV = RstRelationConverter(RELMAP_112_18_FILE).convert_label


def _dump_disdep_file(rst_deptree, f):
    """Actually do dump"""
    writer = csv.writer(f, dialect=csv.excel_tab)

    # 0 is the fake root, there is no point in writing its info
    edus = rst_deptree.edus[1:]
    heads = rst_deptree.heads[1:]
    labels = rst_deptree.labels[1:]
    nucs = rst_deptree.nucs[1:]
    ranks = rst_deptree.ranks[1:]

    for i, (edu, head, label, nuc, rank) in enumerate(
            zip(edus, heads, labels, nucs, ranks), start=1):
        # text of EDU ; some EDUs have newlines in their text, so convert
        # those to simple spaces
        txt = edu.text().replace('\n', ' ')
        clabel = RELCONV(label)
        writer.writerow([i, txt, head, label, clabel, nuc, rank])


def dump_disdep_file(rst_deptree, f):
    """Dump dependency RST tree to a disdep file.

    Parameters
    ----------
    rst_deptree : RstDepTree
        RST dependency tree.
    f : str
        Path of the output file.
    """
    with codecs.open(f, 'wb', 'utf-8') as f_out:
        _dump_disdep_file(rst_deptree, f_out)


def dump_disdep_files(rst_deptrees, out_dir):
    """Dump dependency RST trees to a folder.

    This creates one file per RST tree, plus a metadata file that
    specifies the encoding of n-ary relations and the mapping from
    fine-grained relation labels to their classes.

    Parameters
    ----------
    rst_deptrees : list of RstDepTree
        RST dependency trees, one per document.
    out_dir : str
        Path to the output folder.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # metadata file
    nary_encs = [x.nary_enc for x in rst_deptrees]
    assert len(set(nary_encs)) == 1
    nary_enc = nary_encs[0]
    f_meta = os.path.join(out_dir, 'metadata')
    with codecs.open(f_meta, mode='w', encoding='utf-8') as f_meta:
        print('nary_enc: {}'.format(nary_enc), file=f_meta)
        print('relmap: {}'.format(RELMAP_112_18_FILE), file=f_meta)
    # deptrees, one file per doc
    for rst_deptree in rst_deptrees:
        doc_name = rst_deptree.origin.doc
        f_doc = os.path.join(out_dir, '{}.dis_dep'.format(doc_name))
        dump_disdep_file(rst_deptree, f_doc)
