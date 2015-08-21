"""This module provides support for the CODRA discourse parser.
"""

import codecs
import glob
import os

from .parse import parse_rst_dt_tree


def load_codra_output_files(container_path, level='doc'):
    """Load ctrees output by CODRA on the TEST section of RST-WSJ.

    Parameters
    ----------
    container_path: string
        Path to the main folder containing CODRA's output

    level: {'doc', 'sent'}, optional (default='doc')
        Level of decoding: document-level or sentence-level

    Returns
    -------
    data: dict
        Dictionary that should be akin to a sklearn Bunch, with
        interesting keys 'filenames', 'doc_names' and 'rst_ctrees'.

    Notes
    -----
    To ensure compatibility with the rest of the code base, doc_names
    are automatically added the ".out" extension. This would not work
    for fileX documents, but they are absent from the TEST section of
    the RST-WSJ treebank.
    """
    if level == 'doc':
        file_ext = '.doc_dis'
    elif level == 'sent':
        file_ext = '.sen_dis'
    else:
        raise ValueError("level {} not in ['doc', 'sent']".format(level))

    # find all files with the right extension
    pathname = os.path.join(container_path, '*{}'.format(file_ext))
    # filenames are sorted by name to avoid having to realign data
    # loaded with different functions
    filenames = sorted(glob.glob(pathname))  # glob.glob() returns a list

    # find corresponding doc names
    doc_names = [os.path.splitext(os.path.basename(filename))[0] + '.out'
                 for filename in filenames]

    # load the RST trees
    rst_ctrees = []
    for filename in filenames:
        with codecs.open(filename, 'r', 'utf-8') as f:
            # TODO (?) add support for and use RSTContext
            rst_ctree = parse_rst_dt_tree(f.read(), None)
            rst_ctrees.append(rst_ctree)

    data = dict(filenames=filenames,
                doc_names=doc_names,
                rst_ctrees=rst_ctrees)

    return data
