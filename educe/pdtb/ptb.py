# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Alignment with the Penn Treebank
"""

from os import path as fp

# pylint: disable=no-name-in-module
# pylint squawks about import error, but this seems to
# be some sort of fancy lazily loaded module which it's
# maybe a bit confused by
from nltk.corpus.reader import BracketParseCorpusReader
# pylint: enable=no-name-in-module


def _guess_ptb_name(k):
    """
    Given a PDTB corpus key, guess the equivalent filename from the
    Penn Tree Bank.

    Return None if the name doesn't look like it has an equivalent
    (note that returning something is not a guarantee either)
    """
    bname = fp.splitext(fp.basename(k.doc))[0]
    nparts = bname.split("_")
    if len(nparts) > 1:
        section = nparts[1][:2]  # wsj_2431 => 24
        return fp.join(section, bname + ".mrg")
    else:
        return None


def parse_trees(corpus, k, ptb):
    """
    Given an PDTB document and an NLTK PTB reader,
    return the PTB trees.

    Note that a future version of this function will try to
    educify the trees as well, but for now things will be
    fairly rudimentary
    """
    ptb_name = _guess_ptb_name(k)
    if ptb_name is None:
        return None

    return ptb.parsed_sents(ptb_name)


def reader(corpus_dir):
    """
    An instantiated NLTK BracketedParseCorpusReader for the PTB
    section relevant to the PDTB corpus.

    Note that the path you give to this will probably end with
    something like `parsed/mrg/wsj`
    """
    return BracketParseCorpusReader(corpus_dir,
                                    r'../wsj_.*\.mrg',
                                    encoding='ascii')
