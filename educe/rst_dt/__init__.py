# Author: Eric Kow
# License: BSD3

"""
Conventions specific to the RST discourse treebank project
"""

from glob import glob
import copy
from   educe.annotation import Unit, Relation, Schema
import educe.corpus
from   educe.corpus import *
import educe.glozz as glozz
import itertools
import math
import os
import re
import warnings

from educe.rst_dt import parse

# ---------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------

class Reader(educe.corpus.Reader):
    """
    See `educe.corpus.Reader` for details
    """
    def __init__(self, dir):
        educe.corpus.Reader.__init__(self, dir)

    def files(self):
        anno_files={}
        full_glob=os.path.join(self.rootdir, '*.dis')

        for f in glob(full_glob):
            bname = os.path.basename(f)
            doc   = os.path.splitext(bname)[0]
            k     = FileId(doc=doc,
                           subdoc=None,
                           stage='discourse',
                           annotator='None')
            anno_files[k] = f
        return anno_files

    def slurp_subcorpus(self, cfiles, verbose=False):
        corpus={}
        counter=0
        for k in cfiles.keys():
            if verbose:
                sys.stderr.write("\rSlurping corpus dir [%d/%d]" % (counter, len(cfiles)))
            f = cfiles[k]
            annotations=parse.read_annotation_file(f)
            annotations.set_origin(k)
            corpus[k]=annotations
            counter=counter+1
        if verbose:
            sys.stderr.write("\rSlurping corpus dir [%d/%d done]\n" % (counter, len(cfiles)))
        return corpus

def id_to_path(k):
    """
    Given a fleshed out FileId (none of the fields are None),
    return a filepath for it following RST Discourse Treebank
    conventions.

    You will likely want to add your own filename extensions to
    this path
    """
    return k.doc
