# Author: Eric Kow
# License: BSD3

"""
Conventions specific to the STAC_ project

This includes things like

* corpus layout (see `corpus_files`)
* which annotations are of interest
* renaming/deleting/collapsing annotation labels

.. _STAC: http://www.irit.fr/STAC/
"""

from educe.corpus import *
from glob import glob
import educe.corpus
import educe.glozz as glozz
import os

def dialogue_act(x):
    """
    Set of dialogue act annotations for a Unit, taking into
    consideration STAC conventions like collapsing
    Strategic_comment into Other
    """
    renames={'Strategic_comment':'Other'}

    def rename(k):
        if k in renames.keys():
            return renames[k]
        else:
            return k

    return frozenset([rename(k) for k in split_type(x)])

def relation_labels(x):
    """
    Set of relation labels (eg. Elaboration, Explanation),
    taking into consideration any applicable STAC-isms
    """
    renames={}

    def rename(k):
        if k in renames.keys():
            return renames[k]
        else:
            return k

    return frozenset([rename(k) for k in split_type(x)])

def split_type(x):
    """
    An object's type as a (frozen)set of items
    """
    return frozenset(x.type.split("/"))

def is_real_annotation(annotation):
    """
    the subset of annotations which come from an annotator,
    as opposed to be prefilled 'structural' annotations
    """
    blacklist=['Turn','paragraph','dialogue','Dialogue','Segment','default']
    return (annotation.type not in blacklist)

def cleanup_comments(x):
    placeholder = "Please write in remarks..."
    ckey        = "Comments"
    if ckey in x.features.keys() and x.features[ckey] == placeholder:
        del x.features[ckey]

class Reader(educe.corpus.Reader):
    """
    See `educe.corpus.Reader` for details
    """
    def __init__(self, dir):
        educe.corpus.Reader.__init__(self, dir)

    def files(self):
        corpus={}
        full_glob=os.path.join(self.rootdir, 'pilot??')
        for doc_dir in glob(full_glob):
            doc=os.path.basename(doc_dir)
            for stage in ['units', 'discourse']:
                stage_dir=os.path.join(doc_dir,stage)
                for annotator in os.listdir(stage_dir):
                    annotator_dir=os.path.join(stage_dir,annotator)
                    annotator_files=glob(os.path.join(annotator_dir, '*.aa'))
                    for f in annotator_files:
                        prefix=os.path.splitext(f)[0]
                        subdoc=os.path.basename(prefix)
                        tf    =prefix + ".ac"
                        if "_" in subdoc:
                            subdoc=subdoc.split("_",1)[1]
                        file_id=FileId(doc, subdoc, stage, annotator)
                        corpus[file_id]=(f,tf)
        return corpus

    def slurp_subcorpus(self, cfiles, verbose=False):
        corpus={}
        counter=0
        for k in cfiles.keys():
            if verbose:
                sys.stderr.write("\rSlurping corpus dir [%d/%d]" % (counter, len(cfiles)))
            annotations=glozz.read_annotation_file(*cfiles[k])
            for u in annotations.units:
                u.origin=k
            corpus[k]=annotations
            counter=counter+1
        if verbose:
            sys.stderr.write("\rSlurping corpus dir [%d/%d done]\n" % (counter, len(cfiles)))
        return corpus
