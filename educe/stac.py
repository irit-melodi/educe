# Author: Eric Kow
# License: BSD3

"""
Conventions specific to the STAC_ project

This includes things like

* corpus layout (see `corpus_files`)
* which annotations are of interest
* renaming/deleting/collapsing annotation labels

Some notes worth keeping in mind.

STAC/Glozz annotations are divided into units and relations.

Units
-----
There is a typology of unit types worth noting:

* structure : represent the document structure (eg. Dialogue, Turn, paragraph)
* segments  : spans of text associated with a dialogue act (eg. Offer, CounterOffer)
* resources : subspans of segments (Resource)

.. _STAC: http://www.irit.fr/STAC/
"""

from educe.corpus import *
from glob import glob
import educe.corpus
import educe.glozz as glozz
import os

structure_types=['Turn','paragraph','dialogue','Dialogue']
resource_types =['default','Resource']

def dialogue_act(x):
    """
    Set of dialogue act (aka speech act) annotations for a Unit, taking into
    consideration STAC conventions like collapsing Strategic_comment into Other
    """
    renames={ 'Strategic_comment':'Other'
            , 'Segment':'Other'
            }

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


def is_resource(annotation):
    """
    See Unit typology above
    """
    return (annotation.type in resource_types)

def is_dialogue_act(annotation):
    """
    See Unit typology above
    """
    blacklist = structure_types + resource_types
    return (annotation.type not in blacklist)

def is_structure(annotation):
    """
    """
    return (annotation.type not in structure_types)

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
        anno_glob='*.aa'

        for doc_dir in glob(full_glob):
            doc=os.path.basename(doc_dir)
            for stage in ['unannotated', 'units', 'discourse']:
                def register(annotator, f):
                    prefix = os.path.splitext(f)[0]
                    subdoc = os.path.basename(prefix)
                    if "_" in subdoc:
                        subdoc=subdoc.split("_",1)[1]
                        tf = prefix + ".ac"
                        file_id = FileId(doc, subdoc, stage, annotator)
                    corpus[file_id] = (f,tf)

                stage_dir=os.path.join(doc_dir,stage)
                if stage == 'unannotated':
                    unannotated_files=glob(os.path.join(stage_dir, anno_glob))
                    for f in unannotated_files:
                        register(None,f)
                else:
                    for annotator in os.listdir(stage_dir):
                        annotator_dir=os.path.join(stage_dir,annotator)
                        annotator_files=glob(os.path.join(annotator_dir, anno_glob))
                        for f in annotator_files:
                            register(annotator,f)
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


def write_annotation_file(anno_filename, doc):
    """
    Write a GlozzDocument to XML in the given path
    """
    glozz.write_annotation_file(anno_filename, doc)
