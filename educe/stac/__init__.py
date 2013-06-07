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

**Units**

There is a typology of unit types worth noting:

* structure : represent the document structure (eg. Dialogue, Turn, paragraph)
* segments  : spans of text associated with a dialogue act (eg. Offer, CounterOffer)
* resources : subspans of segments (Resource)

----

.. _STAC: http://www.irit.fr/STAC/
"""

from educe.corpus import *
from glob import glob
import copy
import educe.corpus
import educe.glozz as glozz
import itertools
import math
import os
import warnings

structure_types=['Turn','paragraph','dialogue','Dialogue']
resource_types =['default','Resource']

subordinating_relations =\
   [ 'Explanation'
   , 'Background'
   , 'Elaboration'
   , 'Correction'
   , 'Q-Elab'
   , 'Comment'
   ]

coordinating_relations =\
   [ 'Result'
   , 'Narration'
   , 'Continuation'
   , 'Contrast'
   , 'Parallel'
   ]

# TODO: I don't yet know how to classify these
unknown_relations =\
   [ 'Question-answer_pair'
   , 'Conditional'
   , 'Clarification_question'
   , 'Alternation'
   , 'Acknowledgement'
   ]

# ---------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------

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

def is_edu(annotation):
    """
    See Unit typology above
    """
    blacklist = structure_types + resource_types
    return (annotation.type not in blacklist)

def is_dialogue_act(annotation):
    """
    Deprecated in favour of is_edu
    """
    warnings.warn("deprecated, use is_edu instead", DeprecationWarning)
    return is_edu(annotation)

def is_structure(annotation):
    """
    """
    return (annotation.type not in structure_types)

def cleanup_comments(x):
    placeholder = "Please write in remarks..."
    ckey        = "Comments"
    if ckey in x.features.keys() and x.features[ckey] == placeholder:
        del x.features[ckey]

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
        corpus={}
        full_glob=os.path.join(self.rootdir, '*')
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
                elif os.path.exists(stage_dir):
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
            annotations.set_origin(k)
            corpus[k]=annotations
            counter=counter+1
        if verbose:
            sys.stderr.write("\rSlurping corpus dir [%d/%d done]\n" % (counter, len(cfiles)))
        return corpus

def id_to_path(k):
    """
    Given a fleshed out FileId (none of the fields are None),
    return a filepath for it following STAC conventions.

    You will likely want to add your own filename extensions to
    this path
    """
    for field in [ "doc", "subdoc", "stage" ]:
        if k.__dict__[field] is None:
            raise Exception('Need all FileId fields to be set (%s is unset)' % field)
    root = k.doc + '_' + k.subdoc
    pathparts = [k.doc, k.stage]
    if k.annotator is not None:
        pathparts.append(k.annotator)
    elif k.stage in [ "units", "discourse" ]:
        raise Exception('FileId.annotator must be set for unit/discourse items')
    pathparts.append(root)
    return os.path.join(*pathparts)


# ---------------------------------------------------------------------
# Adding annotations
# ---------------------------------------------------------------------

class PartialUnit:
    """
    Partially instantiated unit, for use when you want to programmatically
    insert annotations into a document

    A partially instantiated unit does not have any metadata (creation date,
    etc); as these will be derived automatically
    """
    def __init__(self, span, type, features):
        self.span     = span
        self.type     = type
        self.features = features

def create_units(k, doc, author, partial_units):
    """
    Return a collection of instantiated new unit objects.

    * `k` is of type `FileId`; it's used to create identifiers
    * `partial_units` should be of type `PartialUnit`
    """
    # It seems like Glozz uses the creation-date metadata field to
    # identify units (symptom: units that have different ids, but
    # some date don't appear in UI).
    #
    # Also, other tools in the STAC pipeline seem to use the convention
    # of negative numbers for fields where the notion of a creation date
    # isn't very appropriate (automatically derived annotations)
    #
    # So we take the smallest negative date (largest absolute value)
    # and subtract from there.
    #
    # For readability, we'll jump up a couple powers of 10
    creation_dates    = [ int(u.metadata['creation-date']) for u in doc.units ]
    smallest_neg_date = min(creation_dates)
    if smallest_neg_date > 0:
        smallest_neg_date = 0
    # next two power of 10
    id_base = 10 ** (int(math.log10(abs(smallest_neg_date))) + 2)

    def mk_creation_date(i):
        return str(0 - (id_base + i))

    def mk_unit(x,i):
        # Glozz seems to use creation date internally to identify
        # units, something ms based here doesn't seem so good
        # because not unique (too fast); using a counter instead
        # although by rights we also need to filter out
        # existing creation dates
        creation_date = mk_creation_date(i)
        metadata = { 'author'        : author
                   , 'creation-date' : creation_date
                   , 'lastModifier'  : 'n/a'
                   , 'lastModificationDate' : '0'
                   }
        unit_id = '_'.join([author,k.doc,k.subdoc,str(i)])
        return glozz.GlozzUnit(unit_id, x.span, x.type, x.features, metadata)

    return [ mk_unit(x,i) for x,i in itertools.izip(partial_units, itertools.count(0)) ]

def write_annotation_file(anno_filename, doc):
    """
    Write a GlozzDocument to XML in the given path
    """
    glozz.write_annotation_file(anno_filename, doc)
