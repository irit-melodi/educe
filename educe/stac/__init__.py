# Author: Eric Kow
# License: BSD3

"""
Conventions specific to the STAC_ project

This includes things like

* corpus layout (see `corpus_files`)
* which annotations are of interest
* renaming/deleting/collapsing annotation labels

STAC/Glozz annotations can be a bit confusing because for two reasons, first
that Glozz objects are used to annotate very different things; and second
that annotations are done on different stages

Stage 1 (units)

+-----------+---------------------------------------------+
| Glozz     | Uses                                        |
+===========+=============================================+
| units     | doc structure, EDUs, resources, preferences |
+-----------+---------------------------------------------+
| relations | coreference                                 |
+-----------+---------------------------------------------+
| schemas   | composite resources                         |
+-----------+---------------------------------------------+

Stage 2 (discourse)

+-----------+----------------------------------+
| Glozz     | Uses                             |
+===========+==================================+
| units     | doc structure, EDUs              |
+-----------+----------------------------------+
| relations | relation instances, coreference  |
+-----------+----------------------------------+
| schemas   | CDUs                             |
+-----------+----------------------------------+

**Units**

There is a typology of unit types worth noting:

* doc structure : type eg. `Dialogue`, `Turn`, `paragraph`
* resources     : subspans of segments (type `Resource`)
* preferences   : subspans of segments (type `Preference`)
* EDUs          : spans of text associated with a dialogue act (eg. type
  `Offer`, `Accept`) (during discourse stage, these are just type `Segment`)

**Relations**

* coreference : (type `Anaphora`)
* relation instances : links between EDUs, annotated with relation label
  (eg. type `Elaboration`, type `Contrast`, etc).  These can be further
  divided in subordinating or coordination relation instances according
  to their label

**Schemas**

* composite resources : boolean combinations of resources (eg. "sheep or ore")
* CDUs: type `Complex_discourse_unit` (discourse stage)

----

.. _STAC: http://www.irit.fr/STAC/
"""

from educe.corpus import *
from glob import glob
import copy
from   educe.annotation import Unit, Relation, Schema
import educe.corpus
import educe.glozz as glozz
import itertools
import math
import os
import re
import warnings

structure_types=['Turn','paragraph','dialogue','Dialogue']
resource_types =['default','Resource']
preference_types = ['Preference']

subordinating_relations =\
   [ 'Explanation'
   , 'Background'
   , 'Elaboration'
   , 'Correction'
   , 'Q-Elab'
   , 'Comment'
   , 'Question-answer_pair'
   , 'Clarification_question'
   , 'Acknowledgement'
   ]

coordinating_relations =\
   [ 'Result'
   , 'Narration'
   , 'Continuation'
   , 'Contrast'
   , 'Parallel'
   , 'Conditional'
   , 'Alternation'
   ]

def split_turn_text(t):
    """
    STAC turn texts are prefixed with a turn number and speaker
    to help the annotators
    (eg. "379: Bob: I think it's your go, Alice").

    Given the text for a turn, split the string into a prefix
    containing this turn/speaker information (eg. "379: Bob: "),
    and a body containing the turn text itself (eg. "I think it's
    your go, Alice").

    Mind your offsets! They're based on the whole turn string.
    """
    prefix_re = re.compile(r'(^[0-9]+ ?: .*? ?: )(.*)$')
    match     = prefix_re.match(t)
    if match:
        return (match.group(1),match.group(2))
    else:
        # it's easy to just return the body here, but when this arises
        # it's a sign that something weird has happened
        raise Exception("Turn does not start with number/speaker prefix: " + t)

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
    return isinstance(annotation, Unit) and\
            annotation.type in resource_types

def is_preference(annotation):
    """
    See Unit typology above
    """
    return isinstance(annotation, Unit) and\
            annotation.type in preference_types

def is_turn(annotation):
    """
    See Unit typology above
    """
    return isinstance(annotation, Unit) and\
            annotation.type == 'Turn'

def is_edu(annotation):
    """
    See Unit typology above
    """
    blacklist = structure_types + resource_types + preference_types
    return isinstance(annotation, Unit) and\
            annotation.type not in blacklist

def is_relation_instance(annotation):
    """
    See Relation typology above
    """
    return isinstance(annotation, Relation) and\
            annotation.type in subordinating_relations or\
            annotation.type in coordinating_relations

def is_subordinating(annotation):
    """
    See Relation typology above
    """
    return isinstance(annotation, Schema) and\
            annotation.type in subordinating_relations

def is_coordinating(annotation):
    """
    See Relation typology above
    """
    return isinstance(annotation, Schema) and\
            annotation.type in coordinating_relations

def is_cdu(annotation):
    """
    See CDUs typology above
    """
    return isinstance(annotation, Schema) and\
            annotation.type == 'Complex_discourse_unit'

def is_dialogue_act(annotation):
    """
    Deprecated in favour of is_edu
    """
    warnings.warn("deprecated, use is_edu instead", DeprecationWarning)
    return is_edu(annotation)

def is_structure(annotation):
    """
    """
    return isinstance(annotation, Unit) and\
            annotation.type in structure_types

def cleanup_comments(x):
    placeholder = "Please write in remarks..."
    ckey        = "Comments"
    if ckey in x.features.keys() and x.features[ckey] == placeholder:
        del x.features[ckey]

def twin(corpus, anno, stage='units'):
    """
    Given an annotation in a corpus, retrieve the equivalent annotation
    (by local identifier) from a a different stage of the corpus.
    Return this "twin" annotation or None if it is not found

    Note that the annotation's origin must be set

    The typical use of this would be if you have an EDU in the 'discourse'
    stage and need to get its 'units' stage equvialent to have its
    dialogue act.
    """
    if anno.origin is None:
        raise Exception('Annotation origin must be set')
    anno_local_id  = anno.local_id()
    twin_key       = copy.copy(anno.origin)
    twin_key.stage = stage
    if twin_key in corpus:
        udoc  = corpus[twin_key]
        twins = [ u for u in udoc.annotations() if u.local_id() == anno_local_id ]
        if len(twins) > 0:
            return twins[0]
        else:
            return None
    else:
        return None

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
                        file_id    = FileId(doc, subdoc, stage, annotator)
                        ac_file_id = FileId(doc, subdoc, 'unannotated', None)
                        tf = os.path.join(self.rootdir, id_to_path(ac_file_id)) + ".ac"
                    else:
                        raise Exception('STAC corpus filenames should be in the form doc_subdocument: %s', subdoc)
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

class LiveInputReader(Reader):
    """
    Reader for unannotated 'live' data that we want to parse.

    The data is assumed to be in a directory with one aa/ac file
    pair.

    There is no notion of subdocument (`subdoc = None`) and the
    stage is `'unannotated'`
    """

    def __init__(self, dir):
        Reader.__init__(self, dir)

    def files(self):
        corpus = {}
        for aa in glob(os.path.join(self.rootdir, '*.aa')):
            prefix = os.path.splitext(aa)[0]
            pair   = (aa, prefix + '.ac')
            k   = educe.corpus.FileId(doc=os.path.basename(prefix),
                                      subdoc=None,
                                      stage='unannotated',
                                      annotator=None)
            corpus[k] = pair
        return corpus

def id_to_path(k):
    """
    Given a fleshed out FileId (none of the fields are None),
    return a filepath for it following STAC conventions.

    You will likely want to add your own filename extensions to
    this path
    """
    for field in [ "doc", "stage" ]:
        if k.__dict__[field] is None:
            raise Exception('Need all FileId fields to be set (%s is unset)' % field)
    root = k.doc
    if k.subdoc is not None:
        root += '_' + k.subdoc
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

STAC_GLOZZ_FS_ORDER =\
    [ 'Status'
    , 'Quantity'
    , 'Correctness'
    , 'Kind'
    , 'Comments'
    , 'Developments'
    , 'Emitter'
    , 'Identifier'
    , 'Timestamp'
    , 'Resources'
    , 'Trades'
    , 'Dice_rolling'
    , 'Gets'
    , 'Has_resources'
    , 'Amount_of_resources'
    , 'Addressee'
    , 'Surface_act'
    ]
STAC_UNANNOTATED_FS_ORDER =\
    [ 'Status'
    , 'Quantity'
    , 'Correctness'
    , 'Kind'
    , 'Identifier'
    , 'Timestamp'
    , 'Emitter'
    , 'Resources'
    , 'Developments'
    , 'Comments'
    , 'Dice_rolling'
    , 'Gets'
    , 'Trades'
    , 'Has_resources'
    , 'Amount_of_resources'
    , 'Addressee'
    , 'Surface_act'
    ]

STAC_MD_ORDER =\
    [ 'author'
    , 'creation-date'
    , 'lastModifier'
    , 'lastModificationDate'
    ]

stac_output_settings =\
        glozz.GlozzOutputSettings(STAC_GLOZZ_FS_ORDER,
                                  STAC_MD_ORDER)

stac_unannotated_output_settings =\
        glozz.GlozzOutputSettings(STAC_UNANNOTATED_FS_ORDER,
                                  STAC_MD_ORDER)

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

    :param k: document key; used to create identifiers
    :type  k: FileId

    :param partial_units:
    :type  partial_units: iterable of `PartialUnit`
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
        # Note that Glozz seems to identify items by the pair of author and creation
        # date, ignoring the unit ID altogether (assumed to be author_date)
        creation_date = mk_creation_date(i)
        metadata = { 'author'        : author
                   , 'creation-date' : creation_date
                   , 'lastModifier'  : 'n/a'
                   , 'lastModificationDate' : '0'
                   }
        unit_id = '_'.join([author,str(i)])
        return glozz.GlozzUnit(unit_id, x.span, x.type, x.features, metadata)

    return [ mk_unit(x,i) for x,i in itertools.izip(partial_units, itertools.count(0)) ]

def write_annotation_file(anno_filename, doc):
    """
    Write a GlozzDocument to XML in the given path
    """
    glozz.write_annotation_file(anno_filename, doc, settings=stac_output_settings)
