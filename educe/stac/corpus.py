# Author: Eric Kow
# License: BSD3

"""
Corpus layout conventions (re-exported by educe.stac)
"""

from collections import OrderedDict
from glob import glob
import os
import re
import sys

from educe.corpus import FileId
import educe.corpus
import educe.glozz as glozz
from .annotation import STAC_OUTPUT_SETTINGS

# pylint: disable=too-few-public-methods


class Reader(educe.corpus.Reader):
    """
    See `educe.corpus.Reader` for details
    """
    def __init__(self, corpusdir):
        educe.corpus.Reader.__init__(self, corpusdir)

    def files(self):
        corpus = OrderedDict()
        full_glob = os.path.join(self.rootdir, '*')
        anno_glob = '*.aa'

        def register(stage, annotator, anno_file):
            """
            Determine annotation key for annotation file and helper
            file for the given annotation file, and update the corpus
            dictionary accordingly
            """
            prefix = os.path.splitext(anno_file)[0]
            subdoc = os.path.basename(prefix)
            if "_" in subdoc:
                subdoc = subdoc.rsplit("_", 1)[1]
                file_id = FileId(doc, subdoc, stage, annotator)
                ac_file_id = FileId(doc, subdoc, 'unannotated', None)
                text_file = os.path.join(self.rootdir,
                                         id_to_path(ac_file_id))\
                    + ".ac"
            else:
                raise Exception("STAC corpus filenames should be in"
                                "the form doc_subdocument: %s", subdoc)
            corpus[file_id] = (anno_file, text_file)

        for doc_dir in sorted(glob(full_glob)):
            doc = os.path.basename(doc_dir)
            for stage in ['unannotated', 'units', 'discourse']:
                stage_dir = os.path.join(doc_dir, stage)
                if stage == 'unannotated':
                    unannotated_files = glob(os.path.join(stage_dir,
                                                          anno_glob))
                    for anno_file in unannotated_files:
                        register(stage, None, anno_file)
                elif os.path.exists(stage_dir):
                    for annotator in os.listdir(stage_dir):
                        anno_dir = os.path.join(stage_dir, annotator)
                        anno_files = glob(os.path.join(anno_dir, anno_glob))
                        for anno_file in anno_files:
                            register(stage, annotator, anno_file)
        return corpus

    def slurp_subcorpus(self, cfiles, verbose=False):
        corpus = {}
        counter = 0
        for k in cfiles.keys():
            if verbose:
                sys.stderr.write("\rSlurping corpus dir [%d/%d]" %
                                 (counter, len(cfiles)))
            annotations = glozz.read_annotation_file(*cfiles[k])
            annotations.set_origin(k)
            corpus[k] = annotations
            counter = counter+1
        if verbose:
            sys.stderr.write("\rSlurping corpus dir [%d/%d done]\n" %
                             (counter, len(cfiles)))
        return corpus


class LiveInputReader(Reader):
    """
    Reader for unannotated 'live' data that we want to parse.

    The data is assumed to be in a directory with one aa/ac file
    pair.

    There is no notion of subdocument (`subdoc = None`) and the
    stage is `'unannotated'`
    """

    def __init__(self, corpusdir):
        Reader.__init__(self, corpusdir)

    def files(self):
        corpus = {}
        for anno_file in glob(os.path.join(self.rootdir, '*.aa')):
            prefix = os.path.splitext(anno_file)[0]
            pair = (anno_file, prefix + '.ac')
            k = educe.corpus.FileId(doc=os.path.basename(prefix),
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
    for field in ["doc", "stage"]:
        if k.__dict__[field] is None:
            raise Exception("Need all FileId fields to be set"
                            " (%s is unset)" % field)
    root = k.doc
    if k.subdoc is not None:
        root += '_' + k.subdoc
    pathparts = [k.doc, k.stage]
    if k.annotator is not None:
        pathparts.append(k.annotator)
    elif k.stage in ["units", "discourse"]:
        raise Exception("FileId.annotator must be set for "
                        "unit/discourse items")
    pathparts.append(root)
    # pylint:disable=W0142
    # Is there a better way to express this?
    return os.path.join(*pathparts)


def write_annotation_file(anno_filename, doc):
    """
    Write a GlozzDocument to XML in the given path
    """
    glozz.write_annotation_file(anno_filename,
                                doc,
                                settings=STAC_OUTPUT_SETTINGS)


METAL_REVIEWERS = ["bronze", "silver", "gold"]
METAL_STR = "(?i)" + "|".join(METAL_REVIEWERS)
METAL_RE = re.compile(METAL_STR)


def is_metal(fileid):
    "If the annotator is one of the distinguished standard annotators"
    anno = fileid.annotator or ""
    return anno.lower() in METAL_REVIEWERS
