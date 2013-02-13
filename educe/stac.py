# Author: Eric Kow
# License: BSD3

# Things which are specific to the STAC project
# Not sure if these should really live in educe

from educe.corpus import *
from glob import glob
import os

def is_real_annotation(annotation):
    """
    the subset of annotations which come from an annotator,
    as opposed to be prefilled 'structural' annotations
    """
    blacklist=['Turn','paragraph','dialogue']
    return (annotation.type not in blacklist)

def corpus_files(dir, cglob='*', anno_glob='*.aa'):
    """
    Traverse the corpus, looking for files in directories that match
    a certain unix glob, eg

    corpus_files('data/pilot', cglob='pilot??')

    Return a dictionary mapping FileId data structures to filepaths
    """
    corpus={}
    full_glob=os.path.join(dir, cglob)
    for doc_dir in glob(full_glob):
        doc=os.path.basename(doc_dir)
        for stage in ['units', 'discourse']:
            stage_dir=os.path.join(doc_dir,stage)
            for annotator in os.listdir(stage_dir):
                annotator_dir=os.path.join(stage_dir,annotator)
                annotator_files=glob(os.path.join(annotator_dir, anno_glob))
                for f in annotator_files:
                    subdoc=os.path.splitext(os.path.basename(f))[0]
                    file_id=FileId(doc, subdoc, stage, annotator)
                    corpus[file_id]=f
    return corpus
