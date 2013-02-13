# Author: Eric Kow
# License: BSD3

# Corpus management
#
# A corpus consists of a set of annotation files organised by
#
# - document
# - subdocument
# - (annotation) stage
# - annotator
#
# We try to be somewhat agnostic to your directory structure.
# To this end we provide a FileId class which is a tuple of
# the above. Give us a mapping from FileId to filepaths and we
# do the rest.

# TODO: do we *really* want slurp_corpus to live here?
# Seems kind of yucky to have educe.glozz as an import
#

import sys

class FileId:
    def __init__(self, doc, subdoc, stage, annotator):
       self.doc=doc
       self.subdoc=subdoc
       self.stage=stage
       self.annotator=annotator

    def __str__(self):
        return "%s [%s] %s %s" % (self.doc, self.subdoc, self.stage, self.annotator)

def subcorpus(pattern, corpus):
    """
    Return the portion of a corpus for which matches the given file
    pattern.

    See the file_pattern function

    Hint: this works on any dictionary which uses file_id as its keys
    For example, if you wanted to avoid reading in the whole corpus,
    you could apply this function to take a slice of results from the
    corpus_files function so you're not reading all of them
    """
    corpus2=corpus.copy()
    for k in corpus.keys():
        if not matches_file_pattern(pattern, k):
            del corpus2[k]
    return corpus2

def file_pattern(doc=None, subdoc=None, stage=None, annotator=None):
    """
    A file pattern is can be fed into matches_file_pattern.
    Set as many fields as you know
    """
    return FileId(doc, subdoc, stage, annotator)

def matches_file_pattern(pattern, instance):
    """
    matches_file_pattern(pattern, instance) returns True if case
    the pattern is satisfied by instance
    """
    def match(f):
        return (f(pattern) is None or f(pattern) == f(instance))

    return (all(map(match,
                    [ lambda x : x.doc
                    , lambda x : x.subdoc
                    , lambda x : x.annotator
                    , lambda x : x.stage
                    ])))


