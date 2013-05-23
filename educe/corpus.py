# Author: Eric Kow
# License: BSD3

"""
Corpus management
"""
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

import sys

class FileId:
    """
    Information needed to uniquely identify an annotation file.

    Note that this includes the annotator, so if you want to do
    comparisons on the "same" file between annotators you'll want
    to ignore this field.
    """
    def __init__(self, doc, subdoc, stage, annotator):
       self.doc=doc
       self.subdoc=subdoc
       self.stage=stage
       self.annotator=annotator

    def __str__(self):
        return "%s [%s] %s %s" % (self.doc, self.subdoc, self.stage, self.annotator)


    def _tuple(self):
        """
        For internal use by __hash__, __eq__, etc
        """
        return (self.doc, self.subdoc, self.stage, self.annotator)

    def __hash__(self):
        return hash(self._tuple())

    def __eq__(self, other):
        return self._tuple() == other._tuple()


    def mk_global_id(self, local_id):
        """
        String representation of an identifier that should be unique
        to this corpus at least.

        If the unit has an origin (see "FileId"), we use the

        * document
        * subdocument
        * stage
        * (but not the annotator!)
        * and the id from the XML file

        If we don't have an origin we fall back to just the id provided
        by the XML file

        See also `position` as potentially a safer alternative to this
        (and what we mean by safer)
        """
        return "_".join([self.doc, self.subdoc, self.stage, local_id])

class Reader:
    """
    `Reader` provides little more than dictionaries from `FileId`
    to data.

    A potentially useful pattern to apply here is to take a slice of
    these dictionaries for processing. For example, you might not want
    to read the whole corpus, but only the files which are modified by
    certain annotators.

    .. code-block:: python

        reader    = Reader(corpus_dir)
        files     = reader.files()
        subfiles  = { k:v in files.items() if k.annotator in [ 'Bob', 'Alice' ] }
        corpus    = reader.slurp(subfiles)

    Alternatively, having read in the entire corpus, you might be doing
    processing on various slices of it at a time

    .. code-block:: python

        corpus    = reader.slurp()
        subcorpus = { k:v in corpus.items() if k.doc = 'pilot14' }

    This is an abstract class; you should use the version from a
    data-set, eg. `educe.stac.Reader` instead

    Fields:

        * rootdir - the top directory of the corpus
    """
    def __init__(self, dir):
        self.rootdir=dir

    def files():
        """
        Return a dictionary from FileId to (tuples of) filepaths.
        The tuples correspond to files that are considered to 'belong'
        together; for example, in the case of standoff annotation, both
        the text file and its annotations

        Derived classes
        """

    def slurp(self, cfiles=None, verbose=False):
        """
        Read the entire corpus if `cfiles` is `None` or else the
        subset specified by `cfiles`.

        Return a dictionary from FileId to `educe.Annotation.Document`

        Kwargs:

            cfiles (dict): a dictionary like what `Corpus.files` would return.

            verbose (bool)
        """
        if cfiles is None:
            subcorpus=self.files()
        else:
            subcorpus=cfiles
        return self.slurp_subcorpus(subcorpus, verbose)

    def slurp_subcorpus(self, cfiles, verbose=False):
        """
        Derived classes should implement this function
        """
        return {}

    def filter(self, d, pred):
        """
        Convenience function equivalent to ::

            { k:v for k,v in d.items() if pred(k) }
        """
        return dict([(k,v) for k,v in d.items() if pred(k)])

