# Author: Eric Kow
# License: BSD3

"""
Miscellaneous utility functions
"""

import argparse
import re

fileid_fields = [ 'stage', 'doc', 'subdoc', 'annotator' ]
"""
String representation of fields recognised in an educe.corpus.FileId
"""

def add_corpus_filters(arg_parser, fields=fileid_fields):
    """
    For help with script-building:

    Augment an argparer with options to filter a corpus on
    the various attributes in a 'educe.corpus.FileId'
    (eg, document, annotator).

    Meant to be used in conjunction with `mk_is_interesting`
    """

    for x in fields:
        arg_parser.add_argument( ('--%s' % x)
                               , metavar='PY_REGEX'
                               , help=('Limit to a particular %s(s)' % x)
                               )

def mk_is_interesting(args):
    """
    Return a function that when given a FileId returns 'True'
    if the FileId would be considered interesting according to
    the arguments passed in.

    Meant to be used in conjunction with `add_corpus_filters`
    """
    def mk_checker(fn):
        if fn(args) is None:
            return lambda _ : True
        else:
            r = re.compile(fn(args))
            def check(k):
                val = fn(k)
                if val is None:
                    return False
                else:
                    return r.match(val)
            return check

    doc_checkers=map(mk_checker,
                 [ lambda x:x.stage  if 'stage' in x.__dict__ else None
                 , lambda x:x.doc    if 'doc'   in x.__dict__ else None
                 , lambda x:x.subdoc if 'subdoc' in x.__dict__ else None
                 , lambda x:x.annotator if 'annotator' in x.__dict__ else None])

    return lambda k : all([check(k) for check in doc_checkers])
