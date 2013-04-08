# Author: Eric Kow
# License: BSD3

"""
Miscellaneous utility functions
"""

import argparse
import re

def add_corpus_filters(arg_parser):
    """
    For help with script-building:

    Augment an argparer with options to filter a corpus on
    the various attributes in a 'educe.corpus.FileId'
    (eg, document, annotator).

    Meant to be used in conjunction with `mk_is_interesting`
    """

    for x in [ 'stage', 'doc', 'subdoc', 'annotator' ]:
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
                 [ lambda x:x.stage
                 , lambda x:x.doc
                 , lambda x:x.subdoc
                 , lambda x:x.annotator ])

    return lambda k : all([check(k) for check in doc_checkers])
