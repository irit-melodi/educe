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
    def mk_checker(attr):
        """
        Given an attr name, return a function that checks a FileId
        to see if its attribution value matches the requested pattern.
        If the attribute was not requested, we skip the check.
        """

        if attr in args.__dict__:
            argval = args.__dict__[attr]
            regex = re.compile(argval)
            def check(fileid):
                "regex matching on k value"
                val = fileid.__dict__[attr]
                if val is None:
                    return False
                else:
                    return regex.match(val)
            return check
        else:
            return lambda _ : True

    doc_checkers = [mk_checker(attr) for attr in
                    ['stage', 'doc', 'subdoc', 'annotator']]
    return lambda k : all(check(k) for check in doc_checkers)
