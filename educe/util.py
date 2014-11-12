# Author: Eric Kow
# License: BSD3

"""
Miscellaneous utility functions
"""

from itertools import chain
import re


def concat(items):
    ":: iter(iter(a)) -> iter(a)"
    return chain.from_iterable(items)


def concat_l(items):
    ":: [[a]] -> [a]"
    return list(chain.from_iterable(items))


FILEID_FIELDS = ['stage', 'doc', 'subdoc', 'annotator']
"""
String representation of fields recognised in an educe.corpus.FileId
"""


def fields_without(unwanted):
    """
    Fields for `add_corpus_filters` without the unwanted members
    """
    return [x for x in FILEID_FIELDS if x not in unwanted]


def add_corpus_filters(parser,
                       fields=None,
                       choice_fields=None):
    """
    For help with script-building:

    Augment an argparer with options to filter a corpus on
    the various attributes in a 'educe.corpus.FileId'
    (eg, document, annotator).

    :param fields: which flag names to include (defaults
    to `FILEID_FIELDS`)
    :type fields: [String]

    :param choice_fields: fields which accept a limited range
    of answers
    :type fields: Dict String [String]

    Meant to be used in conjunction with `mk_is_interesting`
    """
    def add(flag, choices):
        "add a flag"
        parser.add_argument("--{}".format(flag),
                            choices=choices,
                            metavar="PY_REGEX" if choices is None else None,
                            help="Limit to a particular {}(s)".format(field))

    fields = FILEID_FIELDS if fields is None else fields
    choice_fields = {} if choice_fields is None else choice_fields

    for field, choices in choice_fields.items():
        add(field, choices)

    for field in fields:
        if field not in choice_fields:
            add(field, choices=None)


def mk_is_interesting(args,
                      preselected=None):
    """
    Return a function that when given a FileId returns 'True'
    if the FileId would be considered interesting according to
    the arguments passed in.

    :param preselected: fields for which we already know what
    matches we want
    :type fields: Dict String [String]

    Meant to be used in conjunction with `add_corpus_filters`
    """

    def mk_checker_generic(attr, pred):
        "generalised helper for mk_checker"
        def check(fileid):
            "matching on k value"
            val = fileid.__dict__[attr]
            return False if val is None else pred(val)
        return check

    def mk_checker(attr):
        """
        Given an attr name, return a function that checks a FileId
        to see if its attribution value matches the requested pattern.
        If the attribute was not requested, we skip the check.
        """
        if attr in preselected:
            pred = lambda v: v in preselected[attr]
            return mk_checker_generic(attr, pred)
        elif args.__dict__.get(attr) is None:
            return lambda _: True
        else:
            argval = args.__dict__[attr]
            regex = re.compile(argval)
            return mk_checker_generic(attr, regex.match)

    doc_checkers = [mk_checker(attr) for attr in FILEID_FIELDS]
    return lambda k: all(check(k) for check in doc_checkers)
