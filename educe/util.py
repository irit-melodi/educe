# Author: Eric Kow
# License: BSD3

"""
Miscellaneous utility functions
"""

from itertools import chain, groupby
import re


def concat(items):
    ":: Iterable (Iterable a) -> Iterable a"
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


def add_corpus_filters(parser, fields=None, choice_fields=None):
    """
    For help with script-building:

    Augment an argparser with options to filter a corpus on
    the various attributes in a 'educe.corpus.FileId'
    (eg, document, annotator).

    :param fields: which flag names to include (defaults to `FILEID_FIELDS`)
    :type fields: [String]
    :param choice_fields: fields which accept a limited range of answers
    :type choice_fields: Dict String [String]

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


def mk_is_interesting(args, preselected=None):
    """
    Return a function that when given a FileId returns 'True'
    if the FileId would be considered interesting according to
    the arguments passed in.

    :param preselected: fields for which we already know what
                        matches we want
    :type preselected: Dict String [String]

    Meant to be used in conjunction with `add_corpus_filters`
    """
    preselected = preselected or {}

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


def add_subcommand(subparsers, module):
    '''
    Add a subcommand to an argparser following some conventions:

        - the module can have an optional NAME constant
          (giving the name of the command); otherwise we
          assume it's the unqualified module name
        - the first line of its docstring is its help text
        - subsequent lines (if any) form its epilog

    Returns the resulting subparser for the module
    '''

    if 'NAME' in module.__dict__:
        module_name = module.NAME
    else:
        module_name = module.__name__.split('.')[-1]

    module_help_parts = [x for x in module.__doc__.strip().split('\n', 1)
                         if x]
    if len(module_help_parts) > 1:
        module_help = module_help_parts[0]
        module_epilog = '\n'.join(module_help_parts[1:]).strip()
    else:
        module_help = module.__doc__
        module_epilog = None
    return subparsers.add_parser(module_name,
                                 help=module_help,
                                 epilog=module_epilog)


# To rewrite using np.{ediff1d, where, ...} when we go full numpy
def relative_indices(group_indices, reverse=False, valna=None):
    """Generate a list of relative indices inside each group.
    Missing (None) values are handled specifically: each missing
    value is mapped to `valna`.

    Parameters
    ----------
    reverse: boolean, optional
        If True, compute indices relative to the end of each group.
    valna: int or None, optional
        Relative index for missing values.
    """
    if reverse:
        group_indices = list(group_indices)
        group_indices.reverse()

    result = []
    for group_idx, dup_values in groupby(group_indices):
        if group_idx is None:
            rel_indices = (valna for dup_value in dup_values)
        else:
            rel_indices = (rel_idx for rel_idx, dv in enumerate(dup_values))
        result.extend(rel_indices)

    if reverse:
        result.reverse()

    return result
