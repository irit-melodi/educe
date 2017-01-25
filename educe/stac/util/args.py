# Author Eric Kow
# License: CeCILL-B (BSD3-ish)

"""
Command line options
"""

from __future__ import print_function
import argparse
import glob
import os
import sys
import tempfile

from educe.stac.corpus import METAL_STR
import educe.annotation
import educe.stac
import educe.util

STAC_GLOBS = frozenset([
    "data/pilot",
    "data/socl-season1",
    "data/socl-season2",
    "data/TEST",
])


def check_easy_settings(args):
    """Modify args to reflect user-friendly defaults.

    Terminates the program if `args.corpus` is set but does not point to
    an existing folder ; otherwise `args.doc` must be set and everything
    else is expected to be empty.

    Notes
    -----
    All callers for this function are in the `scripts` folder of the
    educe repository: scripts/stac-{util,edit,oneoff}.

    Parameters
    ----------
    args : Namespace
        Arguments of the argparser.

    See also
    --------
    `educe.stac.sanity.main.easy_settings()`
    """
    if args.corpus:
        # 2017-01-25 explicitly break if there is no such folder
        if not os.path.exists(args.corpus):
            sys.exit("No corpus directory {corpus}".format(
                corpus=args.corpus))
        # end explicitly break if no such folder
        return  # not easy mode

    if not args.doc:
        raise ValueError("no document specified for easy mode")

    # figure out where this thing lives
    for sdir in STAC_GLOBS:
        if glob.glob(os.path.join(sdir, args.doc)):
            args.corpus = sdir
    if not args.corpus:
        if not any(os.path.isdir(x) for x in STAC_GLOBS):
            sys.exit("You don't appear to be in the STAC root dir")
        else:
            sys.exit("I don't know about any document called " + args.doc)
    # prepare info message for user
    guess_report = "{corpus} --doc \"{doc}\""

    # annotator
    want_unanno = ('stage' in args.__dict__ and
                   args.stage is not None and
                   'unannotated'.startswith(args.stage))
    if ('annotator' in args.__dict__ and
            not want_unanno and
            args.annotator is None):
        args.annotator = METAL_STR
        guess_report += ' --annotator "{}"'.format(args.annotator)

    print("Guessing convenience settings:", file=sys.stderr)
    print(guess_report.format(**args.__dict__), file=sys.stderr)


def read_corpus(args, preselected=None, verbose=True):
    """
    Read the section of the corpus specified in the command line arguments.
    """
    is_interesting = educe.util.mk_is_interesting(args,
                                                  preselected=preselected)
    reader = educe.stac.Reader(args.corpus)
    anno_files = reader.filter(reader.files(), is_interesting)
    return reader.slurp(anno_files, verbose)


def read_corpus_with_unannotated(args, verbose=True):
    """
    Read the section of the corpus specified in the command line arguments.
    """
    reader = educe.stac.Reader(args.corpus)
    all_files = reader.files()
    is_interesting = educe.util.mk_is_interesting(args)
    anno_files = reader.filter(all_files, is_interesting)
    unannotated_twins = frozenset(educe.stac.twin_key(k, 'unannotated')
                                  for k in anno_files)
    for key in unannotated_twins:
        if key in all_files:
            anno_files[key] = all_files[key]
    return reader.slurp(anno_files, verbose=verbose)


def get_output_dir(args, default_overwrite=False):
    """Return the output dir specified or inferred from command
    line args.

    We try the following in order:

    1. If `--output` is given explicitly, we'll just use/create that
    2. If `default_overwrite` is True, or the user specifies `--overwrite`
       on the command line (provided the command supports it),
       the output directory may well be the original corpus dir
       (*gulp*! Better use version control!)
    3. OK just make a temporary directory. Later on, you'll probably want
       to call `announce_output_dir`.
    """
    if args.output:
        if os.path.isfile(args.output):
            oops = "Sorry, %s already exists and is not a directory" %\
                args.output
            sys.exit(oops)
        elif not os.path.isdir(args.output):
            os.makedirs(args.output)
        return args.output
    elif (default_overwrite or
          ("overwrite_input" in args.__dict__ and args.overwrite_input)):
        return args.corpus
    else:
        return tempfile.mkdtemp()


def announce_output_dir(output_dir):
    """
    Tell the user where we saved the output
    """
    print("Output files written to", output_dir, file=sys.stderr)


def add_commit_args(parser):
    """
    Augment a subcommand argparser with an option to emit a commit
    message for your version control tracking
    """
    parser.add_argument('--no-commit-msg', action='store_true',
                        help='Skip commit message summary')


def add_usual_input_args(parser, doc_subdoc_required=False, help_suffix=None):
    """Augment a subcommand argparser with typical input arguments.
    Sometimes your subcommand may require slightly different input
    arguments, in which case, just don't call this function.

    Parameters
    ----------
    parser : ArgumentParser
        Argument parser.

    doc_subdoc_required : bool, defaults to False
        force user to supply --doc/--subdoc
        for this subcommand (note you'll need to add stage/anno
        yourself)

    help_suffix : string, defaults to None
        appended to --doc/--subdoc help strings
    """
    parser.add_argument('corpus', metavar='DIR', nargs='?',
                        help='corpus dir')
    if doc_subdoc_required:
        doc_help = 'document'
        subdoc_help = 'subdocument'
        if help_suffix:
            doc_help = doc_help + ' ' + help_suffix
            subdoc_help = subdoc_help + ' ' + help_suffix
        parser.add_argument('--doc', metavar='DOC',
                            help=doc_help, required=doc_subdoc_required)
        parser.add_argument('--subdoc', metavar='SUBDOC',
                            help=subdoc_help, required=doc_subdoc_required)
    else:
        educe.util.add_corpus_filters(parser)


def add_usual_output_args(parser, default_overwrite=False):
    """
    Augment a subcommand argparser with typical output arguments,
    Sometimes your subcommand may require slightly different output
    arguments, in which case, just don't call this function.
    """
    default = '(default {})'.format('overwrite!' if default_overwrite
                                    else 'mktemp')
    parser.add_argument('--output', '-o', metavar='DIR',
                        help='output directory ' + default)
    parser.add_argument('--overwrite-input', action='store_true',
                        help='save results back to input dir')


def anno_id(string):
    """
    Split AUTHOR_DATE string into tuple, complaining if we don't have such a
    string. Used for argparse
    """
    parts = string.split('_')
    if len(parts) != 2:
        msg = "%r is not of form author_date" % string
        raise argparse.ArgumentTypeError(msg)
    return (parts[0], int(parts[1]))


def comma_span(string):
    """
    Split a comma delimited pair of integers into an educe span
    """
    parts = [int(x) for x in string.split(',')]
    if len(parts) != 2:
        msg = "%r is not of form n,m" % string
        raise argparse.ArgumentTypeError(msg)
    return educe.annotation.Span(parts[0], parts[1])
