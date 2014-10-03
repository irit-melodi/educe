# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Experimental sandbox (ignore)
"""

from __future__ import print_function
import educe.stac
import educe.stac.annotation

from ..args import\
    add_usual_output_args,\
    get_output_dir, announce_output_dir
from ..output import save_document

NAME = 'tmp'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    parser.add_argument('corpus', metavar='DIR', help='corpus dir')
    # don't allow stage control; must be units
    educe.util.add_corpus_filters(parser,
                                  fields=['doc', 'subdoc', 'annotator'])
    add_usual_output_args(parser)
    parser.set_defaults(func=main)

# not the same as educe.stac.annotation
RENAMES = {'Strategic_comment': 'Other'}


def read_corpus_at_stage(args, stage, verbose=True):
    """
    Read the section of the corpus specified in the command line arguments.
    """
    is_interesting0 = educe.util.mk_is_interesting(args)
    is_interesting = lambda k: is_interesting0(k) and k.stage == stage
    reader = educe.stac.Reader(args.corpus)
    anno_files = reader.filter(reader.files(), is_interesting)
    return reader.slurp(anno_files, verbose)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """

    corpus = read_corpus_at_stage(args, 'units')
    output_dir = get_output_dir(args)
    for k in corpus:
        doc = corpus[k]
        for edu in filter(educe.stac.is_edu, doc.units):
            etypes = frozenset(educe.stac.split_type(edu))
            etypes2 = frozenset(RENAMES.get(t, t) for t in etypes)
            if etypes != etypes2:
                edu.type = "/".join(sorted(etypes2))
        save_document(output_dir, k, doc)
    announce_output_dir(output_dir)

