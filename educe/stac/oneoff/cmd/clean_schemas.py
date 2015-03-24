# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Delete empty CDUs
"""

from __future__ import print_function

from educe.util import add_corpus_filters, fields_without
from educe.stac.util.args import\
    (add_usual_output_args,
     read_corpus,
     get_output_dir, announce_output_dir)
from educe.stac.util.output import save_document


NAME = 'clean-schemas'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    parser.add_argument('corpus', metavar='DIR',
                        nargs='?',
                        help='corpus dir')
    # don't allow stage control
    add_corpus_filters(parser, fields=fields_without(["stage"]))
    add_usual_output_args(parser)
    parser.set_defaults(func=main)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    corpus = read_corpus(args,
                         preselected={'stage': ['discourse', 'units']})
    output_dir = get_output_dir(args)
    for key in corpus:
        doc = corpus[key]
        to_delete = []
        for sch in doc.schemas:
            if not sch.members:
                to_delete.append(sch)
        for sch in to_delete:
            doc.schemas.remove(sch)
        save_document(output_dir, key, doc)
    announce_output_dir(output_dir)
