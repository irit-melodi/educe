# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Replace Strategic_comment with Other
"""

from __future__ import print_function

from educe.util import add_corpus_filters, fields_without
import educe.stac
import educe.stac.annotation

from educe.stac.util.args import\
    (read_corpus, add_usual_output_args,
     get_output_dir, announce_output_dir)
from educe.stac.util.output import save_document

NAME = 'clean-dialogue-acts'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    parser.add_argument('corpus', metavar='DIR',
                        nargs='?',
                        help='corpus dir')
    add_corpus_filters(parser, fields=fields_without(["stage"]))
    add_usual_output_args(parser, default_overwrite=True)
    parser.set_defaults(func=main)

# not the same as educe.stac.annotation
RENAMES = {'Strategic_comment': 'Other'}


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """

    corpus = read_corpus(args,
                         preselected={"stage": ["units"]})
    output_dir = get_output_dir(args, default_overwrite=True)
    for k in corpus:
        doc = corpus[k]
        for edu in [x for x in doc.units if educe.stac.is_edu(x)]:
            etypes = frozenset(educe.stac.split_type(edu))
            etypes2 = frozenset(RENAMES.get(t, t) for t in etypes)
            if etypes != etypes2:
                edu.type = "/".join(sorted(etypes2))
        save_document(output_dir, k, doc)
    announce_output_dir(output_dir)
