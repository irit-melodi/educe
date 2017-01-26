# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Insert new text into a portion
"""

from __future__ import print_function
import sys

import educe.stac

from educe.stac.util.annotate import show_diff
from educe.stac.util.args import\
    (add_usual_input_args,
     add_usual_output_args,
     announce_output_dir,
     get_output_dir)
from educe.stac.util.output import save_document
from educe.stac.util.doc import\
    compute_renames, move_portion
from .move import is_requested


# ---------------------------------------------------------------------
# command and options
# ---------------------------------------------------------------------
def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser, doc_subdoc_required=True,
                         help_suffix='to insert into')
    add_usual_output_args(parser, default_overwrite=True)
    parser.add_argument('insert', metavar='DIR',
                        help='dir with just one pair of .aa/.ac files')
    parser.add_argument('start', metavar='INT', type=int,
                        help='insert before Nth char (negative indices ok)')
    parser.set_defaults(func=main)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    output_dir = get_output_dir(args, default_overwrite=True)

    src_reader = educe.stac.LiveInputReader(args.insert)
    src_corpus = src_reader.slurp(src_reader.files())

    if not src_corpus:
        sys.exit("Insert dir must have exactly one .aa/.ac pair (none found)")
    elif len(src_corpus) > 1:
        sys.exit("Insert dir must have exactly one .aa/.ac pair (%d found)" %
                 len(src_corpus))

    src_doc = src_corpus.values()[0]

    reader = educe.stac.Reader(args.corpus)
    tgt_files = reader.filter(reader.files(), is_requested(args))
    tgt_corpus = reader.slurp(tgt_files)

    renames = compute_renames(tgt_corpus, src_corpus)
    for tgt_k in tgt_corpus:
        tgt_doc = tgt_corpus[tgt_k]
        _, new_tgt_doc = move_portion(renames,
                                      src_doc,
                                      tgt_doc,
                                      -1,
                                      tgt_split=args.start)
        diffs = ["======= INSERT IN %s   ========" % tgt_k,
                 show_diff(tgt_doc, new_tgt_doc)]
        print("\n".join(diffs).encode('utf-8'), file=sys.stderr)
        save_document(output_dir, tgt_k, new_tgt_doc)

    announce_output_dir(output_dir)
