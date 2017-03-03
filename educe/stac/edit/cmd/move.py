# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Move a block of text from one doc to another

The typical use case here would in reportioning a document
"""

from __future__ import print_function
import copy
import sys

import educe.stac

from educe.stac.util.annotate import show_diff
from educe.stac.util.args import\
    (add_usual_input_args, add_usual_output_args,
     comma_span,
     get_output_dir, announce_output_dir)
from educe.stac.util.doc import\
    (compute_renames, evil_set_text, move_portion, split_doc)
from educe.stac.util.output import save_document


def is_target(args):
    """
    Corpus filter to pick out the part we want to move to
    """
    def is_match(k):
        "is a target entry"
        return k.doc == args.doc and k.subdoc == args.target
    return is_match


def is_requested(args):
    """
    Corpus filter to pick out the part we want to move from
    """
    def is_match(k):
        "is a source entry"
        return k.doc == args.doc and k.subdoc == args.subdoc
    return is_match


def read_source_corpus(args):
    """
    Read the part of the corpus that we want to move from
    """
    reader = educe.stac.Reader(args.corpus)
    src_files = reader.filter(reader.files(), is_requested(args))
    return reader.slurp(src_files)


def read_target_corpus(args):
    """
    Read the part of the corpus that we want to move to
    """
    reader = educe.stac.Reader(args.corpus)
    tgt_files = reader.filter(reader.files(), is_target(args))
    return reader.slurp(tgt_files)

# ---------------------------------------------------------------------
# command and options
# ---------------------------------------------------------------------

NAME = 'move'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser, doc_subdoc_required=True,
                         help_suffix='to move from')
    add_usual_output_args(parser, default_overwrite=True)
    parser.add_argument('span', metavar='INT,INT', type=comma_span,
                        help='Text span')
    parser.add_argument('target', metavar='SUBDOC')
    parser.set_defaults(func=main)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    output_dir = get_output_dir(args, default_overwrite=True)
    start = args.span.char_start
    end = args.span.char_end

    src_corpus = read_source_corpus(args)
    tgt_corpus = read_target_corpus(args)

    renames = compute_renames(tgt_corpus, src_corpus)

    for src_k, src_doc in src_corpus.items():
        # retrieve target subdoc
        tgt_k = copy.copy(src_k)
        tgt_k.subdoc = args.target
        print(src_k, tgt_k, file=sys.stderr)
        if tgt_k not in tgt_corpus:
            raise ValueError("Uh-oh! we don't have %s in the corpus" % tgt_k)
        tgt_doc = tgt_corpus[tgt_k]
        # move portion from source to target subdoc
        if start == 0:
            # move up
            new_src_doc, new_tgt_doc = move_portion(
                renames, src_doc, tgt_doc,
                end,  # src_split
                tgt_split=-1)
        elif end == len(src_doc.text()):  # src_doc.text_span().char_end:
            # move down
            # move_portion inserts src_doc[0:src_split] between
            # tgt_doc[0:tgt_split] and tgt_doc[tgt_split:],
            # so we detach src_doc[start:] into a temporary doc,
            # then call move_portion on this temporary doc
            new_src_doc, src_doc2 = split_doc(src_doc, start)
            _, new_tgt_doc = move_portion(
                renames, src_doc2, tgt_doc,
                -1,  # src_split
                tgt_split=0)
            # the whitespace between new_src_doc and src_doc2 went to
            # src_doc2, so we need to append a new whitespace to new_src_doc
            evil_set_text(new_src_doc, new_src_doc.text() + ' ')
        else:
            raise ValueError("Sorry, can only move to the start or to the "
                             "end of a document at the moment")
        # print diff for suggested commit message
        diffs = ["======= TO %s   ========" % tgt_k,
                 show_diff(tgt_doc, new_tgt_doc),
                 "^------ FROM %s" % src_k,
                 show_diff(src_doc, new_src_doc),
                 ""]
        print("\n".join(diffs), file=sys.stderr)
        # dump the modified documents
        save_document(output_dir, src_k, new_src_doc)
        save_document(output_dir, tgt_k, new_tgt_doc)

    announce_output_dir(output_dir)
