# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""Delete an annotation"""

from __future__ import print_function
import sys

from educe.stac.util.args import\
    add_usual_input_args, add_usual_output_args,\
    read_corpus, get_output_dir, announce_output_dir,\
    anno_id
from educe.stac.util.glozz import anno_id_from_tuple, anno_id_to_tuple
from educe.stac.util.output import save_document



def _is_match(wanted):
    """
    Given an annotation id, return a predicate that checks if
    an annotation id matches
    """
    def pred(anno):
        "curried second arg"
        return anno_id_to_tuple(anno.local_id()) == wanted
    return pred


def _delete_in_doc(del_id, doc):
    """Delete the annotations with the given id in the given document

    NB: modifies doc
    """
    pretty_id = anno_id_from_tuple(del_id)
    is_ok = lambda x: not _is_match(del_id)(x)
    matches = [x for x in doc.annotations() if not is_ok(x)]

    if not matches:
        print("Skipping... no annotations found with id %s" % pretty_id,
              file=sys.stderr)
        return
    elif len(matches) > 1:
        sys.exit("Huh?! More than one annotation with id %s" % pretty_id)

    doc.units = [x for x in doc.units if is_ok(x)]
    doc.relations = [x for x in doc.relations if is_ok(x)]
    doc.schemas = [x for x in doc.schemas if is_ok(x)]

    def oops(reason):
        "quit because of illegal delete"
        sys.exit("Can't delete %s because %s " % pretty_id, reason)

    for anno in doc.relations:
        if anno.span.t1 == pretty_id:
            oops("it is the source for a relation: %s" % anno)
        if anno.span.t2 == pretty_id:
            oops("it is the target for a relation: %s" % anno)
    for anno in doc.schemas:
        if pretty_id in anno.units:
            oops("it is a unit member of %s" % anno)
        if pretty_id in anno.relations:
            oops("it is a relation member of %s" % anno)
        if pretty_id in anno.schemas:
            oops("it is a schema member of %s" % anno)

# ---------------------------------------------------------------------
# command and options
# ---------------------------------------------------------------------

NAME = 'delete-anno'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser, doc_subdoc_required=True)
    add_usual_output_args(parser, default_overwrite=True)
    parser.add_argument('anno_id', type=anno_id, metavar='ANNO_ID',
                        help='id to rename (eg. kowey_398190)')
    parser.add_argument('--stage', metavar='STAGE',
                        choices=['discourse', 'units', 'unannotated'])
    parser.add_argument('--annotator', metavar='STRING')
    parser.set_defaults(func=main)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    if args.stage:
        if args.stage != 'unannotated' and not args.annotator:
            sys.exit("--annotator is required unless --stage is unannotated")
        elif args.stage == 'unannotated' and args.annotator:
            sys.exit("--annotator is forbidden if --stage is unannotated")
    output_dir = get_output_dir(args, default_overwrite=True)
    corpus = read_corpus(args, verbose=True)

    for key in corpus:
        print(key)
        doc = corpus[key]
        _delete_in_doc(args.anno_id, doc)
        save_document(output_dir, key, doc)
    pretty_id = anno_id_from_tuple(args.anno_id)
    print("Deleted %s" % pretty_id, file=sys.stderr)
    announce_output_dir(output_dir)
