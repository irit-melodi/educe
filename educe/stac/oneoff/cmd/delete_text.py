"""Delete a text span in a document.

"""

from __future__ import print_function
import copy
import sys

from educe.annotation import Span, Unit
import educe.stac
from educe.stac.edit.cmd.move import is_requested
from educe.stac.util.annotate import annotate_doc, show_diff
from educe.stac.util.args import (add_commit_args,
                                  add_usual_input_args,
                                  add_usual_output_args,
                                  announce_output_dir,
                                  comma_span,
                                  get_output_dir)
from educe.stac.util.doc import evil_set_text
from educe.stac.util.output import save_document


NAME = 'delete-text'


def delete_text_at_span(doc, span, minor=True):
    """Delete text at `span` in `doc`.

    Parameters
    ----------
    doc : Document
        Original document
    span : Span
        Span of the substitution site
    minor : boolean, default True
        If True, the text deletion is considered minor and annotations
        are kept as they are (with shifted spans) ; otherwise unit
        annotations and discourse relations are deleted.

    Returns
    -------
    doc2 : Document
        Updated document
    del_annos : set of Annotation
        Deleted annotations
    """
    def shift_anno(anno, offset, point):
        """Get a shifted copy of an annotation"""
        anno2 = copy.deepcopy(anno)
        if not isinstance(anno, Unit):
            return anno2

        anno_span = anno2.text_span()
        if anno_span.char_start >= point:
            # if the annotation is entirely after the deletion site,
            # shift the whole span
            anno2.span = anno_span.shift(offset)
        elif anno_span.char_end >= point:
            # if the annotation straddles the substitution site,
            # stretch (shift its end)
            anno2.span = Span(anno_span.char_start,
                              anno_span.char_end + offset)
        return anno2

    # compute text update
    old_txt = doc.text()
    new_txt = old_txt[:span.char_start] + old_txt[span.char_end:]
    offset = len(new_txt) - len(old_txt)
    point = span.char_end
    # create a copy of the doc
    doc2 = copy.copy(doc)
    evil_set_text(doc2, new_txt)
    # shift or stretch all units
    # if not minor, skip annotations in the subsitution site so that
    # they get lost
    if minor:
        doc2.units = [shift_anno(x, offset, point)
                      for x in doc.units]
        doc2.schemas = [shift_anno(x, offset, point)
                        for x in doc.schemas]
        doc2.relations = [shift_anno(x, offset, point)
                          for x in doc.relations]
    else:
        deleted_units = set(x for x in doc.units
                            if span.encloses(x.text_span()))
        # fixed point deletion of schemas and relations
        deleted_schms = set(x for x in doc.schemas
                            if any(y in deleted_units for y in x.members))
        deleted_rels = set(x for x in doc.relations
                           if (x.source in deleted_units or
                               x.target in deleted_units))
        new_del_schms = deleted_schms
        new_del_rels = deleted_rels
        while new_del_schms or new_del_rels:
            next_del_schms = set(x for x in doc.schemas
                                 if any(y in new_del_schms
                                        for y in x.members))
            next_del_rels = set(x for x in doc.relations
                                if (x.source in new_del_rels or
                                    x.target in new_del_rels))
            deleted_schms.update(next_del_schms)
            deleted_rels.update(next_del_rels)
            new_del_schms = next_del_schms
            new_del_rels = next_del_rels
        # shift everything we keep
        doc2.units = [shift_anno(x, offset, point) for x in doc.units
                      if x not in deleted_units]
        doc2.schemas = [shift_anno(x, offset, point) for x in doc.schemas
                        if x not in deleted_schms]
        doc2.relations = [shift_anno(x, offset, point) for x in doc.relations
                          if x not in deleted_rels]
    # TODO print deleted_units, deleted_rels, deleted_schms
    del_annos = (deleted_units, deleted_schms, deleted_rels)
    return doc2, del_annos


def commit_msg(key, anno_str_before, all_del_annos):
    """Generate a commit message describing the operation we just did.

    """
    lines = [
        "{}_{}: very scary edit (delete text)".format(
            key.doc, key.subdoc),
        "",
        "    " + anno_str_before,
        "==> " + '',
        ""
    ]

    if all_del_annos:
        lines.append("======= Deleted annotations =======")
    for tgt_k, del_annos in sorted(all_del_annos):
        lines.append(
            '------- {}{} -------'.format(
                tgt_k.stage,
                (' / ' + tgt_k.annotator if tgt_k.annotator is not None
                 else ''))
        )
        if del_annos[0]:
            lines.append(
                'Units: ' + ', '.join(sorted(
                    str(x.local_id()) for x in del_annos[0]))
            )
        if del_annos[1]:
            lines.append(
                'Schemas: ' + ', '.join(sorted(
                    str(x.local_id()) for x in del_annos[1]))
            )
        if del_annos[2]:
            lines.append(
                'Relations: ' + ', '.join(sorted(
                    str(x.local_id()) for x in del_annos[2]))
            )

    return "\n".join(lines)


def config_argparser(parser):
    """Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser, doc_subdoc_required=True,
                         help_suffix='to insert into')
    parser.add_argument('--span', metavar='SPAN', type=comma_span,
                        required=True,
                        help='span of the substitution site')
    parser.add_argument('--minor', action='store_true',
                        help='minor fix, leave annotations as they are')
    add_usual_output_args(parser, default_overwrite=True)
    add_commit_args(parser)
    parser.set_defaults(func=main)


def main(args):
    """Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`.
    """
    output_dir = get_output_dir(args, default_overwrite=True)

    # locate insertion site: target document
    reader = educe.stac.Reader(args.corpus)
    tgt_files = reader.filter(reader.files(), is_requested(args))
    tgt_corpus = reader.slurp(tgt_files)

    # TODO mark units with FIXME, optionally delete in/out relations
    span = args.span
    minor = args.minor
    # store before/after
    annos_before = []
    all_del_annos = []
    for tgt_k, tgt_doc in tgt_corpus.items():
        annos_before.append(annotate_doc(tgt_doc, span=span))
        # process
        new_tgt_doc, del_annos = delete_text_at_span(
            tgt_doc, span, minor=minor)
        all_del_annos.append((tgt_k, del_annos))
        # show diff and save doc
        diffs = ["======= DELETE TEXT IN %s   ========" % tgt_k,
                 show_diff(tgt_doc, new_tgt_doc)]
        print("\n".join(diffs).encode('utf-8'), file=sys.stderr)
        save_document(output_dir, tgt_k, new_tgt_doc)
    announce_output_dir(output_dir)
    # commit message
    tgt_k, tgt_doc = list(tgt_corpus.items())[0]
    anno_str_before = annos_before[0]
    if tgt_k and not args.no_commit_msg:
        print("-----8<------")
        print(commit_msg(tgt_k, anno_str_before, all_del_annos))
