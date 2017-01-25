#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: BSD3

"""
Check the corpus for any consistency problems
"""

from __future__ import print_function
from os import path as fp
import argparse
import codecs
import glob
import itertools
import os
import shutil
import subprocess
import sys
import tempfile

from educe import stac
from educe.corpus import FileId
import educe.graph
from educe.stac import graph as egr
from educe.stac.corpus import (METAL_STR, twin_key)
from educe.stac.util.args import STAC_GLOBS
from educe.stac.context import Context
from educe.stac.corenlp import (parsed_file_name)
import educe.util
import educe.stac.sanity.checks.annotation
import educe.stac.sanity.checks.glozz
import educe.stac.sanity.checks.graph
import educe.stac.sanity.checks.type_err

from . import html as h
from .html import ET as ET
from .report import (HtmlReport)


def first_or_none(itrs):
    """
    Return the first element or None if there isn't one
    """
    lst = list(itertools.islice(itrs, 1))
    return lst[0] if lst else None


def create_dirname(path):
    """
    Create the directory beneath a path if it does not exist
    """
    dirname = fp.dirname(path)
    if not fp.exists(dirname):
        os.makedirs(dirname)

# ---------------------------------------------------------------------
# running the checks
# ---------------------------------------------------------------------


def sanity_check_order(k):
    """
    We want to sort file id by order of

    1. doc
    2. subdoc
    3. annotator
    4. stage (unannotated < unit < discourse)

    The important bit here is the idea that we should maybe
    group unit and discourse for 1-3 together
    """
    def stage_num(stg):
        'return a number for a stage name'
        if stg == 'unannotated':
            return 0
        elif stg == 'units':
            return 1
        elif stg == 'discourse':
            return 2
        else:
            return 3

    return (k.doc, (k.subdoc or ''), (k.annotator or ''), stage_num(k.stage))


def run_checks(inputs, k):
    """
    Run sanity checks for a given document
    """
    educe.stac.sanity.checks.glozz.run(inputs, k)
    educe.stac.sanity.checks.annotation.run(inputs, k)
    educe.stac.sanity.checks.type_err.run(inputs, k)
    educe.stac.sanity.checks.graph.run(inputs, k)

# ---------------------------------------------------------------------
# copy stanford parses and stylesheet
# (could be handy for a sort of global dashboard view)
# ---------------------------------------------------------------------


def copy_parses(settings):
    "Copy relevant stanford parser outputs from corpus to report"
    output_dir = settings.output_dir

    docs = set(k.doc for k in settings.corpus)
    for doc in docs:
        subdocs = set(k.subdoc for k in settings.corpus if k.doc == doc)
        if subdocs:
            k = FileId(doc=doc,
                       subdoc=list(subdocs)[0],
                       stage=None,
                       annotator=None)
            i_style_dir = fp.dirname(parsed_file_name(k, settings.corpus_dir))
            o_style_dir = fp.dirname(parsed_file_name(k, output_dir))
            i_style_file = fp.join(i_style_dir, 'CoreNLP-to-HTML.xsl')
            if fp.exists(i_style_file):
                if not fp.exists(o_style_dir):
                    os.makedirs(o_style_dir)
                shutil.copy(i_style_file, o_style_dir)
        for subdoc in subdocs:
            k = FileId(doc=doc, subdoc=subdoc, stage=None, annotator=None)
            i_file = parsed_file_name(k, settings.corpus_dir)
            o_file = parsed_file_name(k, output_dir)
            o_dir = fp.dirname(o_file)
            if fp.exists(i_file):
                if not fp.exists(o_dir):
                    os.makedirs(o_dir)
                shutil.copy(i_file, o_dir)

# ---------------------------------------------------------------------
# generate graphs
# ---------------------------------------------------------------------


def generate_graphs(settings):
    """
    Draw SVG graphs for each of the documents in the corpus
    """
    discourse_only = [k for k in settings.corpus if k.stage == 'discourse']
    report = settings.report

    # generate dot files
    for k in discourse_only:
        try:
            gra = egr.DotGraph(egr.Graph.from_doc(settings.corpus, k))
            dot_file = report.subreport_path(k, '.dot')
            create_dirname(dot_file)
            if gra.get_nodes():
                with codecs.open(dot_file, 'w', encoding='utf-8') as fout:
                    print(gra.to_string(), file=fout)
        except educe.graph.DuplicateIdException:
            warning = ("Couldn't graph %s because it has duplicate "
                       "annotation ids") % dot_file
            print(warning, file=sys.stderr)

    # attempt to graphviz them
    try:
        print("Generating graphs... (you can safely ^-C here)",
              file=sys.stderr)
        for k in discourse_only:
            dot_file = report.subreport_path(k, '.dot')
            svg_file = report.subreport_path(k, '.svg')
            if fp.exists(dot_file) and settings.draw:
                subprocess.call('dot -T svg -o %s %s' % (svg_file, dot_file),
                                shell=True)
    except OSError as oops:
        print("Couldn't run graphviz. (%s)" % oops, file=sys.stderr)
        print("You should install it for easier sanity check debugging.",
              file=sys.stderr)

# ---------------------------------------------------------------------
# index
# ---------------------------------------------------------------------


def add_element(settings, k, html, descr, mk_path):
    """
    Add a link to a report element for a given document,
    but only if it actually exists
    """
    abs_p = mk_path(k, settings.output_dir)
    rel_p = mk_path(k, '.')
    if fp.exists(abs_p):
        h.span(html, text=' | ')
        h.elem(html, 'a', text=descr, href=rel_p)


def issues_descr(report, k):
    """
    Return a string characterising a report as either being
    warnings or error (helps the user scan the index to
    figure out what needs clicking on)
    """
    contents = "issues" if report.has_errors(k) else "warnings"
    return "%s %s" % (k.stage, contents)


def _apppend_subdoc_entry(settings, hlist, key):
    """
    Append a bullet point for a given subdocument, pointing to
    any reports and helper elements we may have generated
    """
    report = settings.report
    mk_report_path =\
        lambda k, odir: report.mk_output_path(odir, k, '.report.html')
    mk_svg_path =\
        lambda k, odir: report.mk_output_path(odir, k, '.svg')
    k_review = twin_key(key, 'review')
    k_units = twin_key(key, 'units')
    k_discourse = twin_key(key, 'discourse')
    h_sub_li = h.elem(hlist, 'li', text=' (' + key.subdoc + ')')
    add_element(settings, k_units, h_sub_li,
                issues_descr(report, k_units),
                mk_report_path)
    add_element(settings, k_discourse, h_sub_li,
                issues_descr(report, k_discourse),
                mk_report_path)
    add_element(settings, k_discourse, h_sub_li, 'graph',
                mk_svg_path)
    add_element(settings, k_review, h_sub_li, 'parses',
                parsed_file_name)


def write_index(settings):
    """
    Write the report index
    """
    corpus = settings.corpus
    htree = ET.Element('html')

    h.elem(htree, 'h2', text='general')
    h.elem(htree, 'div',
           text="NB: Try Firefox if Google Chrome won't open the parses")

    annotators = set(k.annotator for k in corpus if k.annotator)
    for anno in sorted(annotators):
        h_anno_hdr = ET.SubElement(htree, 'h2')
        h_anno_hdr.text = anno
        h_list = ET.SubElement(htree, 'ul')
        anno_keys = set(k for k in corpus if k.annotator == anno)
        for doc in sorted(set(k.doc for k in anno_keys)):
            h_li = h.elem(h_list, 'li', text=doc)
            h_sublist = h.elem(h_li, 'ul')
            for subdoc in sorted(set(k.subdoc for k in anno_keys
                                     if k.doc == doc)):
                key = FileId(doc=doc,
                             subdoc=subdoc,
                             annotator=anno,
                             stage=None)
                _apppend_subdoc_entry(settings, h_sublist, key)

    with open(fp.join(settings.output_dir, 'index.html'), 'w') as fout:
        print(ET.tostring(htree), file=fout)

# ---------------------------------------------------------------------
# put it all together
# ---------------------------------------------------------------------


# pylint: disable=too-many-instance-attributes
class SanityChecker(object):
    """
    Sanity checker settings and state
    """
    def __init__(self, args):
        is_interesting = educe.util.mk_is_interesting(args)
        self.corpus_dir = args.corpus
        self.corpus = None
        self.contexts = None
        self.__init_read_corpus(is_interesting, self.corpus_dir)
        self.__init_set_output(args.output)
        self.report = HtmlReport(self.anno_files, self.output_dir)
        self.draw = args.draw

    def __init_read_corpus(self, is_interesting, corpus_dir):
        """
        Read the corpus specified in our args
        """
        reader = stac.Reader(corpus_dir)
        all_files = reader.files()
        self.anno_files = reader.filter(all_files, is_interesting)
        interesting = list(self.anno_files)  # or list(self.anno_files.keys())
        for key in interesting:
            ukey = twin_key(key, 'unannotated')
            if ukey in all_files:
                self.anno_files[ukey] = all_files[ukey]
        self.corpus = reader.slurp(self.anno_files, verbose=True)
        self.contexts = {k: Context.for_edus(self.corpus[k])
                         for k in self.corpus}

    def __init_set_output(self, output):
        """
        Create our output directory as necessary
        """
        if output:
            if fp.isfile(output):
                sys.exit("Sorry, %s already exists and is not a directory" %
                         output)
            elif not fp.isdir(output):
                os.makedirs(output)
            self.output_dir = output
            self._output_to_temp = False
        else:
            self.output_dir = tempfile.mkdtemp()
            self._output_to_temp = True

    def output_is_temp(self):
        """
        True if we are writing to an output directory
        """
        return self._output_to_temp

    def run(self):
        """
        Perform sanity checks and write the output
        """
        for k in sorted(self.corpus, key=sanity_check_order):
            run_checks(self, k)
            create_dirname(self.report.subreport_path(k))
            self.report.flush_subreport(k)

        copy_parses(self)
        generate_graphs(self)
        write_index(self)

        output_dir = self.output_dir
        if self.output_is_temp():
            print("See temp directory: %s" % output_dir, file=sys.stderr)
            print("HINT: use --output if you want to specify "
                  "an output directory",
                  file=sys.stderr)
        else:
            print("Fancy results saved in %s" % output_dir, file=sys.stderr)
# pylint: enable=too-many-instance-attributes

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def easy_settings(args):
    """Modify args to reflect user-friendly defaults.

    Terminates the program if `args.corpus` is set but does not point to
    an existing folder ; otherwise `args.doc` must be set and everything
    else is expected to be empty.

    Parameters
    ----------
    args : Namespace
        Arguments of the argparser.

    See also
    --------
    `educe.stac.util.args.check_easy_settings()`
    """
    if args.corpus:
        # 2017-01-25 explicitly break if there is no such folder
        if not os.path.exists(args.corpus):
            raise ValueError("No corpus directory {corpus}\n{hint}".format(
                corpus=args.corpus))
        # end explicitly break if no such folder
        return  # not easy mode

    if not args.doc:
        raise ValueError("no document specified for easy mode\n{hint}")

    # figure out where this thing lives
    for sdir in STAC_GLOBS:
        if glob.glob(fp.join(sdir, args.doc)):
            args.corpus = sdir
    if not args.corpus:
        if not any(fp.isdir(x) for x in STAC_GLOBS):
            raise ValueError("You don't appear to be in the STAC root dir")
        else:
            raise ValueError("I don't know about any document called "
                             + args.doc)
    # prepare info message for user
    guess_report = "{corpus} --doc \"{doc}\""

    # annotator
    args.annotator = METAL_STR
    guess_report += ' --annotator "{}"'.format(args.annotator)

    # output
    if not args.output:
        # 2017-01-25 don't override output folder if it is given
        args.output = fp.join("/tmp", "sanity-" + args.doc)
    guess_report += ' --output "{}"'.format(args.output)

    print("Guessing convenience settings:", file=sys.stderr)
    print("stac-check " + guess_report.format(**args.__dict__),
          file=sys.stderr)


EASY_SETTINGS_HINT = "Try this: stac-check --doc pilot03"


def main():
    """
    Sanity checker CLI entry point
    """
    arg_parser = argparse.ArgumentParser(description='Check corpus for '
                                         'potential problems.')
    arg_parser.add_argument('corpus', metavar='DIR', nargs='?')
    arg_parser.add_argument('--output', '-o', metavar='DIR')
    arg_parser.add_argument('--verbose', '-v',
                            action='count',
                            default=0)
    arg_parser.add_argument('--no-draw', action='store_true',
                            dest='draw', default=True,
                            help='Do not draw relations graph')
    educe.util.add_corpus_filters(arg_parser)
    args = arg_parser.parse_args()
    try:
        easy_settings(args)
    except ValueError as e:
        err_msg = e.args[0]
        sys.exit(err_msg.format(hint=EASY_SETTINGS_HINT))
    checker = SanityChecker(args)
    checker.run()
