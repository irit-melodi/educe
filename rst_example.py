#!/usr/bin/env python

import argparse
import codecs
import os
import sys
import xml.etree.ElementTree as ET

from educe import rst_dt, glozz
from educe.rst_dt import sdrt, graph

# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------

def write_glozz(gdoc, path_stub):
    ac_path   = path_stub + '.ac'
    aa_path   = path_stub + '.aa'

    with codecs.open(ac_path, 'w', 'utf-8') as ac_f:
        print >> ac_f, gdoc.text()

    gdoc.hashcode = glozz.hashcode(ac_path) # why doees hashcode not just accept a byte array?
    glozz.write_annotation_file(aa_path, gdoc)

def render(gr, path_stub):
    dot_g = graph.DotGraph(gr)

    dot_path = path_stub + '.dot'
    png_path = path_stub + '.png'
    with codecs.open(dot_path, 'w', encoding='utf-8') as f:
        print >> f, dot_g.to_string()
    os.system('dot -T png -o %s %s' % (png_path, dot_path))

# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------

arg_parser = argparse.ArgumentParser(description='RST toy/example')
arg_parser.add_argument('input',  metavar='DIR', help='RST directory')
arg_parser.add_argument('output', metavar='DIR', help='output directory')
args = arg_parser.parse_args()

# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------

odir = args.output
if not os.path.exists(odir):
    os.makedirs(odir)

reader     = rst_dt.Reader(args.input)
anno_files = reader.files()
corpus     = reader.slurp_subcorpus(anno_files, True)
gcorpus    = {}

for k in corpus:
    print >> sys.stderr, k

    path_stub = os.path.join(odir, rst_dt.id_to_path(k))

    gcorpus[k] = sdrt.rst_to_glozz_sdrt(corpus[k])
    write_glozz(gcorpus[k], path_stub)

    gr = graph.Graph.from_doc(gcorpus, k)
    render(gr, path_stub)
