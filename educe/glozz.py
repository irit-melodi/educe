# Author: Eric Kow
# License: BSD3

import xml.etree.ElementTree as ET
import sys

from educe.annotation import *

# Glozz annotations
# The aim here is to be fairly low-level and just capture how glozz
# thinks about things.

# ---------------------------------------------------------------------
# xml processing
# -----------------------------------------------------------

# TODO: learn how exceptions work in Python; can I embed
# arbitrary strings in them?
#
# TODO: probably replace starting/ending integers with
# exception of some sort
class GlozzException(Exception):
    def __init__(self, *args, **kw):
        Exception.__init__(self, *args, **kw)

def on_single_element(root, default, f, name):
    """Return
       * the default if no elements
       * f(the node) if one element
       * an exception if more than one
    """
    nodes=root.findall(name)
    if len(nodes) == 0:
        return default
    elif len(nodes) > 1:
        raise GlozzException()
    else:
        return f(nodes[0])

# ---------------------------------------------------------------------
# glozz files
# ---------------------------------------------------------------------

def read_node(node, context=None):
    def get_one(name, default, ctx=None):
        f = lambda n : read_node(n, ctx)
        return on_single_element(node, default, f, name)

    def get_all(name):
        return map(read_node, node.findall(name))

    if node.tag == 'annotations':
        units = get_all('unit')
        rels  = get_all('relation')
        return (units, rels)

    elif node.tag == 'characterisation':
        fs        = get_one('featureSet', [])
        unit_type = get_one('type'      , GlozzException)
        return (unit_type, fs)

    elif node.tag == 'feature':
        attr=node.attrib['name']
        val =node.text
        return Feature(attr, val)

    elif node.tag == 'featureSet':
        return get_all('feature')

    elif node.tag == 'positioning' and context == 'unit':
        start = get_one('start', -2)
        end   = get_one('end',   -2)
        return Span(start,end)

    elif node.tag == 'positioning' and context == 'relation':
        terms = get_all('term')
        if len(terms) != 2:
            raise GlozzException()
        else:
            return RelSpan(terms[0], terms[1])

    elif node.tag == 'relation':
        rel_id          = node.attrib['id']
        (unit_type, fs) = get_one('characterisation', GlozzException)
        span            = get_one('positioning',      RelSpan(-1,-1), 'relation')
        return Relation(rel_id, span, unit_type, fs)

    elif node.tag == 'singlePosition':
        return int(node.attrib['index'])

    elif node.tag == 'start' or node.tag == 'end':
        return get_one('singlePosition', -3)

    elif node.tag == 'term':
        return node.attrib['id']

    elif node.tag == 'type':
        return node.text.strip()

    elif node.tag == 'unit':
        unit_id         = node.attrib['id']
        (unit_type, fs) = get_one('characterisation', GlozzException)
        span            = get_one('positioning',      Span(-1,-1), 'unit')
        return Unit(unit_id, span, unit_type, fs)

def read_annotation_file(filename):
    tree = ET.parse(filename)
    res  = read_node(tree.getroot())
    return Document(res[0],res[1])

def slurp_corpus(cfiles, verbose=False):
    """
    Given a dictionary that maps keys to filepaths, return a dictionary
    mapping keys to the annotations within that file
    """
    corpus={}
    counter=0
    for k in cfiles.keys():
        if verbose:
            sys.stderr.write("\rSlurping corpus dir [%d/%d]" % (counter, len(cfiles)))
        annotations=read_annotation_file(cfiles[k])
        for u in annotations.units:
            u.origin=k
        corpus[k]=annotations
        counter=counter+1
    if verbose:
        sys.stderr.write("\rSlurping corpus dir [%d/%d done]\n" % (counter, len(cfiles)))
    return corpus
