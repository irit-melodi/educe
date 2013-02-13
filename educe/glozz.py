# Author: Eric Kow
# License: BSD3

import xml.etree.ElementTree as ET

# Glozz annotations
# The aim here is to be fairly low-level and just model how glozz
# thinks about things
# If you want to do any more sophisticated analysis, you'll probably
# want to build a more abstract layer on top of it, eg. using the
# python graph library

class Span:
    def __init__(self, start, end):
        self.char_start=start
        self.char_end=end

    def  __str__(self):
        return ('(%d,%d)' % (self.char_start, self.char_end))

class RelSpan(Span):
    def __init__(self, t1, t2):
        self.t1=t1
        self.t2=t2

    def  __str__(self):
        return ('%s -> %s' % (self.t1, self.t2))

class Unit:
    def __init__(self, unit_id, span, type, features):
        self.unit_id=unit_id
        self.span=span
        self.type=type
        self.features=features

    def __str__(self):
        feats=", ".join(map(feature_str,self.features))
        return ('%s [%s] %s %s' % (self.unit_id,self.type, self.span, feats))

class Relation(Unit):
    def __init__(self, rel_id, span, type, features):
        Unit.__init__(self, rel_id, span, type, features)

class Feature():
    """Attribute-value pair"""
    def __init__(self, attribute, value):
        self.attribute=attribute
        self.value=value

    def __str__(self):
        if self.value is None:
            return self.attribute
        else:
            return ('%s:%s' % (self.attribute,self.value))

class Annotations:
    def __init__(self, units, rels):
        self.units=units
        self.rels=rels


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
        return node.text

    elif node.tag == 'unit':
        unit_id         = node.attrib['id']
        (unit_type, fs) = get_one('characterisation', GlozzException)
        span            = get_one('positioning',      Span(-1,-1), 'unit')
        return Unit(unit_id, span, unit_type, fs)

def read_annotation_file(filename):
    tree = ET.parse(filename)
    res  = read_node(tree.getroot())
    return Annotations(res[0],res[1])
