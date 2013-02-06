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

class Unit:
    def __init__(self, span, type, features):
        self.span=span
        self.type=type
        self.features=features

    def __str__(self):
        feats=", ".join(map(feature_str,self.features))
        return ('%s %s %s' % (self.type, self.span, feats))

def feature_str((a,v)):
    if v is None:
        return a
    else:
        return ('%s:%s' % (a,v))

# TODO: learn how exceptions work in Python; can I embed
# arbitrary strings in them?
#
# TODO: probably replace starting/ending integers with
# exception of some sort
class GlozzException(Exception):
    def __init__(self, *args, **kw):
        Exception.__init__(self, *args, **kw)

def onElementWithName(root, default, f, name):
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
# playing around with an example
# ---------------------------------------------------------------------

# TODO: learn how to use one of these Python XML libs correctly
# surely this painful dom-walking code is not the way to go

def read_singlePosition(node):
    def read_sp(n):
        return int(n.attrib['index'])
    return onElementWithName(node,-3, read_sp, 'singlePosition')

def read_positioning(node):
    start = onElementWithName(node, -2, read_singlePosition, 'start')
    end   = onElementWithName(node, -2, read_singlePosition, 'end')
    return Span(start,end)

def read_type(node):
    return node.text

def read_featureSet(node):
    return map(read_feature, node.findall('feature'))

def read_feature(node):
    attr=node.attrib['name']
    val =node.text
    return(attr, val)

def read_characterisation(node):
    fs        = onElementWithName(node, [],             read_featureSet, 'featureSet')
    unit_type = onElementWithName(node, GlozzException, read_type,       'type')
    return (unit_type, fs)

def read_unit(node):
    (unit_type, fs) = onElementWithName(node, GlozzException, read_characterisation, 'characterisation')
    span            = onElementWithName(node, Span(-1,-1),    read_positioning,      'positioning')
    return Unit(span, unit_type, fs)

def read_glozz(node):
    return map(read_unit, node.findall('unit'))

tree = ET.parse('example.aa')
root = tree.getroot()
units = read_glozz(root)
for u in units:
    print u
