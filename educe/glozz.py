# Author: Eric Kow
# License: BSD3

"""
The Glozz_ file format in `educe.annotation` form

You're likely most interested in
`slurp_corpus` and `read_annotation_file`

.. _Glozz: http://www.glozz.org/
"""

import codecs
import xml.etree.ElementTree as ET
import sys

from educe.annotation import *

class GlozzDocument(Document):
    def __init__(self, hashcode, unit, rels, schemas, text):
        Document.__init__(self, unit, rels, schemas, text)
        self.hashcode = hashcode

    def to_xml(self):
        elm = ET.Element('annotations')
        if self.hashcode is not None:
            elm.append(ET.Element('metadata', corpusHashcode=self.hashcode))
        elm.extend([glozz_annotation_to_xml(x, 'unit')     for x in self.units])
        elm.extend([glozz_annotation_to_xml(x, 'relation') for x in self.relations])
        elm.extend([glozz_annotation_to_xml(x, 'schema')   for x in self.schemas])
        return elm

    def set_origin(self, origin):
        Document.set_origin(self, origin)
        for x in self.schemas:
            x.origin = origin

def ordered_keys(preferred, d):
    """
    Keys from a dictionary starting with 'preferred' ones
    in the order of preference
    """
    return [ k for k in preferred if k in d ] +\
           [ k for k in d if k not in preferred ]

STAC_GLOZZ_FS_ORDER =\
    [ 'Status'
    , 'Quantity'
    , 'Correctness'
    , 'Kind'
    , 'Comments'
    , 'Developments'
    , 'Emitter'
    , 'Identifier'
    , 'Timestamp'
    , 'Resources'
    , 'Trades'
    , 'Dice_rolling'
    , 'Gets'
    , 'Has_resources'
    , 'Amount_of_resources'
    , 'Addressee'
    , 'Surface_act'
    ]
STAC_UNANNOTATED_FS_ORDER =\
    [ 'Status'
    , 'Quantity'
    , 'Correctness'
    , 'Kind'
    , 'Identifier'
    , 'Timestamp'
    , 'Emitter'
    , 'Resources'
    , 'Developments'
    , 'Comments'
    , 'Dice_rolling'
    , 'Gets'
    , 'Trades'
    , 'Has_resources'
    , 'Amount_of_resources'
    , 'Addressee'
    , 'Surface_act'
    ]

def glozz_annotation_to_xml(self, tag='annotation'):
    meta_elm = ET.Element('metadata')

    # FIXME: while these preferences are harmless
    # (the elements listed here are completely optional and
    # we except the document to mean the same regardless of
    # the order of these elements), they're a bit silly!
    #
    # We have them here to help
    # debugging (via visual diff), but we should either remove
    # them later or think of a way to model task-specific
    # things (here, STAC)
    #
    # Right now, there's not a very clear distinction between
    # STAC specific stuff and glozz generalities
    preferred_md_order = [ 'author'
                         , 'creation-date'
                         , 'lastModifier'
                         , 'lastModificationDate'
                         ]
    preferred_fs_order = STAC_GLOZZ_FS_ORDER
    for k in ordered_keys(preferred_md_order, self.metadata):
        e = ET.Element(k)
        e.text = self.metadata[k]
        meta_elm.append(e)

    char_elm = ET.Element('characterisation')
    char_tag_elm = ET.Element('type')
    char_tag_elm.text = self.type
    char_tag_fs  = ET.Element('featureSet')
    for k in ordered_keys(preferred_fs_order, self.features):
        e = ET.Element('feature',name=k)
        e.text = self.features[k]
        char_tag_fs.append(e)
    char_elm.extend([char_tag_elm, char_tag_fs])

    elm = ET.Element(tag, id=self.local_id())
    if (tag == 'unit'):
        span_elm = glozz_unit_to_span_xml(self)
    elif (tag == 'relation'):
        span_elm = glozz_relation_to_span_xml(self)
    elif (tag == 'schema'):
        span_elm = glozz_schema_to_span_xml(self)
    else:
        raise Exception("Don't know how to emit XML for non unit/relation annotations (%s)" % tag)
    elm.extend([meta_elm, char_elm, span_elm])
    return elm

def glozz_unit_to_span_xml(self):
    def set_pos(elm,x):
        elm.append(ET.Element('singlePosition', index=str(x)))
    elm = ET.Element('positioning')
    start_elm = ET.Element('start')
    end_elm   = ET.Element('end')
    set_pos(start_elm, self.span.char_start)
    set_pos(end_elm,   self.span.char_end)
    elm.extend([start_elm, end_elm])
    return elm

def glozz_relation_to_span_xml(self):
    def set_pos(elm,x):
        elm.append(ET.Element('term', id=str(x)))
    elm = ET.Element('positioning')
    set_pos(elm,self.span.t1)
    set_pos(elm,self.span.t2)
    return elm

def glozz_schema_to_span_xml(self):
    elm = ET.Element('positioning')
    for x in sorted(self.units):
        elm.append(ET.Element('embedded-unit', id=str(x)))
    for x in sorted(self.relations):
        elm.append(ET.Element('embedded-relation', id=str(x)))
    for x in sorted(self.schemas):
        elm.append(ET.Element('embedded-schema', id=str(x)))
    return elm

# ---------------------------------------------------------------------
# xml processing
# -----------------------------------------------------------

class GlozzException(Exception):
    def __init__(self, *args, **kw):
        Exception.__init__(self, *args, **kw)

def on_single_element(root, default, f, name):
    """
    Return

       * the default if no elements
       * f(the node) if one element
       * an exception if more than one
    """
    nodes=root.findall(name)
    if len(nodes) == 0:
        if default is None:
            raise GlozzException("Expected but did not find any nodes with name %s" % name)
        else:
            return default
    elif len(nodes) > 1:
        raise GlozzException("Found more than one node with name %s" % name)
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
        hashcode = get_one('metadata', '', 'annotations')
        if hashcode is '':
            hashcode = None
        units    = get_all('unit')
        rels     = get_all('relation')
        schemas  = get_all('schema')
        return (hashcode, units, rels, schemas)

    elif node.tag == 'characterisation':
        fs        = get_one('featureSet', {})
        unit_type = get_one('type'      , None)
        return (unit_type, fs)

    elif node.tag == 'feature':
        attr=node.attrib['name']
        val =node.text.strip() if node.text else None
        return (attr, val)

    ## TODO throw exception if we see more than one instance of a key
    elif node.tag == 'featureSet':
        return dict(get_all('feature'))

    elif node.tag == 'metadata' and context == 'annotations':
        return node.attrib['corpusHashcode']

    elif node.tag == 'metadata':
        return dict([(t.tag,t.text.strip()) for t in node ])

    elif node.tag == 'positioning' and context == 'unit':
        start = get_one('start', None)
        end   = get_one('end',   None)
        return Span(start,end)

    elif node.tag == 'positioning' and context == 'relation':
        terms = get_all('term')
        if len(terms) != 2:
            raise GlozzException("Was expecting exactly 2 terms, but got %d" % len(terms))
        else:
            return RelSpan(terms[0], terms[1])

    elif node.tag == 'positioning' and context == 'schema':
        units     = frozenset(get_all('embedded-unit'))
        relations = frozenset(get_all('embedded-relation'))
        schemas   = frozenset(get_all('embedded-schema'))
        return units, relations, schemas

    elif node.tag == 'relation':
        rel_id          = node.attrib['id']
        (unit_type, fs) = get_one('characterisation', None)
        span            = get_one('positioning',      None, 'relation')
        metadata        = get_one('metadata',         {})
        return Relation(rel_id, span, unit_type, fs, metadata=metadata)

    if node.tag == 'schema':
        anno_id         = node.attrib['id']
        (anno_type, fs) = get_one('characterisation', None)
        units, rels, schemas = get_one('positioning',      None, 'schema')
        metadata        = get_one('metadata',         {})
        return Schema(anno_id, units, rels, schemas, anno_type, fs, metadata=metadata)

    elif node.tag == 'singlePosition':
        return int(node.attrib['index'])

    elif node.tag == 'start' or node.tag == 'end':
        return get_one('singlePosition', None)

    elif node.tag in [ 'term', 'embedded-unit', 'embedded-relation', 'embedded-schema' ]:
        return node.attrib['id']

    elif node.tag == 'type':
        return node.text.strip()

    elif node.tag == 'unit':
        unit_id         = node.attrib['id']
        (unit_type, fs) = get_one('characterisation', None)
        span            = get_one('positioning',      None, 'unit')
        metadata        = get_one('metadata',         {})
        return Unit(unit_id, span, unit_type, fs, metadata=metadata)


def read_annotation_file(anno_filename, text_filename=None):
    """
    Read a single glozz annotation file and its corresponding text
    (if any).
    """
    tree = ET.parse(anno_filename)
    (hashcode, units, rels, schemas) = read_node(tree.getroot())
    text = None
    if text_filename is not None:
        with codecs.open(text_filename, 'r', 'utf-8') as tf:
            text = tf.read()
    return GlozzDocument(hashcode, units, rels, schemas, text)

def hashcode(ac_filename):
    """
    Hashcode mechanism as documented in the Glozz manual appendix
    """
    code   = 1L
    length = 0
    with open(ac_filename, 'rb') as f:
        byte = f.read(1)
        while byte:
            length += 1
            if byte and byte != 0:
                code *= ord(byte)
                code =  code % 99999999L
            byte = f.read(1)
        return str(length) + '-' + str(code)

def write_annotation_file(anno_filename, doc):
    """
    Write a GlozzDocument to XML in the given path
    """

    # tweaked from the indent function in
    # http://effbot.org/zone/element-lib.htm
    def reformat(elem):
        """
        insert a break after each element tag
        """
        i = "\n"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                reformat(elem)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if not elem.tail or not elem.tail.strip():
                elem.tail = i

    elem = doc.to_xml()
    reformat(elem) # ugh, imperative
    ET.ElementTree(elem).write(anno_filename, encoding='utf-8', xml_declaration=True)
