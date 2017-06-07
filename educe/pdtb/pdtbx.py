#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author Eric Kow
# LICENSE: BSD3 (2013, Université Paul Sabatier)

"""
PDTB in an adhoc (educe-grown) XML format, unfortunately not a standard, but a
little homegrown language using XML syntax.  I'll call it pdtbx. No reason it
can't be used outside of educe.

Informal DTD:

    * SpanList        is attribute spanList in PDTB string convention
    * GornAddressList is attribute gornList in PDTB string convention
    * SemClass        is attribute semclass1 (and optional attribute semclass2)
                         in PDTB string convention
    * text in `<text>` elements with usual XML escaping conventions
    * args in `<arg>`  elements in order (arg1 before arg2)
    * implicitRelations can have multiple connectives
"""

import xml.etree.cElementTree as ET  # python 2.5 and later

import educe.pdtb.parse as ty
from educe.internalutil import (on_single_element, EduceXmlException,
                                indent_xml)


# ---------------------------------------------------------------------
# XML to internal structure
# ---------------------------------------------------------------------
def _read_GornAddressList(attr):
    return [ty.GornAddress([int(y) for y in x.split(',')])
            for x in attr.split(';')]


def _read_SpanList(attr):
    return [tuple([int(y) for y in x.split('..')]) for x in attr.split(';')]


def _read_SemClass(attr):
    return ty.SemClass(attr.split('.'))


def _read_Selection(node):
    attr = node.attrib
    return ty.Selection(span=_read_SpanList(attr['spanList']),
                        gorn=_read_GornAddressList(attr['gornList']),
                        text=on_single_element(
                            node, None, lambda x: x.text, 'text'))


def _read_InferenceSite(node):
    attr = node.attrib
    return ty.InferenceSite(strpos=int(attr['strpos']),
                            sentnum=int(attr['sentnum']))


def _read_Connective(node):
    attr = node.attrib
    semclass1_ = attr['semclass1']
    semclass2_ = attr.get('semclass2', None)  # optional
    semclass1 = _read_SemClass(semclass1_)
    semclass2 = _read_SemClass(semclass2_) if semclass2_ else None
    return ty.Connective(text=attr['text'],
                         semclass1=semclass1,
                         semclass2=semclass2)


def _read_Attribution(node):
    attr = node.attrib
    selection = on_single_element(node, (), _read_Selection, 'selection')
    return ty.Attribution(polarity=attr['polarity'],
                          determinacy=attr['determinacy'],
                          type=attr['type'],
                          source=attr['source'],
                          selection=(None if selection is () else selection))


def _read_Sup(node):
    return ty.Sup(_read_Selection(node))


def _read_Arg(node):
    sup = on_single_element(node, (), _read_Sup, 'sup')
    attribution = on_single_element(
        node, (), _read_Attribution, 'attribution')
    return ty.Arg(selection=_read_Selection(node),
                  attribution=(None if attribution is () else attribution),
                  sup=(None if sup is () else sup))


def _read_Args(node):
    args = node.findall('arg')
    if len(args) != 2:
        raise EduceXmlException('Was expecting exactly two arguments '
                                '(got %d)' % len(args))
    return tuple([_read_Arg(x) for x in args])


def _read_ExplicitRelationFeatures(node):
    attribution = on_single_element(
        node, None, _read_Attribution, 'attribution')
    connhead = on_single_element(
        node, None, _read_Connective, 'connhead')
    return ty.ExplicitRelationFeatures(attribution=attribution,
                                       connhead=connhead)


def _read_ExplicitRelation(node):
    return ty.ExplicitRelation(selection=_read_Selection(node),
                               features=_read_ExplicitRelationFeatures(node),
                               args=_read_Args(node))


def _read_ImplicitRelationFeatures(node):
    connectives = node.findall('connective')
    if len(connectives) == 0:
        raise EduceXmlException('Was expecting at least one connective '
                                '(got none)')
    elif len(connectives) > 2:
        raise EduceXmlException('Was expecting no more than two connectives '
                                '(got %d)' % len(connectives))

    attribution = on_single_element(
        node, None, _read_Attribution, 'attribution')
    connective1 = _read_Connective(connectives[0])
    connective2 = (_read_Connective(connectives[1]) if len(connectives) == 2
                   else None)
    return ty.ImplicitRelationFeatures(attribution=attribution,
                                       connective1=connective1,
                                       connective2=connective2)


def _read_ImplicitRelation(node):
    return ty.ImplicitRelation(infsite=_read_InferenceSite(node),
                               features=_read_ImplicitRelationFeatures(node),
                               args=_read_Args(node))


def _read_AltLexRelationFeatures(node):
    attribution = on_single_element(
        node, None, _read_Attribution, 'attribution')
    attr = node.attrib
    semclass1_ = attr['semclass1']
    semclass2_ = attr.get('semclass2', None)  # optional
    semclass1 = _read_SemClass(semclass1_)
    semclass2 = _read_SemClass(semclass2_) if semclass2_ else None
    return ty.AltLexRelationFeatures(attribution=attribution,
                                     semclass1=semclass1,
                                     semclass2=semclass2)


def _read_AltLexRelation(node):
    return ty.AltLexRelation(selection=_read_Selection(node),
                             features=_read_AltLexRelationFeatures(node),
                             args=_read_Args(node))


def _read_EntityRelation(node):
    return ty.EntityRelation(infsite=_read_InferenceSite(node),
                             args=_read_Args(node))


def _read_NoRelation(node):
    return ty.NoRelation(infsite=_read_InferenceSite(node),
                         args=_read_Args(node))


def read_Relation(node):
    tag = node.tag
    if tag == 'explicitRelation':
        return _read_ExplicitRelation(node)
    elif tag == 'implicitRelation':
        return _read_ImplicitRelation(node)
    elif tag == 'altLexRelation':
        return _read_AltLexRelation(node)
    elif tag == 'entityRelation':
        return _read_EntityRelation(node)
    elif tag == 'noRelation':
        return _read_NoRelation(node)
    else:
        raise EduceXmlException("Don't know how to read relation with "
                                "name %s" % tag)


def read_Relations(node):
    return [read_Relation(x) for x in node]


def read_pdtbx_file(filename):
    tree = ET.parse(filename)
    return read_Relations(tree.getroot())


# ---------------------------------------------------------------------
# internal structure to XML
# ---------------------------------------------------------------------
def _Selection_xml(itm, name='selection'):
    elm = ET.Element(name)
    txt = ET.SubElement(elm, 'text')
    txt.text = itm.text
    elm.attrib = {
        'gornList': _GornAddressList_xml(itm.gorn),
        'spanList': _SpanList_xml(itm.span)
    }
    return elm


def _InferenceSite_xml(itm, name='inferenceSite'):
    elm = ET.Element(name)
    elm.attrib = {
        'strpos': str(itm.strpos),
        'sentnum': str(itm.sentnum)
    }
    return elm


def _GornAddressList_xml(itm):
    return ";".join([_GornAddress_xml(x) for x in itm])


def _SpanList_xml(itm):
    return ";".join([_Span_xml(x) for x in itm])


def _GornAddress_xml(itm):
    return ",".join([str(x) for x in itm.parts])


def _Span_xml(itm):
    return "%d..%d" % itm


def _Attribution_xml(itm):
    elm = ET.Element('attribution')
    elm.attrib = {
        'polarity': itm.polarity,
        'determinacy': itm.determinacy,
        'source': itm.source,
        'type': itm.type
    }
    if itm.selection:
        elm.append(_Selection_xml(itm.selection))
    return elm


def _SemClass_xml(itm):
    return ".".join(itm.klass)


def _Connective_xml(itm, name='connective'):
    elm = ET.Element(name)
    elm.attrib['semclass1'] = _SemClass_xml(itm.semclass1)
    if itm.semclass2:
        elm.attrib['semclass2'] = _SemClass_xml(itm.semclass2)
    elm.attrib['text'] = itm.text
    return elm


def _Sup_xml(itm):
    return _Selection_xml(itm, 'sup')


def _Arg_xml(itm):
    elm = _Selection_xml(itm, 'arg')
    if itm.attribution:
        elm.append(_Attribution_xml(itm.attribution))
    if itm.sup:
        elm.append(_Sup_xml(itm.sup))
    return elm


def _RelationArgsXml(itm):
    return [_Arg_xml(itm.arg1),
            _Arg_xml(itm.arg2)]


def _ExplicitRelation_xml(itm):
    elm = _Selection_xml(itm, 'explicitRelation')
    elm.append(_Attribution_xml(itm.attribution))
    elm.append(_Connective_xml(itm.connhead, name='connhead'))
    elm.extend(_RelationArgsXml(itm))
    return elm


def _ImplicitRelation_xml(itm):
    elm = _InferenceSite_xml(itm, 'implicitRelation')
    elm.append(_Attribution_xml(itm.attribution))
    elm.append(_Connective_xml(itm.connective1))
    if itm.connective2:
        elm.append(_Connective_xml(itm.connective2))
    elm.extend(_RelationArgsXml(itm))
    return elm


def _AltLexRelation_xml(itm):
    elm = _Selection_xml(itm, 'altLexRelation')
    elm.attrib['semclass1'] = _SemClass_xml(itm.semclass1)
    if itm.semclass2:
        elm.attrib['semclass2'] = _SemClass_xml(itm.semclass2)
    elm.append(_Attribution_xml(itm.attribution))
    elm.extend(_RelationArgsXml(itm))
    return elm


def _EntityRelation_xml(itm):
    elm = _InferenceSite_xml(itm, 'entityRelation')
    elm.extend(_RelationArgsXml(itm))
    return elm


def _NoRelation_xml(itm):
    elm = _InferenceSite_xml(itm, 'noRelation')
    elm.extend(_RelationArgsXml(itm))
    return elm


def Relation_xml(itm):
    if isinstance(itm, ty.ExplicitRelation):
        return _ExplicitRelation_xml(itm)
    elif isinstance(itm, ty.ImplicitRelation):
        return _ImplicitRelation_xml(itm)
    elif isinstance(itm, ty.AltLexRelation):
        return _AltLexRelation_xml(itm)
    elif isinstance(itm, ty.EntityRelation):
        return _EntityRelation_xml(itm)
    elif isinstance(itm, ty.NoRelation):
        return _NoRelation_xml(itm)
    else:
        raise Exception("Don't know how to translate relation of "
                        "type %s" % type(itm))


def Relations_xml(itms):
    elm = ET.Element('relations')
    elm.extend([Relation_xml(x) for x in itms])
    return elm


def write_pdtbx_file(filename, relations):
    xml = Relations_xml(relations)
    indent_xml(xml)
    ET.ElementTree(xml).write(filename, encoding='utf-8',
                              xml_declaration=True)
