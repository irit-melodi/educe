"""
Helpers for building HTML
Hint: import the ET for the ET package too
"""

import xml.etree.cElementTree as ET


def elem(parent, tag, text=None, attrib=None, **kwargs):
    """
    Create an HTML element under the given parent node,
    with some text inside of it
    """
    attrib = attrib or {}
    child = ET.SubElement(parent, tag, {}, **kwargs)
    if text:
        child.text = text
    return child


def span(parent, text=None, attrib=None, **kwargs):
    """
    Create and return an HTML span under the given
    parent node
    """
    return elem(parent, 'span', text, attrib, **kwargs)


# pylint: disable=invalid-name
def br(parent):
    """
    Create and return an HTML br tag under the parent node
    """
    return ET.SubElement(parent, 'br')
# pylint: enable=invalid-name
