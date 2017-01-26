#!/usr/bin/python
# -*- coding: utf-8 -*-

'''Function to "prettify" XML: courtesy of http://www.doughellmann.com/PyMOTW/xml/etree/ElementTree/create.html
'''

from __future__ import print_function
from xml.etree import ElementTree
from xml.dom import minidom
import sys


def prettify(elem, indent=""):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string.replace('\n', ''))
    return reparsed.toprettyxml(indent=indent)

if __name__ == '__main__':
    TREE = ElementTree.parse(sys.argv[1])
    ROOT = TREE.getroot()
    print(prettify(ROOT))
