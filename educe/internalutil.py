# Author: Eric Kow
# License: BSD3

"""
Utility functions which are meant to be used by educe but aren't expected
to be too useful outside of it
"""

import itertools
import sys


# we've only recently switched to NLTK 3, so there may be
# be bits and pieces of code out there using this function
# it could be worth replacing them
def treenode(tree):
    "API-change padding for NLTK 2 vs NLTK 3 trees"
    return tree.label()


if sys.version > '3':
    ifilter = filter
    ifilterfalse = itertools.filterfalse
    izip = zip

else:
    ifilter = itertools.ifilter
    ifilterfalse = itertools.ifilterfalse
    izip = itertools.izip


class EduceXmlException(Exception):
    def __init__(self, *args, **kw):
        Exception.__init__(self, *args, **kw)


def on_single_element(root, default, f, name):
    """
    Return

       * the default if no elements
       * f(the node) if one element
       * an exception if more than one
    """
    nodes = root.findall(name)
    if len(nodes) == 0:
        if default is None:
            raise EduceXmlException("Expected but did not find any nodes "
                                    "with name %s" % name)
        else:
            return default
    elif len(nodes) > 1:
        raise EduceXmlException("Found more than one node with "
                                "name %s" % name)
    else:
        return f(nodes[0])


def linebreak_xml(elem):
    """
    Insert a break after each element tag

    You probably want `indent_xml` instead
    """
    i = "\n"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            linebreak_xml(elem)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if not elem.tail or not elem.tail.strip():
            elem.tail = i


def indent_xml(elem, level=0):
    """
    From <http://effbot.org/zone/element-lib.htm>

    WARNING: destructive
    """
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
