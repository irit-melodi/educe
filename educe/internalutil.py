# Author: Eric Kow
# License: BSD3

"""
Utility functions which are meant to be used by educe but aren't expected
to be too useful outside of it
"""

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
    nodes=root.findall(name)
    if len(nodes) == 0:
        if default is None:
            raise EduceXmlException("Expected but did not find any nodes with name %s" % name)
        else:
            return default
    elif len(nodes) > 1:
        raise EduceXmlException("Found more than one node with name %s" % name)
    else:
        return f(nodes[0])
