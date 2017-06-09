"""This module provides an API for fragmented EDUs."""

from __future__ import absolute_import, print_function


def edu_num(edu_id):
    """Get the index (number) of an EDU from its identifier.

    A variant of this probably exists elsewhere in the code base, but I
    can't seem to find it as of 2017-02-01.

    Parameters
    ----------
    edu_id : str
        Identifier of the EDU.

    Returns
    -------
    edu_num : int
        Position index of this EDU in the document.
    """
    if edu_id == 'ROOT':
        return 0
    return int(edu_id.rsplit('_', 1)[1])


def edu_members(du):
    """Get a tuple with the num of the EDUs members of a DU.

    Parameters
    ----------
    du : EDU or :obj:`tuple` of str
        Discourse Unit, either an EDU or a non-elementary DU described
        by the tuple of the identifiers of its EDU members.

    Returns
    -------
    mem_nums : :obj:`tuple` of int
        Numbers of the EDU members of this DU.
    """
    if isinstance(du, tuple):  # frag EDU, CDU
        # get the EDUs from their identifiers
        return tuple(edu_num(x) for x in du[1])
    else:
        return tuple([edu_num(du.identifier())])
