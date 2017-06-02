# Author: Eric Kow
# License: BSD3

"""
Convert RST trees to SDRT style EDU/CDU annotations.

The core of the conversion is `rst_to_sdrt` which produces an intermediary
pointer based representation (a single `CDU` pointing to other
CDUs and EDUs).

A fancier variant, `rst_to_glozz_sdrt` wraps around this core and further
converts the `CDU` into a Glozz-friendly form
"""

from nltk import Tree

import educe.rst_dt.parse as rst
import educe.annotation as anno
from educe import glozz


class CDU:
    """Complex Discourse Unit.

    A CDU contains one or more discourse units, and tracks relation
    instances between its members.
    Both CDU and EDU are discourse units.

    Attributes
    ----------
    members : list of Unit or Scheme
        Immediate member units (EDUs and CDUs) of this CDU.

    rel_insts : list of Relation
        Relation instances between immediate members of this CDU.
    """

    def __init__(self, members, rel_insts):
        self.members = members
        self.rel_insts = rel_insts


class RelInst:
    """Relation instance.

    `educe.annotation` calls these 'Relation's which is really more in
    keeping with how Glozz class them, but properly speaking relation
    instance is a better name.

    Attributes
    ----------
    source : Unit?
        Source of the relation instance.

    target : Unit?
        Target of the relation instance.

    type : string
        Name of the relation.
    """
    def __init__(self, source, target, type):
        self.source = source
        self.target = target
        self.type = type


def debug_du_to_tree(m):
    """Tree representation of CDU.

    The set of relation instances is treated as the parent of each node.
    Loses information ; should only be used for debugging purposes.
    """
    if isinstance(m, rst.EDU):
        return m
    elif isinstance(m, CDU):
        rtypes = set([r.type for r in m.rel_insts])
        rtype_str = list(rtypes)[0] if len(rtypes) == 1 else str(rtypes)
        return Tree(rtype_str, [debug_du_to_tree(x) for x in m.members])
    else:
        raise ValueError("Don't know how to deal with non CDU/EDU")


def rst_to_glozz_sdrt(rst_tree, annotator='ldc'):
    """
    From an RST tree to a STAC-like version using Glozz annotations.
    Uses `rst_to_sdrt`
    """
    intermediary_du = rst_to_sdrt(rst_tree)

    # flatten
    # create document
    # create identifiers
    # object to id map?
    rels = []
    edus = []
    cdus = []

    def walk(m):
        """Recursive helper to walk DUs"""
        if isinstance(m, rst.EDU):
            edus.append(m)
        elif isinstance(m, CDU):
            rels.extend(m.rel_insts)
            cdus.append(m)
            for c in m.members:
                walk(c)

    walk(intermediary_du)  # populate rels/edus/cdus

    orig_to_ctr = {}
    for ctr, x in enumerate(edus + rels + cdus, start=1):
        orig_to_ctr[x] = ctr

    def mk_id(ctr):
        return annotator + '_' + str(ctr)

    def mk_metadata(ctr):
        return {'author': annotator, 'creation-date': str(ctr)}

    def mk_info(obj):
        counter = orig_to_ctr[x]
        glozz_id = mk_id(counter)
        # features = {}
        metadata = mk_metadata(counter)
        return counter, glozz_id, metadata

    glozz_units = []
    glozz_rels = []
    glozz_schemas = []
    objects = {}

    for x in edus:
        counter, glozz_id, metadata = mk_info(x)
        features = {}
        glozz_unit = anno.Unit(glozz_id, x.span, "EDU", features, metadata)
        objects[glozz_id] = glozz_unit
        glozz_units.append(glozz_unit)

    for x in rels:
        counter, glozz_id, metadata = mk_info(x)
        features = {}
        rspan = anno.RelSpan(mk_id(orig_to_ctr[x.source]),
                             mk_id(orig_to_ctr[x.target]))
        glozz_rel = anno.Relation(glozz_id, rspan, x.type, features, metadata)
        objects[glozz_id] = glozz_rel
        glozz_rels.append(glozz_rel)

    for x in cdus:
        counter, glozz_id, metadata = mk_info(x)
        features = {}
        c_units = set(mk_id(orig_to_ctr[m])
                      for m in x.members if isinstance(m, rst.EDU))
        c_schema = set(mk_id(orig_to_ctr[m])
                       for m in x.members if isinstance(m, CDU))
        c_rels = set([])
        glozz_schema = anno.Schema(glozz_id, c_units, c_rels, c_schema,
                                   "CDU",
                                   features, metadata)
        objects[glozz_id] = glozz_schema
        glozz_schemas.append(glozz_schema)

    for x in glozz_rels + glozz_schemas:  # set internal pointers
        x.fleshout(objects)

    glozz_doc = glozz.GlozzDocument(None, glozz_units, glozz_rels,
                                    glozz_schemas, rst_tree.text())
    return glozz_doc


def rst_to_sdrt(tree):
    """
    From `RSTTree` to `CDU` or `EDU` (recursive, top-down transformation).
    We recognise three patterns walking down the tree (anything else is
    considered to be an error):

    * Pre-terminal nodes: Return the leaf EDU

    * Mono-nuclear, N satellites: Return a CDU with a relation instance
      from the nucleus to each satellite.  As an informal example, given
      `X(attribution:S1, N, explanation-argumentative:S2)`, we return a
      CDU with `sdrt(N) -- attribution --> sdrt(S1)` and
      `sdrt(N) -- explanation-argumentative --> sdrt(S2)`

    * Multi-nuclear, 0 satellites: Return a CDU with a relation instance
      across each successive nucleus (assume the same relation).  As an
      informal example, given `X(List:N1, List:N2, List:N3)`, we return a CDU
      containing `sdrt(N1) --List--> sdrt(N2) -- List --> sdrt(N3)`.
    """
    if len(tree) == 1:  # pre-terminal
        edu = tree[0]
        if not isinstance(edu, rst.EDU):
            raise ValueError("Pre-terminal with non-EDU leaf: %s" % edu)
        return edu
    else:
        nuclei = [x for x in tree if x.label().is_nucleus()]
        satellites = [x for x in tree if x.label().is_satellite()]
        if len(nuclei) + len(satellites) != len(tree):
            raise ValueError("Nodes that are neither Nuclei nor "
                             "Satellites\n%s" % tree)

        if len(nuclei) == 0:
            raise ValueError("No nucleus:\n%s" % tree)
        elif len(nuclei) > 1:  # multi-nuclear chain
            if satellites:
                raise ValueError("Multinuclear with satellites:\n%s" % tree)
            c_nucs = [rst_to_sdrt(x) for x in nuclei]
            rtype = nuclei[0].label().rel
            rel_insts = set(RelInst(n1, n2, rtype)
                            for n1, n2 in zip(c_nucs, c_nucs[1:]))
            return CDU(c_nucs, rel_insts)
        else:
            nuc = nuclei[0]
            c_nuc = rst_to_sdrt(nuc)
            c_sats = [rst_to_sdrt(x) for x in satellites]
            rel_insts = set(RelInst(c_nuc, cs, s.label().rel)
                            for s, cs in zip(satellites, c_sats))
            members = [c_nuc] + c_sats
            return CDU(members, rel_insts)
