# Author: Eric Kow
# License: BSD3

"""
Convert RST trees to SDRT style EDU/CDU annotations.

At the moment, this uses a pointer-based representation which sits in between
the Glozz-based representation used for STAC and the hypergraph representation.
We try to resemble educe.annotation via duck-typing
"""

import educe.rst_dt.parse as rst
import educe.annotation   as anno

import itertools
from nltk import Tree

class CDU:
    """
    A CDU contains one or more discourse units.  Both CDU and EDU are
    discourse units
    """

    def __init__(self, members, rel_insts):
        """
        rel_insts refers to any relation instances between immediate
        members of this CDU
        """
        self.members   = members
        self.rel_insts = rel_insts

class RelInst:
    """
    Relation instance (educe.annotation calls these 'Relation's which is
    really more in keeping with how Glozz class them, but properly speaking
    relation instance is a better name)
    """
    def __init__(self, source, target, type):
        self.source = source
        self.target = target
        self.type   = type

def is_node(type):
    def f(x):
        return isinstance(x, rst.RSTTree) and x.node.type == type
    return f

def du_to_tree(m):
    if isinstance(m, rst.EDU):
        return m
    elif isinstance(m, CDU):
        rtypes = set([r.type for r in m.rel_insts])
        rtype_str = list(rtypes)[0] if len(rtypes) == 1 else str(rtypes)
        return Tree(rtype_str, [du_to_tree(m) for m in m.members])
    else:
        raise Exception("Don't know how to deal with non CDU/EDU")

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
        if isinstance(m, rst.EDU):
            edus.append(m)
        elif isinstance(m, CDU):
            rels.extend(m.rel_insts)
            cdus.append(m)
            for c in m.members:
                walk(c)
    walk(intermediary_du) # populate rels/edus/cdus

    orig_to_ctr = {}
    for ctr, x in enumerate(edus + rels + cdus, start=1):
        orig_to_ctr[x] = ctr

    def mk_id(ctr):
        return annotator + '_' + str(ctr)

    def mk_metadata(ctr):
        return {'author':annotator, 'creation-date':str(ctr) }

    def mk_info(obj):
        counter  = orig_to_ctr[x]
        glozz_id = mk_id(counter)
        features = {}
        metadata = mk_metadata(counter)
        return counter, glozz_id, metadata

    glozz_units   = []
    glozz_rels    = []
    glozz_schemas = []
    objects       = {}

    for x in edus:
        counter, glozz_id, metadata = mk_info(x)
        features           = {}
        glozz_unit         = anno.Unit(glozz_id, x.span, "EDU", features, metadata)
        objects[glozz_id]  = glozz_unit
        glozz_units.append(glozz_unit)

    for x in rels:
        counter, glozz_id, metadata = mk_info(x)
        features  = {}
        rspan     = anno.RelSpan(mk_id(orig_to_ctr[x.source]),
                                 mk_id(orig_to_ctr[x.target]))
        glozz_rel = anno.Relation(glozz_id, rspan, x.type, features, metadata)
        objects[glozz_id] = glozz_rel
        glozz_rels.append(glozz_rel)

    for x in cdus:
        counter, glozz_id, metadata = mk_info(x)
        features  = {}
        c_units   = set(mk_id(orig_to_ctr[m]) for m in x.members if isinstance(m,rst.EDU))
        c_schema  = set(mk_id(orig_to_ctr[m]) for m in x.members if isinstance(m,CDU))
        c_rels    = set([])
        glozz_schema = anno.Schema(glozz_id, c_units, c_rels, c_schema,
                                   "CDU",
                                   features, metadata)
        objects[glozz_id] = glozz_schema
        glozz_schemas.append(glozz_schema)

    for x in glozz_rels + glozz_schemas: # set internal pointers
        x.fleshout(objects)

    glozz_doc = anno.Document(glozz_units, glozz_rels, glozz_schemas, rst_tree.text())
    return glozz_doc


def rst_to_sdrt(tree):
    if len(tree) == 1: # pre-terminal
        edu = tree[0]
        if not isinstance(edu, rst.EDU):
            raise Exception("Pre-terminal with non-EDU leaf: %s" % edu)
        return edu
    else:
        nuclei     = filter(is_node('Nucleus'),   tree)
        satellites = filter(is_node('Satellite'), tree)
        if len(nuclei) + len(satellites) != len(tree):
            raise Exception("We have nodes that are neither Nuclei nor Satellites")

        if len(nuclei) == 0:
            raise Exception("No Nucleus nodes in %s" % tree)
        elif len(nuclei) > 1: # multi-nuclear chain
            c_nucs = map(rst_to_sdrt, nuclei)
            rtype  = nuclei[0].node.rel
            rel_insts = [ RelInst(n1, n2, rtype) for n1,n2 in zip(c_nucs, c_nucs[1:]) ]
            return CDU(c_nucs, set(rel_insts))
        elif len(tree) == 2: # mono-nuclear, should be exactly two nodes
            if len(nuclei) == 1 and len(satellites) == 1:
                nuc   = nuclei[0]
                sat   = satellites[0]

                c_nuc = rst_to_sdrt(nuc)
                c_sat = rst_to_sdrt(sat)
                rel_inst = RelInst(c_nuc, c_sat, sat.node.rel)
                return CDU([c_nuc, c_sat], set([rel_inst]))
            else:
                len_nuc = len(nuclei)
                len_sat = len(satellites)
                raise Exception("Don't yet know what to do with %d N, %d S" % (len_nuc, len_sat))
        else:
            raise Exception("Mononuclear what to do with %d children" % len(tree))
