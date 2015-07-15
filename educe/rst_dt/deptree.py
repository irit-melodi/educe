#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Eric Kow
# License: CeCILL-B (BSD3-like)

"""
Convert RST trees to dependency trees and back.
"""

import itertools

from .annotation import EDU
from ..internalutil import treenode


class RstDtException(Exception):
    """
    Exceptions related to conversion between RST and DT trees.
    The general expectation is that we only raise these on bad
    input, but in practice, you may see them more in cases of
    implementation error somewhere in the conversion process.
    """
    def __init__(self, msg):
        super(RstDtException, self).__init__(msg)


class RstDepTree(object):
    """RST dependency tree"""

    _ROOT_HEAD = 0
    _ROOT_LABEL = 'ROOT'

    DEFAULT_HEAD = _ROOT_HEAD
    DEFAULT_LABEL = _ROOT_LABEL

    def __init__(self, edus=[], origin=None):
        _lpad = EDU.left_padding()
        self.edus = [_lpad] + edus
        # mapping from EDU num to idx
        self.idx = {e.num: i for i, e in enumerate(edus, start=1)}
        # init tree structure
        nb_edus = len(self.edus)
        _dft_head = self.DEFAULT_HEAD
        _dft_lbl = self.DEFAULT_LABEL
        self.heads = [_dft_head for _ in range(nb_edus)]
        self.labels = [_dft_lbl for _ in range(nb_edus)]
        self.deps = [[] for _ in range(nb_edus)]
        # set special values for fake root
        self.heads[0] = -1
        self.labels[0] = None

        # set fake root's origin and context to be the same as the first
        # real EDU's
        context = edus[0].context if edus else None
        origin = edus[0].origin if (origin is None and edus) else origin
        # update origin and context for fake root
        self.edus[0].set_context(context)
        self.edus[0].set_origin(origin)
        # update the dep tree's origin
        self.set_origin(origin)

    def append_edu(self, edu):
        """Append an EDU to the list of EDUs"""
        self.edus.append(edu)
        self.idx[edu.num] = len(self.edus) - 1
        # set default values for the tree structure
        self.heads.append(self.DEFAULT_HEAD)
        self.labels.append(self.DEFAULT_LABEL)
        self.deps.append([])

    def add_dependency(self, gov_num, dep_num, label=None):
        """Add a dependency from EDU gov_num to EDU dep_num, labelled label."""
        _idx_gov = self.idx[gov_num]
        _idx_dep = self.idx[dep_num]
        self.heads[_idx_dep] = _idx_gov
        self.labels[_idx_dep] = label
        self.deps[_idx_gov].append((label, _idx_dep))

    def get_dependencies(self):
        """Get the list of dependencies in this dependency tree.
        Each dependency is a 3-uple (gov, dep, label),
        gov and dep being EDUs.
        """
        edus = self.edus

        deps = self.edus[1:]
        gov_idxs = self.heads[1:]
        labels = self.labels[1:]

        result = [(edus[gov_idx], dep, lbl)
                  for gov_idx, dep, lbl
                  in itertools.izip(gov_idxs, deps, labels)]

        return result

    def set_root(self, root_num):
        """Designate an EDU as a real root of the RST tree structure"""
        _idx_fake_root = self._ROOT_HEAD
        _idx_root = self.idx[root_num]
        _lbl_root = self._ROOT_LABEL
        self.heads[_idx_root] = _idx_fake_root
        self.labels[_idx_root] = _lbl_root
        self.deps[_idx_fake_root].append((_lbl_root, _idx_root))

    def real_roots_idx(self):
        """Get the list of the indices of the real roots"""
        return self.deps[self._ROOT_HEAD]

    def set_origin(self, origin):
        """Update the origin of this annotation"""
        self.origin = origin

    @classmethod
    def from_simple_rst_tree(cls, rtree):
        """Converts a ̀SimpleRSTTree` to an `RstDepTree`"""
        edus = sorted(rtree.leaves(), key=lambda x: x.span.char_start)
        dtree = cls(edus)

        def walk(tree):
            """
            Recursively walk down tree, collecting dependency information
            between EDUs as we go.
            Return/percolate the num of the head found in our descent.
            """
            if len(tree) == 1:  # pre-terminal
                # edu = tree[0]
                edu_num = treenode(tree).edu_span[0]
                return edu_num
            else:
                rel = treenode(tree).rel
                left = tree[0]
                right = tree[1]
                nscode = "".join(treenode(kid).nuclearity[0] for kid in tree)
                lhead = walk(left)
                rhead = walk(right)

                if nscode == "NS":
                    head = lhead
                    child = rhead
                elif nscode == "SN":
                    head = rhead
                    child = lhead
                elif nscode == "NN":
                    head = lhead
                    child = rhead
                else:
                    raise RstDtException("Don't know how to handle %s trees" %
                                         nscode)
                dtree.add_dependency(head, child, rel)
                return head

        root = walk(rtree)  # populate tree structure in dtree and get its root
        dtree.set_root(root)

        return dtree
