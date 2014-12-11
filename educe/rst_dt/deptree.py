#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Eric Kow
# License: CeCILL-B (BSD3-like)

"""
Convert RST trees to dependency trees and back.
"""

from collections import namedtuple
import itertools

from educe.rst_dt.annotation import SimpleRSTTree, Node, EDU
from ..internalutil import treenode

_N = "Nucleus"
_S = "Satellite"
_R = "Root"


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
        self.heads = list(itertools.repeat(_dft_head, nb_edus))
        self.labels = list(itertools.repeat(_dft_lbl, nb_edus))
        self.deps = list(itertools.repeat([], nb_edus))
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

    def to_simple_rst_tree(self, multinuclear, strategy='id'):
        r"""
        Given a dependency tree and a collection of relation labels,
        to be interpreted as being multinuclear, return a 'SimpleRSTTree'.
        Note that nodes are ordered by text span, so that `e1 -R-> e2` and
        `e2 -R-> e1` would both be interpreted as `R(e1, e2)`, the
        potential difference being which is treated as the nucleus and as
        the satellite.

        The conversion algorithm can be seen a two-pass descent/conversion,
        ascent/assembly one. First, we go down the dependency tree,
        annotating each node with a trivial RST tree (ie. a leaf node).
        Then, we walk back up the dependency tree, connecting RST treelets
        together, gradually replacing nodes with increasingly larger RST
        trees.

        The first tricky part of this conversion is to keep in mind
        that the conversion process involves some rotation, namely
        that this dependency structure ::

               R
            a ---> b

        (if we gloss over nuclearity and node order details) is rotated
        into something that looks like this ::


            R +-> a                                 R
              |          (or more conventionally,  / \ )
              +-> b                               a   b


        Where things get tricky is if a node in the dependency tree
        points has multiple children ::

                r1
            src +--> tgt1
                |
                |r2
                +--> tgt2
                |
                ..
                |
                |rN
                +--> tgtN


        In that case, having pivoted the `src(r1:tgt1)` link into something
        like `r1(src, tgt)`, it may not be so clear how to deal with
        subsequents targets `tgt2..tgtN`. We want to both reflect that `src`
        has a dependency link with all of these targets at the same time
        even though it's stuck as an RST child node. The trick is to see the
        process as more of a fold than a map. Keeping in mind that `src`,
        `tgt1..tgtN` are all RST treelets (`tgt1..tgtN` having been built up
        from recursive descent), the idea is to gradually replace the src
        node with an increasingly nested RST tree as we go along ::


            +----------+
            |   r1     |    r2
            |  /  \    + -+---> tgt2
            | src tgt1 |  |
            +----------+  | r3
                          +---> tgt3
                          |
                          ..
                          | rN
                          +---> tgtN



            +------------+
            |     r2     |
            |    /  \    |
            |   r1  tgt2 |    r3
            |  /  \      + -+---> tgt3
            | src tgt1   |  |
            +------------+  ..
                            | rN
                            +---> tgtN


        (and so on, until all we have left is a single RST tree).
        """

        def tgt_nuclearity(rel):
            """
            The target of a dep tree link is normally the satellite
            unless the relation is marked multinuclear
            """
            return _N if rel in multinuclear else _S

        def mk_leaf(edu):
            """
            Trivial partial tree for use when processing dependency
            tree leaves
            """
            return TreeParts(edu=edu,
                             edu_span=(edu.num, edu.num),
                             span=edu.text_span(),
                             rel="leaf",
                             kids=[])

        def parts_to_tree(nuclearity, parts):
            """
            Combine root nuclearity information with a partial tree
            to form a full RST `SimpleTree`
            """
            node = Node(nuclearity,
                        parts.edu_span,
                        parts.span,
                        parts.rel)
            kids = parts.kids or [parts.edu]
            return SimpleRSTTree(node, kids)

        def connect_trees(src, tgt, rel):
            """
            Return a partial tree, assigning order and nuclearity to
            child trees
            """
            tgt_nuc = tgt_nuclearity(rel)

            if src.span.overlaps(tgt.span):
                raise RstDtException("Span %s overlaps with %s " %
                                     (src.span, tgt.span))
            elif src.span <= tgt.span:
                left = parts_to_tree(_N, src)
                right = parts_to_tree(tgt_nuc, tgt)
            else:
                left = parts_to_tree(tgt_nuc, tgt)
                right = parts_to_tree(_N, src)

            l_edu_span = treenode(left).edu_span
            r_edu_span = treenode(right).edu_span

            edu_span = (min(l_edu_span[0], r_edu_span[0]),
                        max(l_edu_span[1], r_edu_span[1]))
            res = TreeParts(edu=src.edu,
                            edu_span=edu_span,
                            span=src.span.merge(tgt.span),
                            rel=rel,
                            kids=[left, right])
            return res

        def _sort_inside_out(head, targets, strategy='id'):
            """
            Given a dependency tree node and its children, return the list
            of children but *stably* sorted to fulfill an inside-out
            traversal on either side.

            Let's back up a little bit for some background on this criterion.
            We assume that dependency tree nodes can be characterised and
            ordered by their position in the text. If we did such a thing, the
            head node would sit somewhere between its children ::

                lX, .., l2, l1, h, r1, r2, .., rY

            An inside-out traversal is one in which moves strictly outward
            from the centre node. Note that this does not imply any other
            requirement on the order our traversal (particularly on left vs
            right), so `h l1 l2 .. lX r1 r2, etc` is as much a valid
            inside-outside traversal as `h l1 r1 l2 etc`; however,
            `h l2 l1 etc` is right out.

            The following strategies are currently implemented:
            - 'lllrrr': all left then right dependents,
            - 'rrrlll': all right then left dependents,
            - 'id': stable sort that keeps the original interleaving,
            as defined below.

            TRICKY SORTING! The current implementation of the 'id' strategy was
            reached through a bit of trial and error, so you may want to modify
            with caution.

            Most of the trickiness in the 'id' strategy is in making this a
            *stable* sort, ie. we want to preserve the original order of
            the targets as much as possible because this allows us to have
            round trip conversions from RST to DT and back. This essentially
            means preserving the interleaving of left/right nodes. The basic
            logic in the implementation is to traverse our target list as
            a series of LEFT or RIGHT slots, filling the slots in an
            inside-out order. So for example, if we saw a target list
            `l3 r1 r3 l2 l1 r2`, we would treat it as the slots `L R R L L R`
            and fill them out as `l1 r1 r2 l2 l3 r3`
            """
            sorted_nodes = sorted([head] + targets,
                                  key=lambda x: self.edus[x].span.char_start)
            centre = sorted_nodes.index(head)
            # elements to the left and right of the node respectively
            # these are stacks (outside ... inside)
            left = sorted_nodes[:centre]
            right = list(reversed(sorted_nodes[centre+1:]))

            # built result according to strategy
            if strategy == 'id':
                result = [left.pop() if (tree in left) else right.pop()
                          for tree in targets]
            elif strategy == 'lllrrr':
                result = [left.pop() if left else right.pop()
                          for _ in targets]
            elif strategy == 'rrrlll':
                result = [right.pop() if right else left.pop()
                          for _ in targets]
            else:
                raise RstDtException('Unknown transformation strategy ',
                                     '{stg}'.format(stg=strategy))

            return result

        def walk(ancestor, subtree, strategy):
            """
            The basic descent/ascent driver of our conversion algorithm.
            Note that we are looking at three layers of the dependency
            tree at the same time.


                         r0       r1
                ancestor --> src +--> tgt1
                                 |
                                 |r2
                                 +--> tgt2
                                 |
                                 ..
                                 |
                                 |rN
                                 +--> tgtN

            The base case is if src is a leaf node (no children),
            whereupon we return a tiny tree connecting the two.

            If we do have children, we have to first obtain the
            full RST tree for src (through the folding process
            described in the docstring for the main function)
            before connecting it to its ancestor.
            """
            rel = self.labels[subtree]
            src = mk_leaf(self.edus[subtree])
            # descend into each child, but note that we are folding
            # rather than mapping, ie. we threading along a nested
            # RST tree as go from sibling to sibling
            targets = [t for _, t in self.deps[subtree]]
            for tgt in _sort_inside_out(subtree, targets, strategy):
                src = walk(src, tgt, strategy)
            # ancestor is None in the case of the root node
            return connect_trees(ancestor, src, rel) if ancestor else src

        roots = self.real_roots_idx()
        if len(roots) == 1:
            real_root = roots[0][1]  # roots is a list of (label, num)
            rparts = walk(None, real_root, strategy)
        else:
            msg = ('Cannot convert RstDepTree to SimpleRSTTree, ',
                   'multiple roots: {}'.format(roots))
            raise RstDtException(msg)
        return parts_to_tree(_R, rparts)


# pylint: disable=R0903, W0232
class TreeParts(namedtuple("TreeParts_", "edu edu_span span rel kids")):
    """
    Partially built RST tree when converting from dependency tree
    Kids here is nuclearity-annotated children
    """
    pass
# pylint: enable=R0903, W0232
