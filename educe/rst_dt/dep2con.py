"""Conversion between dependency and constituency trees.

TODO
----
* [ ] refactor strategies
* [ ] use label+nuclearity to help determine the order of attachment:
      greedily take successive modifiers that belong to the same
      underlying multinuclear relation
"""

from collections import namedtuple
import itertools

from .annotation import SimpleRSTTree, Node
from .deptree import RstDtException, NUC_N, NUC_S, NUC_R
from ..internalutil import treenode


class DummyNuclearityClassifier(object):
    """Predict the nuclearity of each EDU using simple rules.

    Parameters
    ----------
    strategy: str
        Strategy to use to generate predictions.

        * "unamb_else_most_frequent": predicts multinuclear when the
        relation label is unambiguously multinuclear in the training
        set, mononuclear otherwise.
        * "most_frequent_by_rel": predicts the most frequent nuclearity
        for the given relation label in the training set.

    TODO
    ----
    complete after `sklearn.dummy.DummyClassifier`
    """

    def __init__(self, strategy="unamb_else_most_frequent"):
        self.strategy = strategy

    def fit(self, X, y):
        """Fit the dummy classifier.

        FIXME: currently a no-op.

        Both X and y are royally ignored.

        Parameters
        ----------
        X: list of RstDepTrees

        y: array-like, shape = [n_samples]
            Target nuclearity array for each EDU of each RstDepTree.
        """
        if self.strategy not in ["unamb_else_most_frequent",
                                 "most_frequent_by_rel"]:
            raise ValueError("Unknown strategy type.")

        # special processing: ROOT is considered multinuclear
        multinuc_lbls = ['ROOT']
        if self.strategy == "unamb_else_most_frequent":
            # FIXME automatically get these from the training set
            multinuc_lbls.extend(['joint', 'same-unit', 'textual'])

        elif self.strategy == "most_frequent_by_rel":
            # FIXME very dirty hack to avoid loading the RST-DT corpus
            # upfront (triggered by the import of
            # load_corpus_as_dataframe)
            try:
                from .corpus_diagnostics import (
                    get_most_frequent_unuc,
                    load_corpus_as_dataframe
                )
            except IOError:
                raise
            train_df = load_corpus_as_dataframe(selection='train')
            multinuc_lbls.extend(rel_name for rel_name, mode_unuc
                                 in get_most_frequent_unuc(train_df).items()
                                 if mode_unuc == 'NN')
        self.multinuc_lbls_ = multinuc_lbls

        # FIXME properly implement fit for the different strategies
        return self

    def predict(self, X):
        """Perform classification on test RstDepTrees X.
        """
        y = []
        for dtree in X:
            # NB: we condition multinuclear relations on (i > head)
            yi = [(NUC_N if (i > head and rel in self.multinuc_lbls_)
                   else NUC_S)
                  for i, (head, rel)
                  in enumerate(itertools.izip(dtree.heads, dtree.labels))]
            y.append(yi)

        return y


class InsideOutAttachmentRanker(object):
    """Rank modifiers, from the inside out on either side.

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

    The ranking is determined by a strategy. Currently implemented
    strategies are:
    - 'id': stable sort that keeps the original interleaving, as
    defined below,
    - 'lllrrr': all left then right dependents,
    - 'rrrlll': all right then left dependents,
    - 'lrlrlr': alternating directions, left first,
    - 'rlrlrl': alternating directions, right first.

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

    def __init__(self, strategy='id', prioritize_same_unit=False):
        if strategy not in ['id',
                            'lllrrr', 'rrrlll',
                            'lrlrlr', 'rlrlrl',
                            'closest-lr', 'closest-rl',
                            'closest-intra-lr-inter-lr',
                            'closest-intra-rl-inter-rl',
                            'closest-intra-rl-inter-lr']:
            raise ValueError('Unknown transformation strategy '
                             '{stg}'.format(stg=strategy))
        self.strategy = strategy
        self.prioritize_same_unit = prioritize_same_unit

    def fit(self, X, y):
        """Here a no-op."""
        return self

    def predict(self, X):
        """Produce a ranking.

        This keeps the alternation of left and right modifiers as it is
        in `targets` but possibly re-orders dependents on either side to
        guarantee inside-out traversal.

        Parameters
        ----------
        X: list of RstDepTrees
            Dependency trees for which we want attachment rankings

        Returns
        -------
        dt_ranks: list of arrays of ranks
            Attachment ranking, one per RstDepTree

        Notes
        -----
        The '*intra*' strategies need to map each EDU to its surrounding
        sentence. This is currently done by attaching to the RstDepTree
        this mapping as a list, `sent_idx`.
        Future versions should make this right.
        """
        strategy = self.strategy

        dt_ranks = []
        for dtree in X:
            # for each RstDepTree, the result will be an array of ranks
            ranks = [0 for hd in dtree.heads]  # initialize result

            unique_heads = set(dtree.heads[1:])  # exclude head of fake root
            for head in unique_heads:
                targets = [i for i, hd in enumerate(dtree.heads)
                           if hd == head]
                # what follows should be well-tested code
                sorted_nodes = sorted(
                    [head] + targets,
                    key=lambda x: dtree.edus[x].span.char_start)
                centre = sorted_nodes.index(head)
                # elements to the left and right of the node respectively
                # these are stacks (outside ... inside)
                left = sorted_nodes[:centre]
                right = list(reversed(sorted_nodes[centre+1:]))

                # special strategy: 'id' (we know the true targets)
                if strategy == 'id':
                    result = [left.pop() if (tree in left) else right.pop()
                              for tree in targets]

                # strategies that try to guess the order of attachment
                else:
                    result = []

                    if self.prioritize_same_unit:
                        # gobble everything between the head and the rightmost
                        # "same-unit"
                        same_unit_tgts = [tgt for tgt in targets
                                          if dtree.labels[tgt] == 'same-unit']
                        if same_unit_tgts:
                            # take first all dependents between the head
                            # and the rightmost same-unit
                            last_same_unit_tgt = same_unit_tgts[-1]
                            priority_tgts = [tgt for tgt in targets
                                             if (tgt > head and
                                                 tgt <= last_same_unit_tgt)]
                            # prepend to the result
                            result.extend(priority_tgts)
                            # remove from the remaining targets
                            targets = [tgt for tgt in targets
                                       if tgt not in priority_tgts]

                    if strategy == 'lllrrr':
                        result.extend(left.pop() if left else right.pop()
                                      for _ in targets)

                    elif strategy == 'rrrlll':
                        result.extend(right.pop() if right else left.pop()
                                      for _ in targets)

                    elif strategy == 'lrlrlr':
                        # reverse lists of left and right modifiers
                        # these are queues (inside ... outside)
                        left_io = list(reversed(left))
                        right_io = list(reversed(right))
                        lrlrlr_gen = itertools.chain.from_iterable(
                            itertools.izip_longest(left_io, right_io))
                        result.extend(x for x in lrlrlr_gen
                                      if x is not None)

                    elif strategy == 'rlrlrl':
                        # reverse lists of left and right modifiers
                        # these are queues (inside ... outside)
                        left_io = list(reversed(left))
                        right_io = list(reversed(right))
                        rlrlrl_gen = itertools.chain.from_iterable(
                            itertools.izip_longest(right_io, left_io))
                        result.extend(x for x in rlrlrl_gen
                                      if x is not None)

                    elif strategy == 'closest-rl':
                        # take closest dependents first, take right over
                        # left to break ties
                        sort_key = lambda e: (abs(e - head),
                                              1 if e > head else 2)
                        result.extend(sorted(targets, key=sort_key))

                    elif strategy == 'closest-lr':
                        # take closest dependents first, take left over
                        # right to break ties
                        sort_key = lambda e: (abs(e - head),
                                              2 if e > head else 1)
                        result.extend(sorted(targets, key=sort_key))

                    # strategies that depend on intra/inter-sentential info
                    # NB: the way sentential info is stored is expected to
                    # change at some point
                    else:
                        if not hasattr(dtree, 'sent_idx'):
                            raise ValueError(('Strategy {stg} depends on '
                                              'sentential information which is'
                                              ' missing here'
                                              '').format(stg=strategy))

                        if strategy == 'closest-intra-rl-inter-lr':
                            # current best
                            # take closest dependents first, take right over
                            # left to break ties
                            sort_key = lambda e: (
                                1 if dtree.sent_idx[e] == dtree.sent_idx[head] else 2,
                                abs(e - head),
                                1 if ((e > head and
                                       dtree.sent_idx[e] == dtree.sent_idx[head]) or
                                      (e < head and
                                       dtree.sent_idx[e] != dtree.sent_idx[head])) else 2
                            )
                            result.extend(sorted(targets, key=sort_key))

                        elif strategy == 'closest-intra-rl-inter-rl':  # current used
                            # sent_idx for all EDUs that need to be locally
                            # ranked (+ their head)
                            loc_edus = sorted(targets + [head])
                            sent_idc = [dtree.sent_idx[x] for x in loc_edus
                                        if dtree.sent_idx[x] is not None]
                            if len(sent_idc) != len(loc_edus):
                                # missing sent_idx => (pseudo-)imputation ;
                                # this is a very local, and dirty, workaround
                                # * left dependents + head
                                sent_idc_left = []
                                sent_idx_cur = min(sent_idc) if sent_idc else 0
                                for x in loc_edus:
                                    if x > head:
                                        break
                                    sent_idx_x = dtree.sent_idx[x]
                                    if sent_idx_x is not None:
                                        sent_idx_cur = sent_idx_x
                                    sent_idc_left.append(sent_idx_cur)
                                # * right dependents
                                sent_idc_right = []
                                sent_idx_cur = max(sent_idc) if sent_idc else 0
                                for x in reversed(targets):
                                    if x <= head:
                                        break
                                    sent_idx_x = dtree.sent_idx[x]
                                    if sent_idx_x is not None:
                                        sent_idx_cur = sent_idx_x
                                    sent_idc_right.append(sent_idx_cur)
                                # * replace sent_idc with the result of the
                                # pseudo-imputation
                                sent_idc = (sent_idc_left +
                                            list(reversed(sent_idc_right)))
                            # build this into a dict
                            sent_idx_loc = {e: s_idx for e, s_idx
                                            in zip(loc_edus, sent_idc)}

                            # take closest dependents first, break ties by
                            # choosing right first, then left
                            sort_key = lambda e: (
                                abs(sent_idx_loc[e] - sent_idx_loc[head]),
                                abs(e - head),
                                1 if e > head else 2
                            )
                            result.extend(sorted(targets, key=sort_key))

                        elif strategy == 'closest-intra-lr-inter-lr':
                            # take closest dependents first, take left over
                            # right to break ties
                            sort_key = lambda e: (
                                1 if dtree.sent_idx[e] == dtree.sent_idx[head] else 2,
                                abs(e - head),
                                2 if e > head else 1
                            )
                            result.extend(sorted(targets, key=sort_key))

                        else:
                            raise RstDtException('Unknown transformation strategy'
                                                 ' {stg}'.format(stg=strategy))

                # update array of ranks for this deptree
                # ranks are 1-based
                for i, tgt in enumerate(result, start=1):
                    ranks[tgt] = i

            dt_ranks.append(ranks)
        return dt_ranks


def deptree_to_simple_rst_tree(dtree, allow_forest=False):
    r"""
    Given a dependency tree with attachment ranking and nuclearity,
    return a 'SimpleRSTTree'.
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

    def connect_trees(src, tgt, rel, nuc):
        """
        Return a partial tree, assigning order and nuclearity to
        child trees
        """
        tgt_nuc = nuc

        if src.span.overlaps(tgt.span):
            raise RstDtException("Span %s overlaps with %s " %
                                 (src.span, tgt.span))
        elif src.span <= tgt.span:
            left = parts_to_tree(NUC_N, src)
            right = parts_to_tree(tgt_nuc, tgt)
        else:
            left = parts_to_tree(tgt_nuc, tgt)
            right = parts_to_tree(NUC_N, src)

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

    def walk(ancestor, subtree):
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

        Parameters
        ----------
        ancestor: TreeParts
            TreeParts of the ancestor

        subtree: int
            Index of the head of the subtree

        Returns
        -------
        res: TreeParts
        """
        rel = dtree.labels[subtree]
        nuc = dtree.nucs[subtree]

        src = mk_leaf(dtree.edus[subtree])
        # descend into each child, but note that we are folding
        # rather than mapping, ie. we threading along a nested
        # RST tree as go from sibling to sibling
        ranked_targets = dtree.deps(subtree)
        for tgt in ranked_targets:
            src = walk(src, tgt)
        # ancestor is None in the case of the root node
        return connect_trees(ancestor, src, rel, nuc) if ancestor else src

    roots = dtree.real_roots_idx()
    if not allow_forest and len(roots) > 1:
        msg = ('Cannot convert RstDepTree to SimpleRSTTree, '
               'multiple roots: {}\t{}'.format(roots, dtree.__dict__))
        raise RstDtException(msg)

    srtrees = []
    for real_root in roots:
        rparts = walk(None, real_root)
        srtree = parts_to_tree(NUC_R, rparts)
        srtrees.append(srtree)

    # for the most common case, return the tree
    if not allow_forest:
        return srtrees[0]
    # otherwise return a forest of SimpleRSTTrees ; needed for e.g.
    # intra-sentential parsing with leaky sentences, or sentence-only
    # document parsing.
    return srtrees


# pylint: disable=R0903, W0232
class TreeParts(namedtuple("TreeParts_", "edu edu_span span rel kids")):
    """
    Partially built RST tree when converting from dependency tree
    Kids here is nuclearity-annotated children
    """
    pass
# pylint: enable=R0903, W0232
