from educe.rst_dt.learning.base import relative_indices


def test_relative_indices():
    """Test for relative_indices"""
    # first example: common case
    example1 = [0, 1, 1, 1, 2, 3, 3]

    rel_exa1 = [0, 0, 1, 2, 0, 0, 1]
    assert relative_indices(example1) == rel_exa1

    inv_exa1 = [0, 2, 1, 0, 0, 1, 0]
    assert relative_indices(example1, reverse=True) == inv_exa1

    # second example: case with None
    # each None is considered to be a distinct subgroup
    example2 = [None, 0, 1, 1, None, None]
    
    rel_exa2 = [0, 0, 0, 1, 0, 0]
    assert relative_indices(example2) == rel_exa2

    inv_exa2 = [0, 0, 1, 0, 0, 0]
    assert relative_indices(example2, reverse=True) == inv_exa2
