# Sample test
# Test files must be named in the form of `*_test.py` or `test_*.py`
import pytest


def func(x):
    return x + 1


def test_answer():
    assert func(3) == 4


def test_lambdas():
    # For refrence for how to make lambda functions and what they do.
    assert (lambda x: x + 1)(3) == 4
    assert (lambda x, y: x + y)(2, 5) == 7
    assert (lambda x: func(3) == x)(4)


def test_fail():
    assert not func(3) == 5


def test_except():
    with pytest.raises(TypeError):
        assert func({"a": 2})
