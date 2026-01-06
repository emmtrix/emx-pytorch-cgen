import pytest
import torch

from codegen_backend.param_normalize import (
    normalize_bool,
    normalize_dtype,
    normalize_int_or_pair,
    normalize_int_or_tuple,
    normalize_optional_int_or_tuple,
    normalize_padding,
)


def _legacy_normalize_pair(value):
    if isinstance(value, int):
        return (value, value)
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return (value[0], value[1])
    raise ValueError("legacy pair invalid")


def _legacy_normalize_single(value):
    if isinstance(value, int):
        return value
    if (
        isinstance(value, (tuple, list))
        and len(value) == 1
        and all(isinstance(item, int) for item in value)
    ):
        return value[0]
    raise ValueError("legacy single invalid")


def test_normalize_int_or_pair_valid():
    assert normalize_int_or_pair("param", 3) == (3, 3)
    assert normalize_int_or_pair("param", (1, 2)) == (1, 2)
    assert normalize_int_or_pair("param", [4, 5]) == (4, 5)


def test_normalize_int_or_pair_invalid():
    with pytest.raises(ValueError, match=r"param must be an int or a tuple of 2 ints"):
        normalize_int_or_pair("param", (1,))
    with pytest.raises(ValueError, match=r"param must be an int or a tuple of 2 ints"):
        normalize_int_or_pair("param", 1.5)


def test_normalize_int_or_tuple_valid():
    assert normalize_int_or_tuple("param", 2, 3) == (2, 2, 2)
    assert normalize_int_or_tuple("param", (1, 2, 3), 3) == (1, 2, 3)
    assert normalize_int_or_tuple("param", [4, 5, 6], 3) == (4, 5, 6)


def test_normalize_int_or_tuple_invalid():
    with pytest.raises(ValueError, match=r"param must be an int or a tuple of 3 ints"):
        normalize_int_or_tuple("param", (1, 2), 3)
    with pytest.raises(ValueError, match=r"param must be an int or a tuple of 3 ints"):
        normalize_int_or_tuple("param", 1.1, 3)


def test_normalize_optional_int_or_tuple():
    assert normalize_optional_int_or_tuple("param", None, 2) is None
    assert normalize_optional_int_or_tuple("param", 4, 2) == (4, 4)
    with pytest.raises(ValueError, match=r"param must be an int or a tuple of 2 ints"):
        normalize_optional_int_or_tuple("param", (1, 2, 3), 2)


def test_normalize_padding_valid():
    assert normalize_padding("padding", 1, 2) == (1, 1)
    assert normalize_padding("padding", (1, 2), 2) == (1, 2)
    assert normalize_padding("padding", "same", 2) == "same"
    assert normalize_padding("padding", "VALID", 2) == "valid"


def test_normalize_padding_invalid():
    with pytest.raises(
        ValueError,
        match=r"padding must be an int, a tuple of 2 ints, or one of 'same', 'valid'",
    ):
        normalize_padding("padding", "full", 2)
    with pytest.raises(
        ValueError,
        match=r"padding must be an int, a tuple of 2 ints, or one of 'same', 'valid'",
    ):
        normalize_padding("padding", 1.25, 2)


def test_normalize_bool():
    assert normalize_bool("flag", True) is True
    with pytest.raises(ValueError, match=r"flag must be a bool"):
        normalize_bool("flag", 1)


def test_normalize_dtype():
    assert normalize_dtype("dtype", None) is None
    assert normalize_dtype("dtype", torch.float32) is torch.float32
    with pytest.raises(ValueError, match=r"dtype must be a torch\.dtype or None"):
        normalize_dtype("dtype", 123)


@pytest.mark.parametrize("value", [1, (2, 3), [4, 5]])
def test_regression_pair_matches_legacy(value):
    assert normalize_int_or_pair("param", value) == _legacy_normalize_pair(value)


@pytest.mark.parametrize("value", [(1,), [2], 3])
def test_regression_single_matches_legacy(value):
    assert normalize_int_or_tuple("param", value, 1)[0] == _legacy_normalize_single(
        value
    )


@pytest.mark.parametrize("value", [(1, 2, 3), [1, 2, 3], 1.5])
def test_regression_invalid_pair_matches_legacy(value):
    with pytest.raises(ValueError):
        normalize_int_or_pair("param", value)
    with pytest.raises(ValueError):
        _legacy_normalize_pair(value)
