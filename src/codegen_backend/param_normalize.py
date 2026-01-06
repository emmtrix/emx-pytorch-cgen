from __future__ import annotations

import operator
from typing import Iterable, Literal, Tuple

import torch

DType = torch.dtype


def _format_allowed_strings(allowed: Iterable[str]) -> str:
    values = tuple(allowed)
    if not values:
        return ""
    return ", or one of " + ", ".join(repr(value) for value in values)


def _normalize_int_tuple(name: str, values: Iterable[object], n: int) -> Tuple[int, ...]:
    items = tuple(values)
    if len(items) != n:
        raise ValueError(f"{name} must be an int or a tuple of {n} ints")
    try:
        return tuple(operator.index(item) for item in items)
    except TypeError as exc:
        raise ValueError(f"{name} must be an int or a tuple of {n} ints") from exc


def normalize_int_or_pair(name: str, value: object) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return _normalize_int_tuple(name, value, 2)
    try:
        scalar = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an int or a tuple of 2 ints") from exc
    return (scalar, scalar)


def normalize_int_or_tuple(
    name: str, value: object, n: int
) -> Tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        return _normalize_int_tuple(name, value, n)
    try:
        scalar = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an int or a tuple of {n} ints") from exc
    return (scalar,) * n


def normalize_optional_int_or_tuple(
    name: str, value: object, n: int
) -> Tuple[int, ...] | None:
    if value is None:
        return None
    return normalize_int_or_tuple(name, value, n)


def normalize_padding(
    name: str,
    value: object,
    n: int,
    *,
    allow_strings: Iterable[str] = ("same", "valid"),
) -> Tuple[int, ...] | Literal["same", "valid"]:
    if isinstance(value, str):
        value_lower = value.lower()
        allowed = tuple(allow_strings)
        if value_lower in allowed:
            return value_lower  # type: ignore[return-value]
        raise ValueError(
            f"{name} must be an int, a tuple of {n} ints{_format_allowed_strings(allowed)}"
        )
    try:
        return normalize_int_or_tuple(name, value, n)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an int, a tuple of {n} ints{_format_allowed_strings(allow_strings)}"
        ) from exc


def normalize_bool(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a bool")


def normalize_dtype(name: str, value: object) -> DType | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    raise ValueError(f"{name} must be a torch.dtype or None")
