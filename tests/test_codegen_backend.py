import os
from pathlib import Path

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed
from codegen_backend import codegen_add_backend, codegen_sub_backend
from codegen_backend.backend import get_add_source, get_sub_source


REFERENCE_DIR = Path(__file__).resolve().parent / "codegen_refs"


def _assert_codegen_source_matches(reference_name: str, source_fn, fn) -> None:
    reference_path = REFERENCE_DIR / reference_name
    gm = torch.fx.symbolic_trace(fn)
    source = source_fn(gm).lstrip()
    if os.getenv("UPDATE_CODEGEN_REFS"):
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(source, encoding="utf-8")
    expected = reference_path.read_text(encoding="utf-8")
    assert source == expected


def add_fn(a, b):
    return a + b


def mul_fn(a, b):
    return a * b


def add_chain_fn(a, b, c):
    return (a + b) + c


def sub_fn(a, b):
    return a - b


def sub_chain_fn(a, b, c):
    return (a - b) - c


@pytest.mark.parametrize(
    ("reference_name", "fn", "source_fn", "backend"),
    [
        ("add_matches_eager.c", add_fn, get_add_source, codegen_add_backend),
        ("sub_matches_eager.c", sub_fn, get_sub_source, codegen_sub_backend),
    ],
)
def test_codegen_binary_matches_eager(reference_name, fn, source_fn, backend):
    _assert_codegen_source_matches(reference_name, source_fn, fn)
    compiled = torch.compile(fn, backend=backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    result = compiled(a, b)
    torch.testing.assert_close(result, fn(a, b))


@pytest.mark.parametrize(
    ("reference_name", "fn", "source_fn", "backend"),
    [
        ("add_handles_non_contiguous.c", add_fn, get_add_source, codegen_add_backend),
        ("sub_handles_non_contiguous.c", sub_fn, get_sub_source, codegen_sub_backend),
    ],
)
def test_codegen_binary_handles_non_contiguous(
    reference_name, fn, source_fn, backend
):
    _assert_codegen_source_matches(reference_name, source_fn, fn)
    compiled = torch.compile(fn, backend=backend)
    a = torch.randn(4, 4, dtype=torch.float32).t()
    b = torch.randn(4, 4, dtype=torch.float32).t()
    result = compiled(a, b)
    torch.testing.assert_close(result, fn(a, b))


@pytest.mark.parametrize(
    ("reference_name", "fn", "source_fn", "backend"),
    [
        ("add_rejects_other_ops.c", add_fn, get_add_source, codegen_add_backend),
        ("sub_rejects_other_ops.c", sub_fn, get_sub_source, codegen_sub_backend),
    ],
)
def test_codegen_binary_rejects_other_ops(reference_name, fn, source_fn, backend):
    _assert_codegen_source_matches(reference_name, source_fn, fn)
    compiled = torch.compile(mul_fn, backend=backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    with pytest.raises(BackendCompilerFailed, match="Unsupported call_function"):
        compiled(a, b)


@pytest.mark.parametrize(
    ("reference_name", "fn", "source_fn", "backend"),
    [
        ("add_chain.c", add_chain_fn, get_add_source, codegen_add_backend),
        ("sub_chain.c", sub_chain_fn, get_sub_source, codegen_sub_backend),
    ],
)
def test_codegen_binary_handles_multiple_ops(
    reference_name, fn, source_fn, backend
):
    _assert_codegen_source_matches(reference_name, source_fn, fn)
    compiled = torch.compile(fn, backend=backend)
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(2, 3, dtype=torch.float32)
    c = torch.randn(2, 3, dtype=torch.float32)
    result = compiled(a, b, c)
    torch.testing.assert_close(result, fn(a, b, c))
