import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import SampleInput, op_db
from torch.testing._internal.common_utils import TestCase, run_tests

from ref_backend.backend import ref_backend_backend
from ref_backend.cffi_bindings import RefBackendError


ADD_SUB_OPS = [op for op in op_db if op.name in ("add", "sub", "mul")]


def _compile_op(op):
    def compiled_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return op(a, b)

    return torch.compile(compiled_fn, backend=ref_backend_backend)


def _iter_supported_samples(op, device, dtype):
    for sample in op.sample_inputs(device, dtype):
        if sample.kwargs:
            continue
        if len(sample.args) != 1:
            continue
        other = sample.args[0]
        if not isinstance(other, torch.Tensor):
            continue
        if sample.input.shape != other.shape:
            continue
        if sample.input.dtype is not dtype or other.dtype is not dtype:
            continue
        yield sample

        if sample.input.ndim >= 2:
            a_t = sample.input.transpose(0, 1)
            b_t = other.transpose(0, 1)
            yield SampleInput(a_t, args=(b_t,))

        if sample.input.ndim >= 1 and sample.input.size(-1) > 1:
            a_s = sample.input[..., ::2]
            b_s = other[..., ::2]
            if a_s.shape == b_s.shape:
                yield SampleInput(a_s, args=(b_s,))


class TestAddSubOpInfo(TestCase):
    @ops(ADD_SUB_OPS, allowed_dtypes=(torch.float32,))
    def test_ref_backend_matches_eager(self, device, dtype, op):
        compiled = _compile_op(op)
        for sample in _iter_supported_samples(op, device, dtype):
            a = sample.input
            b = sample.args[0]
            result = compiled(a, b)
            torch.testing.assert_close(result, op(a, b))

    @ops(ADD_SUB_OPS, allowed_dtypes=(torch.float32,))
    def test_ref_backend_rejects_invalid_shapes(self, device, dtype, op):
        compiled = _compile_op(op)
        too_many_dims = torch.randn((1,) * 9, device=device, dtype=dtype)
        with pytest.raises(
            RefBackendError, match=f"{op.name} supports at most 8 dimensions"
        ):
            compiled(too_many_dims, too_many_dims)

        a = torch.randn((2, 3), device=device, dtype=dtype)
        b = torch.randn((2, 4), device=device, dtype=dtype)
        with pytest.raises(
            RefBackendError,
            match=f"{op.name} requires inputs and output to have identical shapes",
        ):
            compiled(a, b)


instantiate_device_type_tests(TestAddSubOpInfo, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()
