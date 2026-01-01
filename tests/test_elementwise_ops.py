import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import SampleInput, op_db
from torch.testing._internal.common_utils import TestCase, run_tests

from ref_backend.backend import ref_backend_backend
from ref_backend.cffi_bindings import RefBackendError


OP_TEST_CONFIG = {
    "add": {
        "allowed_dtypes": (torch.float32,),
    },
    "sub": {
        "allowed_dtypes": (torch.float32,),
    },
    "mul": {
        "allowed_dtypes": (torch.float32,),
    },
}

DEFAULT_CONSTRAINTS = {
    "allowed_dtypes": None,
    "allow_noncontiguous": True,
    "max_ndim": 8,
    "requires_same_shape": True,
    "max_ndim_error": None,
    "shape_error": None,
    "skip_invalid_shape_tests": False,
}

OPS_UNDER_TEST = [op for op in op_db if op.name in OP_TEST_CONFIG]


def _compile_op(op):
    def compiled_fn(*args: torch.Tensor) -> torch.Tensor:
        return op(*args)

    return torch.compile(compiled_fn, backend=ref_backend_backend)


def _constraints_for(op):
    constraints = DEFAULT_CONSTRAINTS.copy()
    constraints.update(OP_TEST_CONFIG[op.name])
    return constraints


def _all_same_shape(tensors):
    if not tensors:
        return True
    shape = tensors[0].shape
    return all(tensor.shape == shape for tensor in tensors[1:])


def _iter_supported_samples(op, device, dtype, constraints):
    for sample in op.sample_inputs(device, dtype):
        if sample.kwargs:
            continue
        tensors = [sample.input, *sample.args]
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            continue
        if constraints["requires_same_shape"] and not _all_same_shape(tensors):
            continue
        if not all(tensor.dtype is dtype for tensor in tensors):
            continue
        yield sample

        if constraints["allow_noncontiguous"]:
            if all(tensor.ndim >= 2 for tensor in tensors):
                transposed = [tensor.transpose(0, 1) for tensor in tensors]
                yield SampleInput(transposed[0], args=tuple(transposed[1:]))

            if all(tensor.ndim >= 1 and tensor.size(-1) > 1 for tensor in tensors):
                sliced = [tensor[..., ::2] for tensor in tensors]
                if not constraints["requires_same_shape"] or _all_same_shape(sliced):
                    yield SampleInput(sliced[0], args=tuple(sliced[1:]))


def _get_op_arity(op, device, dtype, constraints):
    for sample in _iter_supported_samples(op, device, dtype, constraints):
        return 1 + len(sample.args)
    return None


class TestElementwiseOpInfo(TestCase):
    @ops(OPS_UNDER_TEST)
    def test_ref_backend_matches_eager(self, device, dtype, op):
        constraints = _constraints_for(op)
        allowed_dtypes = constraints["allowed_dtypes"]
        if allowed_dtypes is not None and dtype not in allowed_dtypes:
            pytest.skip("dtype not supported by test constraints")
        compiled = _compile_op(op)
        for sample in _iter_supported_samples(op, device, dtype, constraints):
            inputs = (sample.input, *sample.args)
            result = compiled(*inputs)
            torch.testing.assert_close(result, op(*inputs))

    @ops(OPS_UNDER_TEST)
    def test_ref_backend_rejects_invalid_shapes(self, device, dtype, op):
        constraints = _constraints_for(op)
        if constraints["skip_invalid_shape_tests"]:
            pytest.skip("invalid-shape checks disabled by test constraints")
        allowed_dtypes = constraints["allowed_dtypes"]
        if allowed_dtypes is not None and dtype not in allowed_dtypes:
            pytest.skip("dtype not supported by test constraints")
        compiled = _compile_op(op)
        arity = _get_op_arity(op, device, dtype, constraints)
        if arity is None:
            pytest.skip("no supported sample inputs for this dtype")
        max_ndim = constraints["max_ndim"]
        if max_ndim is not None:
            too_many_dims = torch.randn((1,) * (max_ndim + 1), device=device, dtype=dtype)
            max_ndim_error = constraints["max_ndim_error"]
            if max_ndim_error is None:
                max_ndim_error = f"{op.name} supports at most {max_ndim} dimensions"
            with pytest.raises(RefBackendError, match=max_ndim_error):
                compiled(*([too_many_dims] * arity))

        if constraints["requires_same_shape"] and arity >= 2:
            a = torch.randn((2, 3), device=device, dtype=dtype)
            b = torch.randn((2, 4), device=device, dtype=dtype)
            shape_error = constraints["shape_error"]
            if shape_error is None:
                shape_error = (
                    f"{op.name} requires inputs and output to have identical shapes"
                )
            with pytest.raises(RefBackendError, match=shape_error):
                mismatched = [a] * (arity - 1) + [b]
                compiled(*mismatched)


instantiate_device_type_tests(TestElementwiseOpInfo, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()
