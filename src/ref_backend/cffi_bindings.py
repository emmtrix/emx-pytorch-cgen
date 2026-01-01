import ctypes
import importlib
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


class RefBackendError(RuntimeError):
    pass


class RefDType:
    REF_F32 = 0


class RefOpKind:
    REF_OP_ADD = 0
    REF_OP_SUB = 1
    REF_OP_MUL = 2
    REF_OP_MATMUL = 3
    REF_OP_BMM = 4
    REF_OP_BROADCAST_IN_DIM = 5
    REF_OP_DIV = 6
    REF_OP_MAXIMUM = 7
    REF_OP_MINIMUM = 8
    REF_OP_NEG = 9
    REF_OP_EXP = 10
    REF_OP_ABS = 11
    REF_OP_SQRT = 12
    REF_OP_LOG = 13
    REF_OP_SIN = 14
    REF_OP_COS = 15
    REF_OP_TANH = 16
    REF_OP_FLOOR = 17
    REF_OP_CEIL = 18
    REF_OP_RECIPROCAL = 19
    REF_OP_RELU = 20


class RefTensorView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("ndim", ctypes.c_int32),
        ("sizes", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("dtype", ctypes.c_int32),
    ]


class RefOpCall(ctypes.Structure):
    _fields_ = [
        ("inputs", ctypes.POINTER(RefTensorView)),
        ("n_inputs", ctypes.c_int32),
        ("outputs", ctypes.POINTER(RefTensorView)),
        ("n_outputs", ctypes.c_int32),
        ("params", ctypes.c_void_p),
    ]


class RefBroadcastInDimParams(ctypes.Structure):
    _fields_ = [
        ("n_dims", ctypes.c_int32),
        ("broadcast_dimensions", ctypes.POINTER(ctypes.c_int32)),
    ]


@dataclass
class TensorViewBuffers:
    sizes: ctypes.Array
    strides: ctypes.Array


class RefBackendLibrary:
    def __init__(self) -> None:
        module = importlib.import_module("ref_backend._ref_backend")
        self.lib = ctypes.CDLL(module.__file__)
        self.lib.ref_run_op.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(RefOpCall),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.ref_run_op.restype = ctypes.c_int32

    def run_op(self, op_kind: int, call: RefOpCall) -> None:
        err_cap = 512
        err_buf = ctypes.create_string_buffer(err_cap)
        rc = self.lib.ref_run_op(op_kind, ctypes.byref(call), err_buf, err_cap)
        if rc != 0:
            msg = err_buf.value.decode("utf-8")
            raise RefBackendError(f"ref_run_op failed ({rc}): {msg}")


_lib_instance: Optional[RefBackendLibrary] = None


def _get_library() -> RefBackendLibrary:
    global _lib_instance
    if _lib_instance is None:
        _lib_instance = RefBackendLibrary()
    return _lib_instance


def _dtype_to_ref(dtype: torch.dtype) -> int:
    if dtype is torch.float32:
        return RefDType.REF_F32
    raise RefBackendError(f"Unsupported dtype: {dtype}")


def _tensor_to_view(tensor: torch.Tensor) -> Tuple[RefTensorView, TensorViewBuffers]:
    sizes = (ctypes.c_int64 * tensor.ndim)(*tensor.size())
    strides = (ctypes.c_int64 * tensor.ndim)(*tensor.stride())
    view = RefTensorView(
        data=ctypes.c_void_p(tensor.data_ptr()),
        ndim=ctypes.c_int32(tensor.ndim),
        sizes=sizes,
        strides=strides,
        dtype=_dtype_to_ref(tensor.dtype),
    )
    return view, TensorViewBuffers(sizes=sizes, strides=strides)


def _build_call(
    inputs: Tuple[torch.Tensor, ...],
    outputs: Tuple[torch.Tensor, ...],
    params: Optional[ctypes.c_void_p] = None,
) -> Tuple[RefOpCall, Tuple[object, ...]]:
    input_views = []
    output_views = []
    buffers = []
    for tensor in inputs:
        view, buf = _tensor_to_view(tensor)
        input_views.append(view)
        buffers.append(buf)
    for tensor in outputs:
        view, buf = _tensor_to_view(tensor)
        output_views.append(view)
        buffers.append(buf)
    input_array = (RefTensorView * len(inputs))(*input_views)
    output_array = (RefTensorView * len(outputs))(*output_views)
    call = RefOpCall(
        inputs=input_array,
        n_inputs=ctypes.c_int32(len(inputs)),
        outputs=output_array,
        n_outputs=ctypes.c_int32(len(outputs)),
        params=params,
    )
    buffers.extend([input_array, output_array, params])
    return call, tuple(buffers)


def _validate_float32(op_name: str, *tensors: torch.Tensor) -> None:
    if any(tensor.dtype is not torch.float32 for tensor in tensors):
        raise RefBackendError(f"{op_name} supports only torch.float32 tensors")


def _validate_max_dims(op_name: str, *tensors: torch.Tensor, max_dims: int = 8) -> None:
    if any(tensor.ndim > max_dims for tensor in tensors):
        raise RefBackendError(f"{op_name} supports at most {max_dims} dimensions")


def _run_binary_elementwise(
    op_name: str, op_kind: int, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> None:
    _validate_float32(op_name, a, b, out)
    if a.shape != b.shape or a.shape != out.shape:
        raise RefBackendError(
            f"{op_name} requires inputs and output to have identical shapes"
        )
    _validate_max_dims(op_name, a, b, out)
    call, buffers = _build_call((a, b), (out,))
    _ = buffers
    _get_library().run_op(op_kind, call)


def _run_unary_elementwise(
    op_name: str, op_kind: int, a: torch.Tensor, out: torch.Tensor
) -> None:
    _validate_float32(op_name, a, out)
    if a.shape != out.shape:
        raise RefBackendError(
            f"{op_name} requires input and output to have identical shapes"
        )
    _validate_max_dims(op_name, a, out)
    call, buffers = _build_call((a,), (out,))
    _ = buffers
    _get_library().run_op(op_kind, call)


def run_add(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("add", RefOpKind.REF_OP_ADD, a, b, out)


def run_sub(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("sub", RefOpKind.REF_OP_SUB, a, b, out)


def run_mul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("mul", RefOpKind.REF_OP_MUL, a, b, out)


def run_div(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("div", RefOpKind.REF_OP_DIV, a, b, out)


def run_maximum(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("maximum", RefOpKind.REF_OP_MAXIMUM, a, b, out)


def run_minimum(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    _run_binary_elementwise("minimum", RefOpKind.REF_OP_MINIMUM, a, b, out)


def run_neg(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("neg", RefOpKind.REF_OP_NEG, a, out)


def run_exp(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("exp", RefOpKind.REF_OP_EXP, a, out)


def run_abs(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("abs", RefOpKind.REF_OP_ABS, a, out)


def run_sqrt(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("sqrt", RefOpKind.REF_OP_SQRT, a, out)


def run_log(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("log", RefOpKind.REF_OP_LOG, a, out)


def run_sin(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("sin", RefOpKind.REF_OP_SIN, a, out)


def run_cos(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("cos", RefOpKind.REF_OP_COS, a, out)


def run_tanh(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("tanh", RefOpKind.REF_OP_TANH, a, out)


def run_floor(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("floor", RefOpKind.REF_OP_FLOOR, a, out)


def run_ceil(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("ceil", RefOpKind.REF_OP_CEIL, a, out)


def run_reciprocal(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("reciprocal", RefOpKind.REF_OP_RECIPROCAL, a, out)


def run_relu(a: torch.Tensor, out: torch.Tensor) -> None:
    _run_unary_elementwise("relu", RefOpKind.REF_OP_RELU, a, out)


def run_matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if (
        a.dtype is not torch.float32
        or b.dtype is not torch.float32
        or out.dtype is not torch.float32
    ):
        raise RefBackendError("matmul supports only torch.float32 tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise RefBackendError("matmul requires 2D inputs and output")
    if a.shape[1] != b.shape[0]:
        raise RefBackendError("matmul requires inner dimensions to match")
    if out.shape != (a.shape[0], b.shape[1]):
        raise RefBackendError("matmul requires output shape (m, n)")
    if not a.is_contiguous() or not b.is_contiguous() or not out.is_contiguous():
        raise RefBackendError("matmul requires contiguous tensors")
    call, buffers = _build_call((a, b), (out,))
    _ = buffers
    _get_library().run_op(RefOpKind.REF_OP_MATMUL, call)


def run_bmm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if (
        a.dtype is not torch.float32
        or b.dtype is not torch.float32
        or out.dtype is not torch.float32
    ):
        raise RefBackendError("bmm supports only torch.float32 tensors")
    if a.ndim != 3 or b.ndim != 3 or out.ndim != 3:
        raise RefBackendError("bmm requires 3D inputs and output")
    if a.shape[0] != b.shape[0]:
        raise RefBackendError("bmm requires batch dimensions to match")
    if a.shape[2] != b.shape[1]:
        raise RefBackendError("bmm requires inner dimensions to match")
    if out.shape != (a.shape[0], a.shape[1], b.shape[2]):
        raise RefBackendError("bmm requires output shape (batch, m, n)")
    if not a.is_contiguous() or not b.is_contiguous() or not out.is_contiguous():
        raise RefBackendError("bmm requires contiguous tensors")
    call, buffers = _build_call((a, b), (out,))
    _ = buffers
    _get_library().run_op(RefOpKind.REF_OP_BMM, call)


def run_broadcast_in_dim(
    a: torch.Tensor, out: torch.Tensor, broadcast_dimensions: Tuple[int, ...]
) -> None:
    if a.dtype is not torch.float32 or out.dtype is not torch.float32:
        raise RefBackendError("broadcast_in_dim supports only torch.float32 tensors")
    if not a.is_contiguous() or not out.is_contiguous():
        raise RefBackendError("broadcast_in_dim requires contiguous tensors")
    if len(broadcast_dimensions) != a.ndim:
        raise RefBackendError(
            "broadcast_in_dim expects broadcast_dimensions to match input rank"
        )
    if out.ndim < a.ndim:
        raise RefBackendError("broadcast_in_dim requires output rank >= input rank")

    dims = (ctypes.c_int32 * len(broadcast_dimensions))(*broadcast_dimensions)
    params = RefBroadcastInDimParams(
        n_dims=ctypes.c_int32(len(broadcast_dimensions)),
        broadcast_dimensions=dims,
    )

    call, buffers = _build_call(
        (a,),
        (out,),
        params=ctypes.cast(ctypes.pointer(params), ctypes.c_void_p),
    )
    _ = (buffers, dims, params)
    _get_library().run_op(RefOpKind.REF_OP_BROADCAST_IN_DIM, call)
