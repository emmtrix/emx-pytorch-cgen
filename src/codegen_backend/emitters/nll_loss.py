from __future__ import annotations

from typing import List, Sequence

import torch

from codegen_backend.c_types import _dtype_to_c_type, _format_scalar_literal
from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    _is_contiguous,
)
from codegen_backend.errors import CodegenBackendError
from codegen_backend.indexing import _emit_strided_access
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_nll_loss_kernel(
    node_index: int,
    op_spec: _OpSpec,
    input_shape: Sequence[int],
    target_shape: Sequence[int],
    weight_shape: Sequence[int] | None,
    input_strides: Sequence[int],
    target_strides: Sequence[int],
    weight_strides: Sequence[int] | None,
    output_shape: Sequence[int],
    output_strides: Sequence[int],
    input_dtype: torch.dtype,
    target_dtype: torch.dtype | None,
    weight_dtype: torch.dtype | None,
    reduction: int,
    ignore_index: int,
    has_weight: bool,
    has_target_tensor: bool,
    target_value: int | None,
    dtype: _CodegenDType,
) -> List[str]:
    nll_template = get_template_env().get_template("nll_loss_kernel.c.j2")
    input_suffix = _format_array_suffix(input_shape)
    target_suffix = _format_array_suffix(target_shape)
    out_suffix = _format_array_suffix(output_shape)
    input_c_type = _dtype_to_c_type(input_dtype, dtype)
    signature_parts = [f"const {input_c_type} input{input_suffix}"]
    target_c_type = None
    if has_target_tensor:
        target_c_type = _dtype_to_c_type(target_dtype, dtype)
        signature_parts.append(f"const {target_c_type} target{target_suffix}")
    weight_c_type = None
    weight_suffix = None
    if has_weight:
        weight_c_type = _dtype_to_c_type(weight_dtype, dtype)
        weight_suffix = _format_array_suffix(weight_shape or ())
        signature_parts.append(f"const {weight_c_type} weight{weight_suffix}")
    signature_parts.append(f"{dtype.c_type} out{out_suffix}")
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"{', '.join(signature_parts)}) {{"
    )

    output_access = KindEmitterBase.emit_output_access(
        output_shape, output_strides, c_type=dtype.c_type
    )
    if not has_target_tensor:
        if target_value is None:
            raise CodegenBackendError("nll_loss requires target_value")
        target_access = str(target_value)
    elif target_shape:
        target_indices = [f"i{dim}" for dim in range(len(target_shape))]
        if reduction == 0 and not output_shape and len(target_shape) == 1:
            target_access = "target[0]"
        else:
            target_access = _emit_strided_access(
                "target",
                target_indices,
                target_strides,
                _is_contiguous(target_shape, target_strides),
                sizes=target_shape,
                c_type=target_c_type or dtype.c_type,
            )
    else:
        target_access = "target[0]"

    input_indices = []
    if len(input_shape) == 1:
        input_indices = ["cls"]
    else:
        for dim in range(len(input_shape)):
            if dim == 1:
                input_indices.append("cls")
            elif dim < 1:
                input_indices.append(f"i{dim}")
            else:
                input_indices.append(f"i{dim - 1}")
    input_access = _emit_strided_access(
        "input",
        input_indices,
        input_strides,
        _is_contiguous(input_shape, input_strides),
        sizes=input_shape,
        c_type=dtype.c_type,
    )

    weight_access = None
    if has_weight:
        weight_access = _emit_strided_access(
            "weight",
            ["cls"],
            weight_strides or (),
            _is_contiguous(weight_shape or (), weight_strides or ()),
            sizes=weight_shape or (),
            c_type=weight_c_type or dtype.c_type,
        )

    rendered = nll_template.render(
        signature=signature,
        output_shape=output_shape,
        target_shape=target_shape,
        output_access=output_access,
        target_access=target_access,
        input_access=input_access,
        weight_access=weight_access,
        reduction=reduction,
        ignore_index=ignore_index,
        has_weight=has_weight,
        has_target_tensor=has_target_tensor,
        zero_literal=_format_scalar_literal(0.0, dtype),
        one_literal=_format_scalar_literal(1.0, dtype),
        nan_literal=f"({_format_scalar_literal(0.0, dtype)} / {_format_scalar_literal(0.0, dtype)})",
        acc_type=dtype.c_type,
        output_rank=len(output_shape),
        target_rank=len(target_shape),
    )
    return rendered.splitlines()


class NllLossEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("nll_loss requires op spec and dtype")
        has_weight = bool(req.params.get("has_weight", False))
        has_target_tensor = bool(req.params.get("has_target_tensor", True))
        target_index = 1 if has_target_tensor else None
        weight_index = None
        if has_weight:
            weight_index = 2 if has_target_tensor else 1
        target_shape = req.input_shapes[target_index] if has_target_tensor else ()
        target_strides = req.input_strides[target_index] if has_target_tensor else ()
        target_dtype = req.input_dtypes[target_index] if has_target_tensor else None
        weight_shape = (
            req.input_shapes[weight_index] if weight_index is not None else None
        )
        weight_strides = (
            req.input_strides[weight_index] if weight_index is not None else None
        )
        weight_dtype = (
            req.input_dtypes[weight_index] if weight_index is not None else None
        )
        return _write_nll_loss_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            target_shape,
            weight_shape,
            req.input_strides[0],
            target_strides,
            weight_strides,
            req.output_shape,
            req.output_strides or (),
            req.input_dtypes[0],
            target_dtype,
            weight_dtype,
            int(req.params["reduction"]),
            int(req.params["ignore_index"]),
            has_weight,
            has_target_tensor,
            req.params.get("target_value"),
            dtype,
        )
