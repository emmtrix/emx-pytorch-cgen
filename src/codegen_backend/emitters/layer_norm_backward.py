from __future__ import annotations

from typing import List, Sequence

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.errors import CodegenBackendError
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_layer_norm_backward_kernel(
    node_index: int,
    op_spec: _OpSpec,
    grad_output_shape: Sequence[int],
    input_shape: Sequence[int],
    mean_shape: Sequence[int],
    rstd_shape: Sequence[int],
    output_shape: Sequence[int],
    normalized_shape: Sequence[int],
    weight_shape: Sequence[int] | None,
    bias_shape: Sequence[int] | None,
    dtype: _CodegenDType,
    has_weight: bool,
    has_bias: bool,
) -> List[str]:
    layer_norm_template = get_template_env().get_template(
        "layer_norm_backward_kernel.c.j2"
    )
    grad_suffix = _format_array_suffix(grad_output_shape)
    input_suffix = _format_array_suffix(input_shape)
    mean_suffix = _format_array_suffix(mean_shape)
    rstd_suffix = _format_array_suffix(rstd_shape)
    output_suffix = _format_array_suffix(output_shape)
    weight_suffix = _format_array_suffix(weight_shape or ())
    bias_suffix = _format_array_suffix(bias_shape or ())
    weight_arg = f"const {dtype.c_type} weight{weight_suffix}, " if has_weight else ""
    bias_arg = f"const {dtype.c_type} bias{bias_suffix}, " if has_bias else ""
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} grad_output{grad_suffix}, "
        f"const {dtype.c_type} input{input_suffix}, "
        f"const {dtype.c_type} mean{mean_suffix}, "
        f"const {dtype.c_type} rstd{rstd_suffix}, "
        f"{weight_arg}"
        f"{bias_arg}"
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    grad_zero_indices = "".join("[0]" for _ in grad_output_shape) or "[0]"
    input_zero_indices = "".join("[0]" for _ in input_shape) or "[0]"
    mean_zero_indices = "".join("[0]" for _ in mean_shape) or "[0]"
    rstd_zero_indices = "".join("[0]" for _ in rstd_shape) or "[0]"
    output_zero_indices = "".join("[0]" for _ in output_shape) or "[0]"
    weight_zero_indices = "".join("[0]" for _ in (weight_shape or ())) or "[0]"
    outer_size = 1
    inner_size = 1
    normalized_rank = len(normalized_shape)
    for dim in input_shape[: len(input_shape) - normalized_rank]:
        outer_size *= dim
    for dim in input_shape[len(input_shape) - normalized_rank :]:
        inner_size *= dim
    rendered = layer_norm_template.render(
        signature=signature,
        outer_size=outer_size,
        inner_size=inner_size,
        c_type=dtype.c_type,
        has_weight=has_weight,
        has_bias=has_bias,
        grad_zero_indices=grad_zero_indices,
        input_zero_indices=input_zero_indices,
        mean_zero_indices=mean_zero_indices,
        rstd_zero_indices=rstd_zero_indices,
        output_zero_indices=output_zero_indices,
        weight_zero_indices=weight_zero_indices,
    )
    return rendered.strip().splitlines()


class LayerNormBackwardEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("layer_norm_backward requires op spec and dtype")
        has_weight = bool(req.params.get("has_weight", False))
        has_bias = bool(req.params.get("has_bias", False))
        weight_shape = req.input_shapes[4] if has_weight else None
        bias_shape = (
            req.input_shapes[5 if has_weight else 4] if has_bias else None
        )
        return _write_layer_norm_backward_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
        req.input_shapes[1],
        req.input_shapes[2],
        req.input_shapes[3],
        req.output_shape,
        tuple(req.params.get("normalized_shape", ())),
        weight_shape,
        bias_shape,
        dtype,
            has_weight,
            has_bias,
        )
