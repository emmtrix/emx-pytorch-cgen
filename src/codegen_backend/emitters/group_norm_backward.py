from __future__ import annotations

from typing import List, Sequence

from codegen_backend.dtypes import _CodegenDType
from codegen_backend.emitters.base import KindEmitterBase, _format_array_suffix
from codegen_backend.errors import CodegenBackendError
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.specs import _OpSpec
from codegen_backend.templates import get_template_env


def _write_group_norm_backward_kernel(
    node_index: int,
    op_spec: _OpSpec,
    grad_output_shape: Sequence[int],
    input_shape: Sequence[int],
    mean_shape: Sequence[int],
    rstd_shape: Sequence[int],
    output_shape: Sequence[int],
    weight_shape: Sequence[int] | None,
    dtype: _CodegenDType,
    groups: int,
    has_weight: bool,
) -> List[str]:
    group_norm_template = get_template_env().get_template(
        "group_norm_backward_kernel.c.j2"
    )
    grad_suffix = _format_array_suffix(grad_output_shape)
    input_suffix = _format_array_suffix(input_shape)
    mean_suffix = _format_array_suffix(mean_shape)
    rstd_suffix = _format_array_suffix(rstd_shape)
    output_suffix = _format_array_suffix(output_shape)
    weight_suffix = _format_array_suffix(weight_shape or ())
    weight_arg = f"const {dtype.c_type} weight{weight_suffix}, " if has_weight else ""
    signature = (
        f"void node{node_index}_{op_spec.name}_{dtype.suffix}("
        f"const {dtype.c_type} grad_output{grad_suffix}, "
        f"const {dtype.c_type} input{input_suffix}, "
        f"const {dtype.c_type} mean{mean_suffix}, "
        f"const {dtype.c_type} rstd{rstd_suffix}, "
        f"{weight_arg}"
        f"{dtype.c_type} out{output_suffix}) {{"
    )
    grad_zero_indices = "".join("[0]" for _ in grad_output_shape) or "[0]"
    input_zero_indices = "".join("[0]" for _ in input_shape) or "[0]"
    mean_zero_indices = "".join("[0]" for _ in mean_shape) or "[0]"
    rstd_zero_indices = "".join("[0]" for _ in rstd_shape) or "[0]"
    output_zero_indices = "".join("[0]" for _ in output_shape) or "[0]"
    weight_zero_indices = "".join("[0]" for _ in (weight_shape or ())) or "[0]"
    batch = input_shape[0]
    channels = input_shape[1]
    spatial_size = 1
    for dim in input_shape[2:]:
        spatial_size *= dim
    rendered = group_norm_template.render(
        signature=signature,
        batch=batch,
        channels=channels,
        spatial_size=spatial_size,
        groups=groups,
        c_type=dtype.c_type,
        has_weight=has_weight,
        grad_zero_indices=grad_zero_indices,
        input_zero_indices=input_zero_indices,
        mean_zero_indices=mean_zero_indices,
        rstd_zero_indices=rstd_zero_indices,
        output_zero_indices=output_zero_indices,
        weight_zero_indices=weight_zero_indices,
    )
    return rendered.strip().splitlines()


class GroupNormBackwardEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("group_norm_backward requires op spec and dtype")
        has_weight = bool(req.params.get("has_weight", False))
        weight_shape = req.input_shapes[4] if has_weight else None
        return _write_group_norm_backward_kernel(
            req.node_index,
            op_spec,
            req.input_shapes[0],
            req.input_shapes[1],
            req.input_shapes[2],
            req.input_shapes[3],
            req.output_shape,
            weight_shape,
            dtype,
            int(req.params.get("groups", 1)),
            has_weight,
        )
