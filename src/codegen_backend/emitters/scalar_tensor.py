from __future__ import annotations

from typing import List

from codegen_backend.c_types import _format_scalar_literal
from codegen_backend.errors import CodegenBackendError
from codegen_backend.emitters.base import (
    KindEmitterBase,
    _format_array_suffix,
    emit_output_access,
)
from codegen_backend.kinds import KernelEmitRequest
from codegen_backend.templates import get_template_env


class ScalarTensorEmitter(KindEmitterBase):
    def emit(self, req: KernelEmitRequest) -> List[str]:
        op_spec = req.op_spec
        dtype = req.dtype
        if op_spec is None or dtype is None:
            raise CodegenBackendError("scalar_tensor requires op spec and dtype")
        template = get_template_env().get_template(
            "scalar_tensor_kernel.c.j2"
        )
        signature = (
            f"void node{req.node_index}_{op_spec.name}_{dtype.suffix}("
            f"{dtype.c_type} out{_format_array_suffix(req.output_shape)}) {{"
        )
        output_access = emit_output_access(
            req.output_shape, req.output_strides or (), c_type=dtype.c_type
        )
        output_dims = [
            {"dim": dim, "size": size}
            for dim, size in enumerate(req.output_shape)
        ]
        value_literal = _format_scalar_literal(req.params["value"], dtype)
        return template.render(
            signature=signature,
            output_dims=output_dims,
            output_access=output_access,
            value=value_literal,
        ).splitlines()
