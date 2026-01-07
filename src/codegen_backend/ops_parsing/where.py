from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.fx

from codegen_backend.c_types import _normalize_scalar_value
from codegen_backend.errors import CodegenBackendError
from codegen_backend.specs import _OpSpec


def _error_expected_tensor(op_name: str) -> CodegenBackendError:
    return CodegenBackendError(f"codegen {op_name} expects tensor inputs only")


def _parse_where_inputs(
    op_spec: _OpSpec,
    node: torch.fx.Node,
    shapes: Dict[torch.fx.Node, Tuple[int, ...]],
    scalar_values: Dict[torch.fx.Node, object],
) -> Tuple[List[torch.fx.Node], List[Tuple[int, ...]], Dict[str, object]]:
    if len(node.args) < 3:
        raise CodegenBackendError(f"codegen {op_spec.name} expects three inputs")
    cond_arg, a_arg, b_arg = node.args[:3]
    input_nodes: List[torch.fx.Node] = []
    input_shapes: List[Tuple[int, ...]] = []
    params: Dict[str, object] = {}

    def add_tensor_arg(arg: object) -> None:
        if not isinstance(arg, torch.fx.Node):
            raise _error_expected_tensor(op_spec.name)
        if arg not in shapes:
            raise _error_expected_tensor(op_spec.name)
        input_nodes.append(arg)
        input_shapes.append(shapes[arg])

    def add_where_value(arg: object, scalar_key: str) -> None:
        if isinstance(arg, torch.fx.Node):
            if arg in shapes:
                input_nodes.append(arg)
                input_shapes.append(shapes[arg])
                return
            if arg in scalar_values:
                params[scalar_key] = _normalize_scalar_value(
                    op_spec.name, scalar_values[arg]
                )
                return
            raise _error_expected_tensor(op_spec.name)
        params[scalar_key] = _normalize_scalar_value(op_spec.name, arg)

    add_tensor_arg(cond_arg)
    add_where_value(a_arg, "a_scalar")
    add_where_value(b_arg, "b_scalar")
    return input_nodes, input_shapes, params
