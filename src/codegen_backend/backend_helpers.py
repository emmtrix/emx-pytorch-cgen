from __future__ import annotations

from typing import Dict, List

import torch
import torch.fx

from codegen_backend.analysis_helpers import is_out_overload
from codegen_backend.graph import _OpNode


def _resolve_alias(
    node: torch.fx.Node, alias_map: Dict[torch.fx.Node, torch.fx.Node]
) -> torch.fx.Node:
    while node in alias_map:
        node = alias_map[node]
    return node


def _kernel_inputs(op_node: _OpNode) -> List[torch.fx.Node]:
    if is_out_overload(op_node.node.target) and op_node.inplace_input is not None:
        return [
            arg
            for index, arg in enumerate(op_node.inputs)
            if index != op_node.inplace_input
        ]
    return list(op_node.inputs)
